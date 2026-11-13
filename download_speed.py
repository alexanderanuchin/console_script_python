#!/usr/bin/env python3
"""Measure HTTP download speed for a URL.

The URL is downloaded 10 times in a row. Every response body is read to the
end, then the script prints the average request time, the total amount of
downloaded data and the download speed. Only the standard library is used.
"""

import argparse
import http.client
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

REQUEST_COUNT = 10
CHUNK_SIZE = 64 * 1024  # bytes per read() call
DEFAULT_TIMEOUT = 10.0  # seconds
MAX_TIMEOUT = 3600.0  # larger values overflow the socket timeout on some platforms
BYTES_PER_MIB = 1_048_576  # 2**20
BITS_PER_MEGABIT = 1_000_000

REQUEST_HEADERS = {
    # Ask for the body as is: the script counts the bytes that actually arrive.
    "Accept-Encoding": "identity",
    # Some servers reject the default "Python-urllib/3.x" user agent.
    "User-Agent": "download-speed/1.0",
}


class DownloadError(Exception):
    """A request failed or its response body arrived incomplete."""


@dataclass(frozen=True)
class Download:
    size: int  # bytes of the response body that were actually read
    seconds: float  # from the start of the request to the end of the body


@dataclass(frozen=True)
class Summary:
    total_bytes: int
    total_seconds: float
    average_seconds: float
    mib_per_second: float
    megabits_per_second: float


def download(url: str, timeout: float) -> Download:
    """Send one GET request and read the whole response body in chunks."""
    request = urllib.request.Request(url, headers=REQUEST_HEADERS)
    started = time.perf_counter()  # monotonic clock with high resolution
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            size = 0
            while chunk := response.read(CHUNK_SIZE):
                size += len(chunk)
            seconds = time.perf_counter() - started
            declared = response.headers.get("Content-Length")
    except urllib.error.HTTPError as error:
        error.close()  # release the connection held by the error response
        raise DownloadError(f"HTTP {error.code} {error.reason}") from error
    except (OSError, http.client.HTTPException) as error:
        raise DownloadError(describe_error(error, timeout)) from error

    # If the connection breaks early, read() returns b"" instead of raising,
    # so compare the bytes read with the size announced by the server.
    if declared is not None and declared.isdigit() and int(declared) != size:
        raise DownloadError(f"connection closed after {size} of {declared} bytes")
    return Download(size, seconds)


def describe_error(error: Exception, timeout: float) -> str:
    """Return a short human-readable reason for a failed request."""
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    if isinstance(reason, TimeoutError):
        return f"timed out: no response for {timeout:g} s"
    if isinstance(reason, http.client.IncompleteRead):
        return "connection closed before the whole body was received"
    return str(reason) or type(reason).__name__


def summarize(downloads: Sequence[Download]) -> Summary:
    """Aggregate a complete series: the speed is total bytes / total time."""
    total_bytes = sum(item.size for item in downloads)
    total_seconds = sum(item.seconds for item in downloads)
    return Summary(
        total_bytes=total_bytes,
        total_seconds=total_seconds,
        average_seconds=total_seconds / len(downloads),
        mib_per_second=total_bytes / total_seconds / BYTES_PER_MIB,
        megabits_per_second=total_bytes * 8 / total_seconds / BITS_PER_MEGABIT,
    )


def http_url(value: str) -> str:
    """Argument type for the URL: only absolute http(s) URLs are accepted."""
    try:
        parts = urllib.parse.urlsplit(value)
        _ = parts.port  # raises ValueError for a malformed or out-of-range port
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid URL {value!r}: {error}") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise argparse.ArgumentTypeError(
            f"invalid URL {value!r}: expected http://host/... or https://host/..."
        )
    return value


def timeout_seconds(value: str) -> float:
    """Argument type for --timeout: seconds, above zero and not too large."""
    try:
        seconds = float(value)
    except ValueError:
        seconds = math.nan
    if not 0 < seconds <= MAX_TIMEOUT:  # also rejects nan
        raise argparse.ArgumentTypeError(
            f"expected seconds above 0 and at most {MAX_TIMEOUT:g}, got {value!r}"
        )
    return seconds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Download URL {REQUEST_COUNT} times in a row, reading every response "
            "to the end, and print the average request time, the total "
            "downloaded size and the download speed."
        ),
        epilog=(
            f"Speed = total bytes / total time of the {REQUEST_COUNT} requests, "
            "shown in MiB/s (1 MiB = 1048576 bytes) and Mbit/s "
            "(1 Mbit = 1000000 bits). Exit status: 0 on success, 1 if a request "
            "fails, 2 on invalid arguments."
        ),
    )
    parser.add_argument(
        "url",
        type=http_url,
        help="http:// or https:// address of a file to download, e.g. a large "
        "image; files of 10 MB or more give a more accurate result",
    )
    parser.add_argument(
        "--timeout",
        type=timeout_seconds,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help="limit for connecting and for each wait for data, not for the whole "
        f"download (default: %(default)g, at most {MAX_TIMEOUT:g})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"Downloading {args.url} {REQUEST_COUNT} times in a row")

    downloads: list[Download] = []
    try:
        for number in range(1, REQUEST_COUNT + 1):
            result = download(args.url, args.timeout)
            downloads.append(result)
            print(
                f"  request {number:2}/{REQUEST_COUNT}: "
                f"{result.size} bytes in {result.seconds:.3f} s",
                flush=True,
            )
    except DownloadError as error:
        done = len(downloads)
        print(
            f"error: request {done + 1}/{REQUEST_COUNT} failed: {error}\n"
            f"no result: the series stopped after {done} of {REQUEST_COUNT} downloads",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        done = len(downloads)
        print(
            f"\ninterrupted: no result after {done} of {REQUEST_COUNT} downloads",
            file=sys.stderr,
        )
        return 130

    summary = summarize(downloads)
    if summary.total_bytes == 0:
        print(
            "error: all responses were empty, there is nothing to measure",
            file=sys.stderr,
        )
        return 1

    total_mib = summary.total_bytes / BYTES_PER_MIB
    print()
    print(f"Average request time: {summary.average_seconds:.3f} s")
    print(f"Total downloaded:     {summary.total_bytes} bytes ({total_mib:.2f} MiB)")
    print(
        f"Download speed:       {summary.mib_per_second:.2f} MiB/s "
        f"({summary.megabits_per_second:.2f} Mbit/s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
