"""Tests for download_speed.py.

Network tests use a local HTTP server on 127.0.0.1, the rest replace urlopen
with an in-memory fake, so no internet access is needed.
"""

import http.server
import io
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

import download_speed
from download_speed import CHUNK_SIZE, Download, main, summarize

SCRIPT = Path(download_speed.__file__)
BODY = bytes(range(256)) * 1024  # 256 KiB: several chunks per response
MIB = 1_048_576


# --- helpers ----------------------------------------------------------------


class FakeResponse(io.BytesIO):
    """In-memory HTTP response that records how it was read."""

    def __init__(self, body: bytes) -> None:
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}
        self.read_sizes: list[int | None] = []
        self.read_to_end = False

    def read(self, size: int | None = -1) -> bytes:
        self.read_sizes.append(size)
        data = super().read(size)
        self.read_to_end = self.read_to_end or not data
        return data


class Handler(http.server.BaseHTTPRequestHandler):
    """Counts requests and lets each test choose the response."""

    def do_GET(self) -> None:
        self.server.requests += 1
        self.server.respond(self)

    def log_message(self, format: str, *args: object) -> None:
        pass  # keep the test output clean


def send_body(
    handler: Handler, body: bytes = BODY, length: int | str | None = None
) -> None:
    handler.send_response(200)
    handler.send_header("Content-Length", str(len(body) if length is None else length))
    handler.end_headers()
    handler.wfile.write(body)


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requests to the local server must not go through a system proxy."""
    monkeypatch.setenv("no_proxy", "*")


@pytest.fixture
def server():
    """Local HTTP server; tests replace `server.respond` to change behaviour."""
    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    httpd.handle_error = lambda request, address: None  # clients may hang up early
    httpd.requests = 0
    httpd.respond = send_body
    httpd.url = f"http://127.0.0.1:{httpd.server_port}/file.bin"
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join()


# --- calculations -----------------------------------------------------------


def test_summary_uses_totals():
    downloads = [Download(1 * MIB, 0.25)] * 5 + [Download(5 * MIB, 2.75)] * 5

    summary = summarize(downloads)

    assert summary.total_bytes == 30 * MIB
    assert summary.total_seconds == pytest.approx(15.0)
    assert summary.average_seconds == pytest.approx(1.5)  # 15 s / 10 requests
    # Total bytes / total time, not the mean of per-request speeds (~2.9 MiB/s).
    assert summary.mib_per_second == pytest.approx(2.0)  # 30 MiB / 15 s
    assert summary.megabits_per_second == pytest.approx(16.777216)  # 30*2**20*8/15/1e6


# --- one download and the series (fake urlopen) ------------------------------


def test_download_time_covers_request_and_whole_body(monkeypatch):
    clock = [100.0]
    body = b"x" * (2 * CHUNK_SIZE)  # two full chunks, then an empty read at the end

    class SlowResponse(FakeResponse):
        def read(self, size: int | None = -1) -> bytes:
            clock[0] += 0.25  # each read takes a quarter of a second
            return super().read(size)

    def fake_urlopen(request, timeout):
        clock[0] += 1.0  # connecting and waiting for the headers
        return SlowResponse(body)

    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = download_speed.download("http://example.test/file", timeout=5)

    assert result.size == len(body)
    assert result.seconds == pytest.approx(1.0 + 3 * 0.25)


def test_series_is_sequential_and_reads_bodies_in_chunks(monkeypatch, capsys):
    body = b"x" * (3 * CHUNK_SIZE + 123)
    responses: list[FakeResponse] = []

    def fake_urlopen(request, timeout):
        # The previous response must be read to the end and closed first.
        assert all(r.read_to_end and r.closed for r in responses)
        responses.append(FakeResponse(body))
        return responses[-1]

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert main(["http://example.test/file"]) == 0

    assert len(responses) == 10
    assert all(r.read_to_end and r.closed for r in responses)
    assert {size for r in responses for size in r.read_sizes} == {CHUNK_SIZE}
    out = capsys.readouterr().out
    assert f"Total downloaded:     {10 * len(body)} bytes" in out


# --- command line with a local HTTP server ----------------------------------


def test_cli_downloads_ten_times_and_prints_speed(server):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), server.url],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert server.requests == 10
    assert result.stdout.count(f": {len(BODY)} bytes in ") == 10
    assert f"Total downloaded:     {10 * len(BODY)} bytes" in result.stdout
    assert "Average request time:" in result.stdout
    assert "MiB/s" in result.stdout
    assert result.stderr == ""


def test_http_error_stops_without_retries(server, capsys):
    server.respond = lambda handler: handler.send_error(404)

    assert main([server.url]) == 1

    assert server.requests == 1
    out, err = capsys.readouterr()
    assert "error: request 1/10 failed: HTTP 404" in err
    assert "Download speed" not in out


def test_failure_in_the_middle_gives_no_result(server, capsys):
    def respond(handler: Handler) -> None:
        if handler.server.requests <= 3:
            send_body(handler)
        else:
            handler.send_error(500)

    server.respond = respond

    assert main([server.url]) == 1

    out, err = capsys.readouterr()
    assert out.count(" bytes in ") == 3
    assert "Download speed" not in out
    assert "error: request 4/10 failed: HTTP 500" in err
    assert "no result: the series stopped after 3 of 10 downloads" in err


def test_body_shorter_than_content_length_is_an_error(server, capsys):
    server.respond = lambda handler: send_body(handler, BODY[:1000], len(BODY))

    assert main([server.url]) == 1

    err = capsys.readouterr().err
    assert f"connection closed after 1000 of {len(BODY)} bytes" in err


def test_malformed_content_length_is_ignored(server):
    server.respond = lambda handler: send_body(handler, length="²")  # "²"

    assert main([server.url]) == 0


def test_timeout(server, capsys):
    def respond_late(handler: Handler) -> None:
        time.sleep(0.5)
        send_body(handler)

    server.respond = respond_late

    assert main([server.url, "--timeout", "0.1"]) == 1

    assert "timed out: no response for 0.1 s" in capsys.readouterr().err


def test_connection_refused(capsys):
    with socket.socket() as sock:  # find a free port, then close it
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    assert main([f"http://127.0.0.1:{port}/file.bin"]) == 1

    assert "error: request 1/10 failed:" in capsys.readouterr().err


def test_empty_responses_are_an_error(server, capsys):
    server.respond = lambda handler: send_body(handler, b"")

    assert main([server.url]) == 1

    assert server.requests == 10
    assert "all responses were empty" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["example.com/file.bin"],
        ["ftp://example.com/file.bin"],
        ["http://"],
        ["http://example.com:99999/file.bin"],
        ["http://example..com/file.bin"],
        ["http://пример.рф/file.bin"],
        ["http://example.com/file.bin", "--timeout", "0"],
        ["http://example.com/file.bin", "--timeout", "abc"],
    ],
)
def test_invalid_arguments_exit_with_code_2(args, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(args)

    assert exit_info.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_url_with_spaces_and_non_ascii_path_is_percent_encoded():
    url = download_speed.http_url("http://example.com/фото 1.jpg?size=big")

    assert url == "http://example.com/%D1%84%D0%BE%D1%82%D0%BE%201.jpg?size=big"


def test_help(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "url" in out
    assert "--timeout" in out
    assert "MiB/s" in out
