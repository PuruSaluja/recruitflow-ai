"""The app behind a real uvicorn server, for what only a socket shows.

The test client hands the app its request in memory, so it cannot show what a
client on the other end of a TCP connection receives. Closing a socket that
still has unread data resets the connection, and a reset discards whatever
answer the client had not read yet.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
from collections.abc import Iterator

import pytest
import uvicorn
from fastapi import FastAPI

from app.api.body_limit import MAX_REQUEST_BYTES
from app.config import get_settings

BOUNDARY = "liveboundary"
CHUNK = 64 * 1024


@pytest.fixture(params=["h11", "httptools"])
def live_port(app: FastAPI, request: pytest.FixtureRequest) -> Iterator[int]:
    """Serve ``app`` with uvicorn on a free local port, with each HTTP parser."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    config = uvicorn.Config(app, http=request.param, ws="none", lifespan="off", log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            threading.Event().wait(0.025)
        assert server.started, "uvicorn did not start"
        yield listener.getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()


def _upload(size: int) -> list[bytes]:
    head = (
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="file"; filename="cv.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
    ).encode()
    body = [b"a" * min(CHUNK, size - start) for start in range(0, size, CHUNK)]
    return [head, *body, f"\r\n--{BOUNDARY}--\r\n".encode()]


def _post(port: int, parts: list[bytes], *, chunked: bool) -> tuple[int, dict]:
    """Send the whole body, as a client that only reads once it is done does."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    conn.putrequest("POST", f"{get_settings().api_prefix}/candidates/resume-preview")
    conn.putheader("Content-Type", f"multipart/form-data; boundary={BOUNDARY}")
    if chunked:
        conn.putheader("Transfer-Encoding", "chunked")
    else:
        conn.putheader("Content-Length", str(sum(map(len, parts))))
    conn.endheaders()
    for part in parts:
        conn.send(f"{len(part):X}\r\n".encode() + part + b"\r\n" if chunked else part)
    if chunked:
        conn.send(b"0\r\n\r\n")
    response = conn.getresponse()
    answer = json.loads(response.read())
    conn.close()
    return response.status, answer


@pytest.mark.parametrize("chunked", [False, True], ids=["declared", "chunked"])
def test_a_client_still_sending_a_body_over_the_cap_can_read_the_413(
    live_port: int, chunked: bool
) -> None:
    status, answer = _post(live_port, _upload(3 * 1024 * 1024), chunked=chunked)

    assert MAX_REQUEST_BYTES < 3 * 1024 * 1024
    assert status == 413
    assert answer["code"] == "request_too_large"


def test_a_client_waiting_for_100_continue_gets_the_413_without_sending(live_port: int) -> None:
    """Asking for the body would make it send, and then wait for an answer it never gets."""
    conn = http.client.HTTPConnection("127.0.0.1", live_port, timeout=20)
    conn.putrequest("POST", f"{get_settings().api_prefix}/candidates/resume-preview")
    conn.putheader("Content-Type", f"multipart/form-data; boundary={BOUNDARY}")
    conn.putheader("Content-Length", str(3 * 1024 * 1024))
    conn.putheader("Expect", "100-continue")
    conn.endheaders()

    response = conn.getresponse()

    assert response.status == 413
    assert json.loads(response.read())["code"] == "request_too_large"
    conn.close()
