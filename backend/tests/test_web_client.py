"""Serving the built web client from the API's server (app/api/web_client.py)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.web_client import ASSET_CACHE, PAGE_CACHE, WebClient, built_client
from app.config import Settings, get_settings
from app.main import create_app

INDEX = "<!doctype html><title>RecruitFlow AI</title><div id=root></div>"


@pytest.fixture
def dist(tmp_path: Path) -> Path:
    """A stand-in for frontend/dist, and a file beside it that must never be served."""
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(INDEX, encoding="utf-8")
    (root / "assets" / "index-3f9a1c.js").write_text("console.log(1)", encoding="utf-8")
    (root / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("not for the web", encoding="utf-8")
    return root


@pytest.fixture
def web(dist: Path) -> Iterator[TestClient]:
    settings = get_settings().model_copy(update={"web_client_dir": str(dist)})
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.mark.parametrize("path", ["/", "/jobs", "/jobs/12", "/candidates/3", "/index.html"])
def test_page_addresses_get_the_client(web: TestClient, path: str) -> None:
    response = web.get(path)

    assert response.status_code == 200
    assert response.text == INDEX
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == PAGE_CACHE


def test_built_files_are_sent_as_they_are(web: TestClient) -> None:
    script = web.get("/assets/index-3f9a1c.js")
    icon = web.get("/favicon.svg")

    assert script.status_code == 200
    assert script.text == "console.log(1)"
    assert script.headers["cache-control"] == ASSET_CACHE
    assert icon.text == "<svg/>"
    assert icon.headers["cache-control"] == PAGE_CACHE


def test_head_is_answered_like_get(web: TestClient) -> None:
    assert web.head("/jobs/12").status_code == 200


def test_the_api_and_its_docs_still_answer_first(web: TestClient) -> None:
    assert web.get("/api/v1/health").json()["database"] == "ok"
    assert web.get("/openapi.json").json()["info"]["title"] == "RecruitFlow AI"
    assert "swagger" in web.get("/docs").text.lower()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/v1/nope"),
        ("GET", "/api/v1"),
        ("POST", "/api/v1/nope"),
        ("GET", "/assets/missing.js"),
        ("GET", "/robots.txt"),
        ("POST", "/jobs/12"),
        ("DELETE", "/"),
    ],
)
def test_what_is_not_a_page_stays_a_json_404(web: TestClient, method: str, path: str) -> None:
    response = web.request(method, path)

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_nothing_outside_the_build_is_served(dist: Path) -> None:
    app = WebClient(dist, "/api/v1")

    assert app._file("../secret.txt") is None
    assert app._file("/assets/../../secret.txt") is None
    assert app._file("assets") is None  # a folder, not a file
    assert app._file("/favicon.svg") == (dist / "favicon.svg").resolve()


def test_an_encoded_climb_out_of_the_build_is_a_page_not_a_file(web: TestClient) -> None:
    response = web.get("/..%2Fsecret.txt")

    assert "not for the web" not in response.text


def test_a_websocket_is_left_unanswered(dist: Path) -> None:
    sent: list[object] = []

    async def receive() -> dict[str, str]:
        return {"type": "websocket.connect"}

    async def send(message: object) -> None:
        sent.append(message)

    scope = {"type": "websocket", "path": "/jobs"}
    asyncio.run(WebClient(dist, "/api/v1")(scope, receive, send))

    assert sent == []


def test_without_a_build_nothing_is_mounted(tmp_path: Path) -> None:
    settings = get_settings().model_copy(update={"web_client_dir": str(tmp_path)})
    with TestClient(create_app(settings)) as client:
        assert client.get("/").status_code == 404
        assert client.get("/jobs/12").status_code == 404


def test_an_empty_setting_turns_serving_off(dist: Path) -> None:
    assert Settings(web_client_dir="").resolved_web_client_dir is None
    assert built_client(None) is None
    assert built_client(dist.parent) is None  # no index.html there
    assert built_client(dist) == dist


def test_a_relative_setting_is_under_the_repository() -> None:
    resolved = Settings(web_client_dir="frontend/dist").resolved_web_client_dir

    assert resolved is not None
    assert resolved.parts[-2:] == ("frontend", "dist")
    assert resolved.is_absolute()
