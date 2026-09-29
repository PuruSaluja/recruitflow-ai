"""The app's log lines: format, level from settings, and leaving other loggers alone."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.logging_setup import CurrentStderrHandler
from app.main import create_app

LINE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} (\w+) ([\w.]+): (.*)$")


@pytest.fixture(autouse=True)
def _restore_app_logger() -> Iterator[None]:
    """Put the ``app`` logger's level back, since the session's app shares it."""
    logger = logging.getLogger("app")
    level = logger.level
    yield
    logger.setLevel(level)


def _app_with(level: str) -> None:
    create_app(get_settings().model_copy(update={"log_level": level}))


def _lines(captured: str) -> list[tuple[str, str, str]]:
    return [match.groups() for line in captured.splitlines() if (match := LINE.match(line))]


def test_a_record_names_its_time_level_and_logger(capsys: pytest.CaptureFixture[str]) -> None:
    # Built outside the capture, so this also shows the handler writes to the
    # stderr of the moment rather than the one it was built with.
    with capsys.disabled():
        _app_with("INFO")

    logging.getLogger("app.main").warning("refused %s", "row 7")

    assert _lines(capsys.readouterr().err) == [("WARNING", "app.main", "refused row 7")]


def test_the_level_comes_from_the_settings(capsys: pytest.CaptureFixture[str]) -> None:
    logger = logging.getLogger("app.main")

    _app_with("WARNING")
    logger.info("dropped")
    logger.warning("kept")
    _app_with("DEBUG")
    logger.debug("now kept")

    assert [message for _, _, message in _lines(capsys.readouterr().err)] == ["kept", "now kept"]


def test_building_the_app_again_adds_no_second_handler(capsys: pytest.CaptureFixture[str]) -> None:
    _app_with("INFO")
    _app_with("INFO")

    logging.getLogger("app").warning("once")

    assert len(_lines(capsys.readouterr().err)) == 1
    handlers = logging.getLogger("app").handlers
    assert sum(isinstance(handler, CurrentStderrHandler) for handler in handlers) == 1


def test_other_loggers_are_left_as_they_were() -> None:
    others = [logging.getLogger(), logging.getLogger("uvicorn.error")]
    before = [(logger.level, list(logger.handlers), logger.disabled) for logger in others]

    _app_with("DEBUG")

    assert [(logger.level, list(logger.handlers), logger.disabled) for logger in others] == before
    assert not logging.getLogger("app.api.server_errors").disabled
    assert logging.getLogger("app").propagate


def test_the_level_setting_ignores_case_and_refuses_unknown_levels() -> None:
    assert Settings(log_level="warning").log_level == "WARNING"
    with pytest.raises(ValidationError):
        Settings(log_level="LOUD")
