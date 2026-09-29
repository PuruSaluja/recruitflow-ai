"""The app's log output: one line per record, naming its time, level and logger.

Without this, nothing gave the ``app`` loggers a handler, so their warnings and
errors reached stderr only through Python's last-resort handler (the message
and traceback, with no time, level or logger name) and anything below WARNING
was dropped. Under uvicorn those lines sat untimestamped among its own.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class CurrentStderrHandler(logging.StreamHandler):
    """Writes to ``sys.stderr`` as it is when each record arrives.

    A plain ``StreamHandler`` keeps the stream it was built with, so one built
    while pytest was capturing output would later write to a closed capture.
    """

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property  # type: ignore[override]
    def stream(self) -> TextIO:
        return sys.stderr

    @stream.setter
    def stream(self, value: TextIO) -> None:
        """Ignored: the stream is always the current ``sys.stderr``."""


def configure_logging(level: str) -> None:
    """Send the ``app`` loggers' records at ``level`` and above to stderr.

    Only the ``app`` logger is touched, and its handler is added once however
    many apps are built. Every other logger is left as it is: uvicorn keeps its
    own configuration, and nothing is disabled, as ``logging.config`` would do
    to loggers created before it. Records still propagate, so pytest's capture
    sees them; a deployment that also gives the root logger a handler would
    print them twice.
    """
    logger = logging.getLogger("app")
    logger.setLevel(level)
    if not any(isinstance(handler, CurrentStderrHandler) for handler in logger.handlers):
        handler = CurrentStderrHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
