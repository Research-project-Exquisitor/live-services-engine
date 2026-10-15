# Copyright (C) 2026 Ujjwal Sharma and Omar Shahbaz Khan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""Diagnostic logging setup (Loguru).

This is the developer-facing log: what the server is doing and why it failed.
It is separate from the research event log (`app.services.event_log`), which
never depends on a log level.

Level policy:

- ``ERROR``: failures, always with the traceback (``logger.exception``).
- ``INFO``: lifecycle and milestones (startup, models loaded, search handled).
  The default level.
- ``DEBUG``: internal details while developing.
- ``TRACE``: entry and exit of recursive methods.
- ``WARNING`` is not used by our own code.

Messages from the standard ``logging`` module (uvicorn, FastAPI, libraries)
are forwarded into Loguru so everything shares one format.
"""

import logging
import sys

from loguru import logger

LOG_FORMAT: str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <5}</level> | "
    "session={extra[session]} req={extra[request_id]} | "
    "<cyan>{name}:{line}</cyan> - <level>{message}</level>"
)
"""Line format; ``session`` and ``request_id`` come from `logger.contextualize`."""


class InterceptHandler(logging.Handler):
    """Forward standard-library log records to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk back to the frame that made the logging call, so Loguru
        # reports the caller's module and line rather than this handler's.
        frame, depth = logging.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(level: str = "INFO") -> None:
    """Configure Loguru as the single diagnostic log sink.

    Args:
        level: Minimum level to print (``TRACE``, ``DEBUG``, ``INFO`` or ``ERROR``).
    """
    logger.remove()
    logger.configure(extra={"session": "-", "request_id": "-"})
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=LOG_FORMAT,
        backtrace=False,
        # diagnose=True would print local variable values in tracebacks,
        # which can include participant input.
        diagnose=False,
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        std_logger = logging.getLogger(name)
        std_logger.handlers = []
        std_logger.propagate = True

    # httpx logs every model-download check at INFO; keep only its problems.
    logging.getLogger("httpx").setLevel(logging.WARNING)
