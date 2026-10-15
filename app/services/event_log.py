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


"""Research event log: one JSON line per `LogEvent`, one file per session.

Events are validated by `LogEvent` before they get here. `EventLog.write`
stamps ``time.server`` the moment an event is received (not when it is
flushed), then a background worker appends it to
``<directory>/session-<session_id>.jsonl``. Requests never wait on disk.

This log does not depend on the diagnostic log level: every event that is
written here is kept.
"""

import asyncio
import time
from pathlib import Path
from typing import Optional

from filelock import FileLock
from loguru import logger

from ..schemas.logging import LogEvent


def now_ms() -> int:
    """Current Unix time in milliseconds."""
    return time.time_ns() // 1_000_000


class EventLog:
    """Append-only JSONL writer for research events."""

    def __init__(self, directory: str):
        self.directory: Path = Path(directory)
        """Folder holding one ``session-<id>.jsonl`` file per session."""

        self.directory.mkdir(parents=True, exist_ok=True)

        self._lock: FileLock = FileLock(self.directory / ".write.lock")
        """Guards appends if more than one server process writes here."""

        self._queue: asyncio.Queue[Optional[LogEvent]] = asyncio.Queue()
        """Events waiting for the background writer; ``None`` stops it."""

        self._worker_task: Optional[asyncio.Task] = None
        """Handle to the background writer task."""

    def path_for(self, session_id: str) -> Path:
        """File that holds all events of `session_id`."""
        return self.directory / f"session-{session_id}.jsonl"

    async def start(self) -> None:
        """Start the background writer."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Write all queued events, then stop the background writer."""
        if self._worker_task:
            await self._queue.put(None)
            await self._worker_task
            self._worker_task = None

    async def write(self, event: LogEvent) -> None:
        """Stamp the server receive time and queue `event` for writing."""
        if event.time.server is None:
            event = event.model_copy(
                update={"time": event.time.model_copy(update={"server": now_ms()})}
            )
        await self._queue.put(event)

    async def _worker(self) -> None:
        while True:
            event = await self._queue.get()
            if event is None:
                break
            try:
                self._append(event)
            except Exception:
                logger.exception(
                    "Failed to write event {} for session {}",
                    event.event_id,
                    event.session_id,
                )
            finally:
                self._queue.task_done()

    def _append(self, event: LogEvent) -> None:
        path = self.path_for(event.session_id)
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")
