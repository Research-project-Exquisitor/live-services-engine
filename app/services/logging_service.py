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


"""Legacy audit log for clients that still use the old logging routes.

`AuditLogger` is a low-level async writer. Log entries are pushed onto an
`asyncio.Queue` and consumed by a background worker task that serializes each
entry as MessagePack and appends it to a log file under a `FileLock` for safe
concurrent access. A synchronous ``log_sync`` method is also available for
non-async contexts.

Only ``/log/addModel``, ``/log/removeModel`` and ``/log/clientEvent`` write
here, so UIs that don't send research events keep working as before. Research
events go to `app.services.research_log.ResearchLog`.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, Optional

import msgpack
from filelock import FileLock
from loguru import logger


class AuditLogger:
    """Handles audit logging with structured format and async processing."""

    def __init__(self, log_file: str):
        self.log_file: Path = Path(log_file)
        """Path to the MessagePack-encoded audit log file."""

        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.lockfile: str = f"{self.log_file}.lock"
        """Path to the file-level lock used for concurrent write safety."""

        self._queue: asyncio.Queue = asyncio.Queue()
        """Async queue buffering log entries for the background writer."""

        self._worker_task: Optional[asyncio.Task] = None
        """Handle to the background asyncio task consuming the log queue."""

    async def start(self):
        """Start the async logging worker."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._log_worker())

    async def stop(self):
        """Stop the async logging worker."""
        if self._worker_task:
            await self._queue.put(None)  # Sentinel to stop worker
            await self._worker_task
            self._worker_task = None

    async def _log_worker(self):
        """Background worker that processes log messages."""
        while True:
            try:
                log_entry = await self._queue.get()
                if log_entry is None:  # Stop sentinel
                    break
                self._write_log_sync(log_entry)
                self._queue.task_done()
            except Exception:
                # Log to the diagnostic log if audit logging fails
                logger.exception("Failed to write audit log")

    def _write_log_sync(self, log_entry: Dict[str, Any]):
        """Synchronously write log entry to file."""
        lock = FileLock(self.lockfile)
        with lock:
            with open(self.log_file, "ab") as f:
                packer = msgpack.Packer()
                packed_data = packer.pack(log_entry)
                f.write(packed_data)

    async def log(
        self,
        action: str,
        session: str,
        data: Dict[str, Any],
        request_timestamp: Optional[int] = None,
        completion_time: Optional[int] = None,
    ):
        """Log an audit entry asynchronously."""
        timestamp = request_timestamp or int(time.time())

        log_entry = {
            "timestamp": timestamp,
            "session": session,
            "action": action,
            "data": data,
        }

        if completion_time:
            log_entry["completion_time"] = completion_time

        await self._queue.put(log_entry)

    def log_sync(
        self,
        action: str,
        session: str,
        data: Dict[str, Any],
        request_timestamp: Optional[int] = None,
        completion_time: Optional[int] = None,
    ):
        """Log an audit entry synchronously (for non-async contexts)."""
        timestamp = request_timestamp or int(time.time())

        log_entry = {
            "timestamp": timestamp,
            "session": session,
            "action": action,
            "data": data,
        }

        if completion_time:
            log_entry["completion_time"] = completion_time

        self._write_log_sync(log_entry)

