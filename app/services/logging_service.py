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


"""Centralized logging service for audit trails and system logs.

This module contains the logging facade and the legacy audit writer. During
Phase 3, migrated operations also use the structured research event writer:

`AuditLogger`
    Low-level async writer.  Log entries are pushed onto an `asyncio.Queue`
    and consumed by a background worker task that serializes each entry as
    MessagePack and appends it to a log file under a `FileLock` for safe
    concurrent access.  A synchronous ``log_sync`` method is also available
    for non-async contexts.

`LoggingService`
    Facade consumed by route handlers and services. Migrated operations build
    `LogEvent` envelopes and queue them through `EventLogWriter`. Legacy operations
    continue using `AuditLogger` during Phase 3. Request-bound instances share
    the writers without sharing request context.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import msgpack
from filelock import FileLock
from loguru import logger

from ..schemas.logging import ClientLogEvent, EventTarget, EventType, LogEvent
from .event_log_writer import EventLogWriter


# Old research writer. Phase 3 replaces it with `EventLogWriter` (event_log_writer.py)
# and removes this class, msgpack and logs/audit.log.
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


# Stays in Phase 3, but its methods will build `LogEvent`s and write them to
# `EventLogWriter` instead of free-text entries to `AuditLogger`.
class LoggingService:
    """Service for managing all logging operations."""

    def __init__(
        self,
        audit_logger: AuditLogger,
        event_log_writer: EventLogWriter | None = None,
        *,
        request_id: str | None = None,
        sequence: int | None = None,
    ):
        self.audit_logger: AuditLogger = audit_logger
        """Low-level audit logger that serializes entries to disk."""
        self.event_log_writer = event_log_writer
        self._request_id = request_id
        self._sequence = sequence

    def for_request(self, request_id: str | None, sequence: int | None) -> "LoggingService":
        """Bind context to a new facade, never to the shared application instance."""
        return LoggingService(
            self.audit_logger, self.event_log_writer, request_id=request_id, sequence=sequence
        )

    async def log_event(
        self,
        event_type: EventType,
        target: EventTarget,
        session: str,
        data: Dict[str, Any],
    ) -> None:
        """Build and queue a server event using this request's causal context.

        Old clients have no sequence; leave them on legacy logging rather than
        inventing event order. Research logging failures remain diagnostic and
        must not replace a successful response or the original application error.
        """
        if self.event_log_writer is None or self._sequence is None or not self._request_id:
            return
        try:
            event = LogEvent(
                source="server", session_id=session, sequence=self._sequence,
                type=event_type, target=target, time={},
                request_id=self._request_id, data=data,
            )
            await self.event_log_writer.write(event)
        except Exception:
            logger.exception("Failed to queue research event {} for session {}", event_type, session)

    async def log_client_events(self, events: list[ClientLogEvent]) -> int:
        """Queue validated UI events, preserving their original IDs and context.

        Intake errors propagate, so the endpoint never reports acceptance for
        an event it failed to queue. Acceptance still does not imply disk durability.
        """
        if self.event_log_writer is None:
            raise RuntimeError("Research event writer is not configured")
        for event in events:
            await self.event_log_writer.write(event)
        return len(events)


    async def log_session_init(self, session: str, collections: list):
        """Log session initialization."""
        await self.audit_logger.log(
            action="Initialize Exquisitor LSE Session",
            session=session,
            data={"session": session, "collections": collections},
        )

    async def log_total_items_request(
        self, session: str, collection: str, total_items: int
    ):
        """Log total items request."""
        await self.audit_logger.log(
            action="Get Total Items",
            session=session,
            data={
                "collection": collection,
                "total_items": total_items,
            },
        )

    async def log_search_request(
        self,
        action: str,
        session: str,
        model_id: int,
        collection: str,
        query_data: Dict[str, Any],
        suggestions: list,
        request_timestamp: int,
        completion_time: int,
        *,
        request_data: Dict[str, Any] | None = None,
        execution: Dict[str, Any] | None = None,
    ):
        """Log any search family, with structured data for migrated callers.

        Request and result payloads can contain text, RF inputs or temporal
        chains. The facade does not depend on a particular search request model.
        """
        if request_data is not None and execution is not None:
            await self.log_event("query_execution", "search", session, {
                # Keep execution timing and strategy; legacy timing fields
                # belong exclusively to the old audit representation.
                **{key: value for key, value in execution.items()
                   if key not in {"request_timestamp", "completion_time", "suggestions"}},
                "request": request_data,
                "suggestions": suggestions,
            })
        await self.audit_logger.log(
            action=action,
            session=session,
            data={
                "session": session,
                "modelId": model_id,
                "collection": collection,
                "suggestions": suggestions,
                **query_data,
            },
            request_timestamp=request_timestamp,
            completion_time=completion_time,
        )

    async def log_model_operation(
        self, action: str, session: str, model_id: int, collection: str, body_json: str
    ):
        """Log model add/remove operations."""
        await self.audit_logger.log(
            action=action,
            session=session,
            data={
                "session": session,
                "modelId": model_id,
                "collection": collection,
                "body": body_json,
            },
        )

    async def log_filter_operation(
        self,
        action: str,
        session: str,
        model_id: int,
        collection: str,
        filter_data: Dict[str, Any],
    ):
        """Log filter operations."""
        await self.audit_logger.log(
            action=action,
            session=session,
            data={
                "session": session,
                "modelId": model_id,
                "collection": collection,
                **filter_data,
            },
        )

    async def log_item_request(
        self,
        action: str,
        session: str,
        model_id: int,
        collection: str,
        item_id: Union[int, List[int]],
        item_name: Union[str, List[str]],
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """Log item-related requests.

        A batch request passes lists of IDs and names instead of single values.
        """
        data = {
            "session": session,
            "modelId": model_id,
            "collection": collection,
            "item": item_id,
            "mediaName": item_name,
        }
        if additional_data:
            data.update(additional_data)

        await self.audit_logger.log(action=action, session=session, data=data)

    async def log_error(
        self, error: Exception, session: str, context: Dict[str, Any],
        *, target: EventTarget = "session",
    ):
        """Record an error in event/audit logs and diagnostics with its traceback."""
        await self.log_event("error", target, session, {
            "message": str(error),
            "error_type": type(error).__name__,
            "context": context,
        })
        try:
            await self.audit_logger.log(
                action="Error",
                session=session,
                data={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "error_details": getattr(error, "details", {}),
                    "context": context,
                },
            )
        except Exception:
            logger.exception("Failed to queue error in legacy audit log")

        # Also log to the diagnostic log
        logger.opt(exception=error).error(
            "{}: {} (session={}, context={}, details={})",
            type(error).__name__,
            error,
            session,
            context,
            getattr(error, "details", {}),
        )

    async def start(self):
        """Start the logging service."""
        await self.audit_logger.start()
        if self.event_log_writer is not None:
            await self.event_log_writer.start()

    async def stop(self):
        """Stop the logging service."""
        try:
            await self.audit_logger.stop()
        finally:
            if self.event_log_writer is not None:
                await self.event_log_writer.stop()
