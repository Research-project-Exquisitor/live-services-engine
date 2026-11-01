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



"""Research events; diagnostic output uses Loguru separately."""
import time
from typing import Any, Dict
from loguru import logger
from ..schemas.logging import ClientLogEvent, EventTarget, EventType, LogEvent
from .event_log_writer import EventLogWriter

class SearchTimer:
    """Times a search from the route, so `SearchService` needs no logging code."""

    def __init__(self):
        self._started_ms = time.time_ns() // 1_000_000
        self._started = time.perf_counter()

    def timing(self) -> Dict[str, Any]:
        return {
            "started_ms": self._started_ms,
            "completed_ms": time.time_ns() // 1_000_000,
            "duration_ms": (time.perf_counter() - self._started) * 1000,
        }


class ResearchLog:
    """Request-bound research event recorder sharing one session-file writer."""
    def __init__(
        self,
        event_log_writer: EventLogWriter,
        *,
        request_id: str | None = None,
        sequence: int | None = None,
    ):
        self.event_log_writer = event_log_writer
        self._request_id = request_id
        self._sequence = sequence

    def for_request(self, request_id: str | None, sequence: int | None) -> "ResearchLog":
        """Bind context to a new facade, never to the shared application instance."""
        return ResearchLog(
            self.event_log_writer, request_id=request_id, sequence=sequence
        )

    async def log_event(
        self,
        event_type: EventType,
        target: EventTarget,
        session: str,
        data: Dict[str, Any],
    ) -> None:
        """Build and queue a server event using this request's causal context.

        Clients without a sequence cannot establish causal order; skip their events. Research logging failures remain diagnostic and
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

    async def log_query_execution(
        self,
        session: str,
        request_data: Dict[str, Any],
        execution: Dict[str, Any],
        suggestions: list,
        timer: "SearchTimer",
    ) -> None:
        """Record a finished search as a ``query_execution`` research event.

        Awaited before the response, so ``time.server`` stays before the client's
        response event.
        """
        await self.log_event("query_execution", "search", session, {
            **{key: value for key, value in execution.items() if key != "suggestions"},
            **timer.timing(),
            "request": request_data,
            "suggestions": suggestions,
        })

    async def log_search_error(
        self, session: str, request_data: Dict[str, Any], error: Exception,
    ) -> None:
        """Record a failed search as an ``error`` research event.

        The route already logs the traceback to the developer log.
        """
        await self.log_event("error", "search", session, {
            "message": str(error),
            "error_type": type(error).__name__,
            "context": {"request": request_data},
        })


    async def start(self):
        await self.event_log_writer.start()

    async def stop(self):
        await self.event_log_writer.stop()
