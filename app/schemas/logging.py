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


"""Logging schemas: the research event model and the legacy client event.

`LogEvent` is the common event envelope from the logging framework
(``research-workspace/notes/logging-framework.md`` §3). Client and server
events both use it; every event is written as one JSON line by
`app.services.event_log_writer.EventLogWriter`.

`ClientEventRequest` is the old payload of ``POST /exq/log/clientEvent``. It
stays until the client events move to `LogEvent`.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1
"""Version of the `LogEvent` format, written to every event as ``v``."""

EventType = Literal[
    "session_start",
    "session_end",
    "query_submission",
    "query_execution",
    "query_response_received",
    "result_state",
    "result_click",
    "relevance_label",
    "display_change",
    "visibility_change",
    "filter_change",
    "video_play",
    "video_seek",
    "back_navigation",
    "answer_submission",
    "error",
]
"""Event types from the logging framework. Add new types here, never free text."""

EventTarget = Literal[
    "session",
    "search",
    "result",
    "results",
    "filter",
    "video",
    "answer",
]
"""What an event acted on (framework §9)."""


class EventTime(BaseModel):
    """Client and server timestamps, both Unix milliseconds (framework §5)."""

    model_config = ConfigDict(extra="forbid")

    client: Optional[int] = Field(default=None, ge=0)
    """When the event happened in the browser. ``None`` for server events."""

    server: Optional[int] = Field(default=None, ge=0)
    """When the backend received the event. Set by `EventLogWriter` on receipt."""


class LogEvent(BaseModel):
    """One research event in the common envelope (framework §3).

    Ordering: events are ordered by ``sequence`` within a session. A server
    event has no client event of its own, so it takes the ``sequence`` of the
    client request that caused it and sorts directly after it (see
    `LogEvent.order_key`).
    """

    model_config = ConfigDict(extra="forbid")

    type: EventType
    """What happened."""

    target: EventTarget
    """What it happened to."""

    source: Literal["client", "server"]
    """Which side recorded the event."""

    time: EventTime
    """Client and server timestamps."""

    sequence: int = Field(ge=0)
    """Position of the event within its session."""

    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    """Session the event belongs to. Also used in the log file name."""

    user_id: Optional[str] = Field(default=None, max_length=64)
    """Pseudonymous participant ID. Optional until its source is decided."""

    task_id: Optional[str] = Field(default=None, max_length=64)
    """Retrieval task active when the event happened; ``None`` outside a task."""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1, max_length=64)
    """Unique per event. The client sets its own, so retries can be deduplicated."""

    v: Literal[1] = SCHEMA_VERSION
    """Schema version, so export scripts know which format they are reading."""

    data: dict[str, Any] = Field(default_factory=dict)
    """Event-specific payload, e.g. ``item_id`` or ``request_id``."""

    @model_validator(mode="after")
    def _client_events_have_client_time(self) -> LogEvent:
        if self.source == "client" and self.time.client is None:
            raise ValueError("client events need time.client")
        return self

    def order_key(self) -> tuple[int, int]:
        """Sort key within a session: by sequence, client before server."""
        return (self.sequence, 0 if self.source == "client" else 1)


class ClientLogEvent(LogEvent):
    """Client intake event; the server assigns its own receive timestamp."""

    source: Literal["client"]

    @model_validator(mode="after")
    def server_time_is_assigned_by_server(self):
        if self.time.server is not None:
            raise ValueError("time.server is assigned by the server")
        return self


class ClientEventRequest(BaseModel):
    """Request model for logging client-side events.

    Attributes:
        ts: Timestamp of the event
        action: Description of the action taken by the user
        element_id: Optional stable identifier for the UI element involved
        data: Optional additional data related to the event
        route: Optional route or page where the event occurred
        session: Unique identifier for the search session
    """

    ts: float
    """Timestamp of the event (Unix epoch seconds)."""

    action: str
    """Description of the action taken by the user (e.g. ``'click'``, ``'scroll'``)."""

    element_id: Optional[str] = None
    """Stable identifier for the UI element involved, if any."""

    data: Optional[Union[str, float, bool]] = None
    """Optional payload associated with the event."""

    route: Optional[str] = None
    """Client-side route or page where the event occurred."""

    session: str
    """Session identifier linking this event to a search session."""
