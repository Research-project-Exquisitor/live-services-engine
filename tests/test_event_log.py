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


"""Tests for the research event model (`LogEvent`) and its writer (`EventLogWriter`)."""

import asyncio
import json
import logging

import pytest
from loguru import logger
from pydantic import ValidationError

from app.core.logging_setup import setup_logging
from app.schemas import LogEvent
from app.services.event_log_writer import EventLogWriter


def client_event(**overrides) -> dict:
    event = {
        "source": "client",
        "session_id": "s-1",
        "time": {"client": 1_727_681_234_567},
        "sequence": 0,
        "type": "result_click",
        "target": "result",
        "data": {"item_id": 123},
    }
    event.update(overrides)
    return event


class TestLogEvent:
    def test_valid_client_event(self):
        event = LogEvent(**client_event())
        assert event.v == 1
        assert event.event_id
        assert event.task_id is None
        assert event.time.server is None

    def test_client_event_needs_client_time(self):
        with pytest.raises(ValidationError, match="time.client"):
            LogEvent(**client_event(time={}))

    def test_server_event_without_client_time(self):
        event = LogEvent(
            **client_event(source="server", type="query_execution", target="search", time={})
        )
        assert event.time.client is None

    @pytest.mark.parametrize(
        "overrides",
        [
            {"type": "Client Event: Click"},
            {"target": "button"},
            {"sequence": -1},
            {"session_id": "../etc/passwd"},
            {"session_id": ""},
            {"surprise": True},
            {"time": {"client": 1, "browser": 2}},
        ],
    )
    def test_rejects_invalid_events(self, overrides):
        with pytest.raises(ValidationError):
            LogEvent(**client_event(**overrides))

    def test_server_event_sorts_after_the_request_that_caused_it(self):
        request = LogEvent(**client_event(sequence=5, type="query_submission", target="search"))
        execution = LogEvent(
            **client_event(source="server", sequence=5, type="query_execution", target="search", time={})
        )
        later = LogEvent(**client_event(sequence=6))
        ordered = sorted([later, execution, request], key=LogEvent.order_key)
        assert ordered == [request, execution, later]


class TestEventLogWriter:
    def test_writes_one_json_line_per_event_per_session(self, tmp_path):
        async def run():
            event_log_writer = EventLogWriter(str(tmp_path))
            await event_log_writer.start()
            await event_log_writer.write(LogEvent(**client_event(sequence=0)))
            await event_log_writer.write(LogEvent(**client_event(sequence=1)))
            await event_log_writer.write(LogEvent(**client_event(session_id="s-2")))
            await event_log_writer.stop()
            return event_log_writer

        event_log_writer = asyncio.run(run())

        lines = event_log_writer.path_for("s-1").read_text().splitlines()
        assert [json.loads(line)["sequence"] for line in lines] == [0, 1]
        assert len(event_log_writer.path_for("s-2").read_text().splitlines()) == 1

    def test_stamps_server_time_on_receipt(self, tmp_path):
        async def run():
            event_log_writer = EventLogWriter(str(tmp_path))
            await event_log_writer.start()
            await event_log_writer.write(LogEvent(**client_event()))
            await event_log_writer.stop()
            return event_log_writer

        event_log_writer = asyncio.run(run())

        written = json.loads(event_log_writer.path_for("s-1").read_text())
        assert written["v"] == 1
        assert written["time"]["client"] == 1_727_681_234_567
        assert written["time"]["server"] > written["time"]["client"]
        LogEvent(**written)  # what we write can be read back


class TestLoggingSetup:
    def test_stdlib_records_reach_loguru_with_context(self):
        setup_logging("DEBUG")
        lines: list[str] = []
        sink = logger.add(lines.append, format="{level} {extra[session]} {message}")
        try:
            with logger.contextualize(session="s-1"):
                logging.getLogger("uvicorn.error").info("from uvicorn")
            logger.debug("outside a request")
        finally:
            logger.remove(sink)
            setup_logging("INFO")

        assert lines[0].strip() == "INFO s-1 from uvicorn"
        assert lines[1].strip() == "DEBUG - outside a request"
