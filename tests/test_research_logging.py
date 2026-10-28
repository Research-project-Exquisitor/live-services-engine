# SPDX-FileCopyrightText: 2026 Ujjwal Sharma and Omar Shahbaz Khan
# SPDX-License-Identifier: AGPL-3.0-or-later

"""HTTP-to-JSONL coverage for the first migrated research interaction."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.dependencies import get_search_service
from app.api.routes import admin, search
from app.core.exceptions import SearchError
from app.schemas.logging import LogEvent
from app.services.event_log_writer import EventLogWriter
from app.services.logging_service import LoggingService
from app.services.search_service import SearchService
from app.strategies.base import TextSearchStrategyABC


@pytest.fixture
def logging_app(tmp_path):
    event_log_writer = EventLogWriter(str(tmp_path))
    strategy = MagicMock(spec=TextSearchStrategyABC)
    strategy.search = AsyncMock(return_value=[33, 15])
    service = SearchService.__new__(SearchService)
    service.strategies = {"clip": strategy, "text": strategy}
    legacy = MagicMock()
    legacy.log = AsyncMock()
    legacy.start = AsyncMock()
    legacy.stop = AsyncMock()
    logging_service = LoggingService(legacy, event_log_writer)

    @asynccontextmanager
    async def lifespan(app):
        app.state.logging_service = logging_service
        await logging_service.start()
        yield
        await logging_service.stop()

    app = FastAPI(lifespan=lifespan)
    app.include_router(admin.router)
    app.include_router(search.router)
    app.dependency_overrides[get_search_service] = lambda: service
    return app, event_log_writer, strategy, legacy


def client_event(**overrides):
    return {
        "source": "client", "session_id": "test-session", "sequence": 0,
        "type": "query_submission", "target": "search",
        "time": {"client": 1_793_181_600_123},
        **overrides,
    }


def search_request():
    return {
        "session_info": {"session": "test-session", "collection": "test", "modelId": 0},
        "text": "a red car", "n": 2, "seen": [7, 8], "excluded": [9],
    }


def read_events(event_log_writer):
    return [LogEvent.model_validate_json(line)
            for line in event_log_writer.path_for("test-session").read_text().splitlines()]


def test_validated_client_events_receive_server_time(logging_app):
    app, event_log_writer, _, _ = logging_app
    with TestClient(app) as client:
        response = client.post("/exq/log/clientEventV2", json=[client_event(request_id="search-1")],
                               headers={"X-Exq-Request-Id": "delivery-1"})
        assert response.status_code == 202
        assert response.json() == {"accepted": 1}
    event, = read_events(event_log_writer)
    assert event.time.client == 1_793_181_600_123
    assert event.time.server is not None
    assert event.request_id == "search-1"
    assert "request_id" not in event.data


@pytest.mark.parametrize("invalid", [
    {"source": "server"}, {"time": {"client": 1, "server": 2}},
])
def test_invalid_batch_is_rejected_before_any_event_is_queued(logging_app, invalid):
    app, event_log_writer, _, _ = logging_app
    with TestClient(app) as client:
        response = client.post("/exq/log/clientEventV2", json=[client_event(), client_event(**invalid)])
        assert response.status_code == 422
    assert not event_log_writer.path_for("test-session").exists()


@pytest.mark.parametrize("endpoint", ["clip", "text"])
def test_search_records_full_inputs_results_and_precise_timing(logging_app, endpoint):
    app, event_log_writer, _, legacy = logging_app
    with TestClient(app) as client:
        response = client.post(f"/exq/search/{endpoint}", json=search_request(), headers={
            "X-Exq-Request-Id": "request-1", "X-Exq-Event-Sequence": "0",
        })
        assert response.status_code == 200
        assert response.json() == {"suggestions": [33, 15]}
    event, = read_events(event_log_writer)
    assert event.type == "query_execution"
    assert event.sequence == 0
    assert event.source == "server"
    assert event.request_id == "request-1"
    assert "request_id" not in event.data
    assert event.data["request"]["seen"] == [7, 8]
    assert event.data["request"]["excluded"] == [9]
    assert event.data["suggestions"] == [33, 15]
    assert event.data["started_ms"] <= event.data["completed_ms"]
    assert event.data["started_ms"] > 1_000_000_000_000
    assert event.data["duration_ms"] >= 0
    legacy.log.assert_awaited_once()


def test_old_search_clients_keep_working_without_invented_sequence(logging_app):
    app, event_log_writer, _, legacy = logging_app
    with TestClient(app) as client:
        assert client.post("/exq/search/clip", json=search_request()).status_code == 200
    assert not event_log_writer.path_for("test-session").exists()
    legacy.log.assert_awaited_once()


def test_research_logging_failure_does_not_break_search(logging_app, monkeypatch):
    app, event_log_writer, _, legacy = logging_app
    monkeypatch.setattr(event_log_writer, "write", AsyncMock(side_effect=RuntimeError("queue failure")))
    with TestClient(app) as client:
        response = client.post("/exq/search/clip", json=search_request(), headers={
            "X-Exq-Request-Id": "request-1", "X-Exq-Event-Sequence": "0",
        })
        assert response.status_code == 200
        assert response.json() == {"suggestions": [33, 15]}
    legacy.log.assert_awaited_once()


def test_failed_search_preserves_error_and_request_link_when_audit_logging_fails(logging_app):
    app, event_log_writer, strategy, legacy = logging_app
    strategy.search.side_effect = SearchError("test failure")
    legacy.log.side_effect = RuntimeError("audit queue failure")
    with TestClient(app) as client:
        response = client.post("/exq/search/clip", json=search_request(), headers={
            "X-Exq-Request-Id": "request-1", "X-Exq-Event-Sequence": "0",
        })
        assert response.status_code == 400
        assert "test failure" in response.json()["detail"]
    event, = read_events(event_log_writer)
    assert event.type == "error"
    assert event.request_id == "request-1"
    assert "test failure" in event.data["message"]


def test_intake_does_not_acknowledge_failed_queueing(logging_app, monkeypatch):
    app, event_log_writer, _, _ = logging_app
    monkeypatch.setattr(event_log_writer, "write", AsyncMock(side_effect=RuntimeError("queue failure")))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/exq/log/clientEventV2", json=[client_event()])
        assert response.status_code == 500


def test_concurrent_http_requests_keep_logging_context_isolated(logging_app):
    app, event_log_writer, strategy, _ = logging_app
    arrivals = 0
    both_arrived = None

    async def interleaved_search(**kwargs):
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            both_arrived.set()
        await asyncio.wait_for(both_arrived.wait(), timeout=2)
        return [33, 15]

    strategy.search.side_effect = interleaved_search

    async def run():
        nonlocal both_arrived
        both_arrived = asyncio.Event()
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                async def search_for(suffix, sequence):
                    body = search_request()
                    body["session_info"]["session"] = f"s-{suffix}"
                    return await client.post("/exq/search/clip", json=body, headers={
                        "X-Exq-Request-Id": f"r-{suffix}", "X-Exq-Event-Sequence": str(sequence),
                    })
                responses = await asyncio.gather(search_for("a", 2), search_for("b", 7))
                assert all(response.status_code == 200 for response in responses)

    asyncio.run(run())
    for suffix, sequence in [("a", 2), ("b", 7)]:
        event = LogEvent.model_validate_json(event_log_writer.path_for(f"s-{suffix}").read_text())
        assert event.session_id == f"s-{suffix}"
        assert event.sequence == sequence
        assert event.request_id == f"r-{suffix}"
