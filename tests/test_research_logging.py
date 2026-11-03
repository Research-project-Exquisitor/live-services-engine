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
from loguru import logger

from app.api.dependencies import get_search_service
from app.api.routes import admin, search
from app.core.exceptions import SearchError
from app.schemas import TemporalChain
from app.schemas.logging import LogEvent
from app.services.event_log_writer import EventLogWriter
from app.services.research_log import ResearchLog
from app.services.search_service import SearchService
from app.strategies.base import (
    FacetedSearchStrategyABC, RFSearchStrategyABC, TemporalSearchStrategyABC, TextSearchStrategyABC,
)


TEMPORAL_CHAIN = TemporalChain(group=1, items=[33, 15], length=2, score=0.5, start_ms=0, end_ms=9000)
FACETED_FILTERS = {"root": {"kind": "leaf", "filter": {
    "id": 1, "tagtype_id": 1, "constraint": {"value_ids": [3]},
}}}


@pytest.fixture
def logging_app(tmp_path):
    event_log_writer = EventLogWriter(str(tmp_path))
    strategy = MagicMock(spec=TextSearchStrategyABC)
    strategy.search = AsyncMock(return_value=[33, 15])
    service = SearchService.__new__(SearchService)
    rf_strategy = MagicMock(spec=RFSearchStrategyABC)
    rf_strategy.search = AsyncMock(return_value=[33, 15])
    temporal_strategy = MagicMock(spec=TemporalSearchStrategyABC)
    temporal_strategy.search = AsyncMock(return_value=[TEMPORAL_CHAIN])
    faceted_strategy = MagicMock(spec=FacetedSearchStrategyABC)
    faceted_strategy.search = AsyncMock(return_value=[33, 15])
    service.strategies = {
        "clip": strategy, "text": strategy, "rf": rf_strategy,
        "temporal": temporal_strategy, "faceted": faceted_strategy,
    }
    research_log = ResearchLog(event_log_writer)

    @asynccontextmanager
    async def lifespan(app):
        app.state.research_log = research_log
        await research_log.start()
        yield
        await research_log.stop()

    app = FastAPI(lifespan=lifespan)
    # Legacy routes only; searches and research events must never reach it.
    app.state.audit_logger = MagicMock()
    app.state.audit_logger.log = AsyncMock()
    app.include_router(admin.router)
    app.include_router(search.router)
    app.dependency_overrides[get_search_service] = lambda: service
    return app, event_log_writer, strategy


def client_event(**overrides):
    return {
        "source": "client", "session_id": "test-session", "sequence": 0,
        "type": "query_submission", "target": "search",
        "time": {"client": 1_793_181_600_123},
        **overrides,
    }


def search_request(endpoint="clip"):
    body = {
        "session_info": {"session": "test-session", "collection": "test", "modelId": 0},
        "n": 2, "seen": [7, 8], "excluded": [9],
    }
    if endpoint == "rf":
        body.update(pos=[5], neg=[6], query="a red car")
    elif endpoint == "temporal":
        body["hops"] = [{"kind": "query", "text": "a red car"}, {"kind": "query", "text": "a blue car"}]
    elif endpoint == "faceted":
        body = {"session_info": body["session_info"], "n": 2, "filters": FACETED_FILTERS}
    else:
        body["text"] = "a red car"
    return body


def read_events(event_log_writer):
    return [LogEvent.model_validate_json(line)
            for line in event_log_writer.path_for("test-session").read_text().splitlines()]


def test_hover_events_keep_their_raw_start_and_end(logging_app):
    app, event_log_writer, _ = logging_app
    hover = client_event(type="hover", target="result", request_id="search-1", data={
        "item_id": 731, "start_ms": 1_793_181_600_000, "end_ms": 1_793_181_600_040,
    })
    with TestClient(app) as client:
        assert client.post("/exq/log/clientEventV2", json=[hover]).status_code == 202
    [event] = read_events(event_log_writer)
    assert event.type == "hover" and event.request_id == "search-1"
    # A 40 ms hover is kept: no minimum duration in the browser (R21).
    assert event.data["end_ms"] - event.data["start_ms"] == 40


def test_text_input_keeps_every_change_with_the_full_text(logging_app):
    app, event_log_writer, _ = logging_app
    edits = [client_event(type="text_input", sequence=i, data={
        "surface": "search_bar", "text": text, "input_type": kind,
    }) for i, (text, kind) in enumerate([("d", "insertText"), ("", "deleteContentBackward"), ("dog", "insertFromPaste")])]
    with TestClient(app) as client:
        assert client.post("/exq/log/clientEventV2", json=edits).status_code == 202
    events = read_events(event_log_writer)
    assert [(e.data["text"], e.data["input_type"]) for e in events] == [
        ("d", "insertText"), ("", "deleteContentBackward"), ("dog", "insertFromPaste"),
    ]


def test_validated_client_events_receive_server_time(logging_app):
    app, event_log_writer, _ = logging_app
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
    app, event_log_writer, _ = logging_app
    with TestClient(app) as client:
        response = client.post("/exq/log/clientEventV2", json=[client_event(), client_event(**invalid)])
        assert response.status_code == 422
    assert not event_log_writer.path_for("test-session").exists()


@pytest.mark.parametrize("endpoint", ["clip", "text", "rf"])
def test_search_records_full_inputs_results_and_precise_timing(logging_app, endpoint):
    app, event_log_writer, _ = logging_app
    with TestClient(app) as client:
        response = client.post(f"/exq/search/{endpoint}", json=search_request(endpoint), headers={
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
    assert event.data["strategy"] == endpoint
    if endpoint == "rf":
        assert event.data["request"]["pos"] == [5]
        assert event.data["request"]["neg"] == [6]
        assert event.data["request"]["query"] == "a red car"
    assert event.data["started_ms"] <= event.data["completed_ms"]
    assert event.data["started_ms"] > 1_000_000_000_000
    assert event.data["duration_ms"] >= 0


def test_temporal_search_records_hops_and_chains(logging_app):
    app, event_log_writer, _ = logging_app
    with TestClient(app) as client:
        response = client.post("/exq/search/temporal", json=search_request("temporal"), headers={
            "X-Exq-Request-Id": "request-1", "X-Exq-Event-Sequence": "0",
        })
        assert response.status_code == 200
    chain = TEMPORAL_CHAIN.model_dump(mode="json")
    assert response.json() == {"suggestions": [chain]}
    event, = read_events(event_log_writer)
    assert event.type == "query_execution"
    assert event.request_id == "request-1"
    assert event.data["strategy"] == "temporal"
    assert [hop["text"] for hop in event.data["request"]["hops"]] == ["a red car", "a blue car"]
    assert event.data["request"]["seen"] == [7, 8]
    assert event.data["suggestions"] == [chain]
    assert event.data["duration_ms"] >= 0


def test_faceted_search_records_filters_and_results(logging_app):
    app, event_log_writer, _ = logging_app
    with TestClient(app) as client:
        response = client.post("/exq/search/faceted", json=search_request("faceted"), headers={
            "X-Exq-Request-Id": "request-1", "X-Exq-Event-Sequence": "0",
        })
        assert response.status_code == 200
        assert response.json() == {"suggestions": [33, 15]}
    event, = read_events(event_log_writer)
    assert event.type == "query_execution"
    assert event.data["strategy"] == "faceted"
    assert event.data["request"]["filters"]["root"]["filter"]["constraint"]["value_ids"] == [3]
    assert event.data["suggestions"] == [33, 15]
    assert event.data["started_ms"] <= event.data["completed_ms"]


def test_old_search_clients_keep_working_without_invented_sequence(logging_app):
    app, event_log_writer, _ = logging_app
    with TestClient(app) as client:
        assert client.post("/exq/search/clip", json=search_request()).status_code == 200
    assert not event_log_writer.path_for("test-session").exists()


@pytest.mark.parametrize("headers", [
    {"X-Exq-Request-Id": "r" * 65, "X-Exq-Event-Sequence": "0"},
    {"X-Exq-Request-Id": "request-1", "X-Exq-Event-Sequence": "-1"},
    {"X-Exq-Request-Id": "request-1", "X-Exq-Event-Sequence": "abc"},
    {"X-Exq-Request-Id": "request-1", "X-Exq-Event-Sequence": "9" * 5000},
])
def test_invalid_logging_headers_skip_the_event_but_not_the_search(logging_app, headers):
    app, event_log_writer, _ = logging_app
    with TestClient(app) as client:
        response = client.post("/exq/search/clip", json=search_request(), headers=headers)
        assert response.status_code == 200
    assert not event_log_writer.path_for("test-session").exists()


def test_research_logging_failure_does_not_break_search(logging_app, monkeypatch):
    app, event_log_writer, _ = logging_app
    monkeypatch.setattr(event_log_writer, "write", AsyncMock(side_effect=RuntimeError("queue failure")))
    with TestClient(app) as client:
        response = client.post("/exq/search/clip", json=search_request(), headers={
            "X-Exq-Request-Id": "request-1", "X-Exq-Event-Sequence": "0",
        })
        assert response.status_code == 200
        assert response.json() == {"suggestions": [33, 15]}


@pytest.mark.parametrize("endpoint", ["clip", "text", "rf", "temporal", "faceted"])
def test_failed_search_preserves_error_and_request_link(logging_app, endpoint):
    app, event_log_writer, _ = logging_app
    strategy = app.dependency_overrides[get_search_service]().strategies[endpoint]
    strategy.search.side_effect = SearchError("test failure")
    diagnostics = []
    sink = logger.add(lambda message: diagnostics.append(message.record))
    try:
        with TestClient(app) as client:
            response = client.post(f"/exq/search/{endpoint}", json=search_request(endpoint), headers={
                "X-Exq-Request-Id": "request-1", "X-Exq-Event-Sequence": "0",
            })
            assert response.status_code == 400
            assert "test failure" in response.json()["detail"]
    finally:
        logger.remove(sink)
    failure, = [record for record in diagnostics if record["exception"] is not None]
    assert isinstance(failure["exception"].value, SearchError)
    assert failure["exception"].traceback is not None
    event, = read_events(event_log_writer)
    assert event.type == "error"
    assert event.request_id == "request-1"
    assert "test failure" in event.data["message"]


def test_intake_does_not_acknowledge_failed_queueing(logging_app, monkeypatch):
    app, event_log_writer, _ = logging_app
    monkeypatch.setattr(event_log_writer, "write", AsyncMock(side_effect=RuntimeError("queue failure")))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/exq/log/clientEventV2", json=[client_event()])
        assert response.status_code == 500


def test_concurrent_http_requests_keep_logging_context_isolated(logging_app):
    app, event_log_writer, strategy = logging_app
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


def test_legacy_routes_keep_writing_the_audit_log_and_nothing_else_does(logging_app):
    app, writer, _ = logging_app
    legacy = {"session": "test-session", "collection": "test", "modelId": 0}
    with TestClient(app) as client:
        assert client.post("/exq/search/clip", json=search_request(), headers={
            "X-Exq-Request-Id": "search-1", "X-Exq-Event-Sequence": "0",
        }).status_code == 200
        assert client.post("/exq/log/clientEventV2", json=[client_event()]).status_code == 202
        app.state.audit_logger.log.assert_not_awaited()
        assert client.post("/exq/log/addModel", json=legacy).json() == {"status": "Logged successfully"}
        assert client.post("/exq/log/removeModel", json=legacy).json() == {"status": "Logged successfully"}
        assert client.post("/exq/log/clientEvent", json=[
            {"ts": 1, "action": "Click", "session": "test-session"},
        ]).json() == {"status": "Logged successfully"}
    actions = [call.kwargs["action"] for call in app.state.audit_logger.log.await_args_list]
    assert actions == ["Log Add Model", "Log Remove Model", "Client Event: Click"]
    assert len(read_events(writer)) == 2
