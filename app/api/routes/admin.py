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


"""Administrative, informational, and client-event logging routes.

Defines endpoints under ``/exq/`` for session management, collection
metadata queries, and audit logging of client-side events:

- ``GET  /init/{session}`` -- initialise a session; returns available collections.
- ``POST /info/totalItems`` -- total item count for a collection.
- ``GET  /info/filters/{session}/{collection}`` -- filter (tagset) definitions.
- ``GET  /info/filters/values/{session}/{collection}/{tagtypeId}/{tagsetId}``
  -- possible values for a specific filter.
- ``GET  /info/indexes/{session}/{collection}`` -- the collection's configured
  indexes, so a client can discover index names instead of hardcoding them.
- ``POST /log/addModel`` / ``POST /log/removeModel`` -- audit model lifecycle.
- ``POST /log/clientEvent`` -- batch-log arbitrary client-side UI events.
- ``POST /log/clientEventV2`` -- validate and queue structured research events.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, Depends
from loguru import logger

from app.repositories.database_repository import DatabaseRepository

from ...schemas import (
    ClientEventRequest,
    SessionInfo,
    AddOrRemoveModelRequest,
)
from ...services.logging_service import AuditLogger
from ...services.research_log import ResearchLog
from ...schemas.logging import ClientLogEvent
from ..dependencies import (
    get_database_repository,
    get_config_manager,
    get_audit_logger,
    get_research_log,
)

router = APIRouter(prefix="/exq", tags=["admin"])


@router.get("/init/{session}")
async def init_session(
    session: str,
    config_manager=Depends(get_config_manager),
) -> Dict[str, Any]:
    """Initialize session and return available collections."""
    try:
        config = config_manager.config
        collections = config.collections

        return {"session": session, "collections": collections}

    except Exception as e:
        logger.exception("init_session failed")
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@router.post("/info/totalItems")
async def get_total_items(
    request: SessionInfo,
    database_repo: DatabaseRepository = Depends(get_database_repository),
) -> Dict[str, int]:
    """Get total number of items in a collection."""
    try:
        total_items = database_repo.get_total_items(request.collection)

        return {"total_items": total_items}

    except Exception as e:
        logger.exception("get_total_items failed")
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@router.get("/info/filters/{session}/{collection}")
async def get_filters(
    session: str,
    collection: str,
    database_repo: DatabaseRepository = Depends(get_database_repository),
) -> List[Dict[str, Any]]:
    """Get available filter definitions for a collection"""
    try:
        filters = database_repo.get_filters(collection) or []

        return filters
    except Exception as e:
        logger.exception("get_filters failed")
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@router.get("/info/filters/values/{session}/{collection}/{tagtypeId}/{tagsetId}")
async def get_filter_values(
    session: str,
    collection: str,
    tagtypeId: int,
    tagsetId: int,
    database_repo: DatabaseRepository = Depends(get_database_repository),
) -> List[Dict[str, Any]]:
    """Get possible values for a specific filter in a collection."""
    try:
        filter_values = database_repo.get_filter_values(collection, tagsetId, tagtypeId)

        return filter_values

    except Exception as e:
        logger.exception("get_filter_values failed")
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@router.get("/info/indexes/{session}/{collection}")
async def get_indexes(
    session: str,
    collection: str,
    config_manager=Depends(get_config_manager),
) -> List[Dict[str, Any]]:
    """List the indexes configured for a collection.

    Lets a client discover available index names (and which embedding
    family/media type each belongs to) instead of hardcoding them, for
    picking an `index_name` on /clip, /text, or /rf.
    """
    try:
        collection_config = config_manager.config.collection_configs[collection]
    except KeyError as e:
        logger.exception("get_indexes failed")
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Unknown collection: {collection!r}") from e

    default_index_name = collection_config.default_index.name
    return [
        {
            "name": index.name,
            "index_type": index.index_type,
            "embedding_type": index.embedding_type,
            "source_type": index.source_type,
            "default": index.name == default_index_name,
        }
        for index in collection_config.indexes
    ]


@router.post("/log/addModel")
async def log_add_model(
    request: AddOrRemoveModelRequest,
    background_tasks: BackgroundTasks,
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> Dict[str, str]:
    """Log model addition."""
    background_tasks.add_task(
        audit_logger.log,
        action="Log Add Model",
        session=request.session,
        data={
            "session": request.session,
            "modelId": request.modelId,
            "collection": request.collection,
            "body": request.model_dump_json(),
        },
    )
    return {"status": "Logged successfully"}


@router.post("/log/removeModel")
async def log_remove_model(
    request: AddOrRemoveModelRequest,
    background_tasks: BackgroundTasks,
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> Dict[str, str]:
    """Log model removal."""
    background_tasks.add_task(
        audit_logger.log,
        action="Log Remove Model",
        session=request.session,
        data={
            "session": request.session,
            "modelId": request.modelId,
            "collection": request.collection,
            "body": request.model_dump_json(),
        },
    )
    return {"status": "Logged successfully"}


@router.post("/log/clientEventV2", status_code=202)
async def log_events(
    events: list[ClientLogEvent],
    research_log: ResearchLog = Depends(get_research_log),
) -> dict[str, int]:
    """Validate the whole batch before queueing it for the JSONL writer.

    Accepted means queued, not durably stored. Delivery guarantees and retries
    are Phase 4 work; this endpoint does not promise deduplication yet.
    """
    return {"accepted": await research_log.log_client_events(events)}


@router.post("/log/clientEvent")
async def log_client_event(
    request: List[ClientEventRequest],
    background_tasks: BackgroundTasks,
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> Dict[str, str]:
    """Log a generic client event."""
    # Kept for UIs that don't send research events; /log/clientEventV2 is the new format.
    for event in request:
        background_tasks.add_task(
            audit_logger.log,
            action=f"Client Event: {event.action}",
            session=event.session,
            data={
                "element_id": event.element_id,
                "route": event.route,
                "body": event.data,
            },
        )
    return {"status": "Logged successfully"}
