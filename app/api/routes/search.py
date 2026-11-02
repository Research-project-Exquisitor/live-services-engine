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


"""Search endpoint route handlers.

Defines five POST endpoints under ``/exq/search/``:

- ``/clip`` -- CLIP text-to-image similarity search.
- ``/text`` -- text-embedding search (sentence-transformers) over
  non-CLIP indexes, e.g. searching transcripts by meaning.
- ``/rf`` -- relevance-feedback search (SVM-based); targets whichever
  index ``RFSearchRequest.index_name`` names, or the collection's
  default when omitted.
- ``/temporal`` matches an ordered sequence of text or RF queries, each
  against its own index, to chains of segments occurring one after another in a video.
- ``/faceted`` -- filter-only search with no vector similarity.

Each endpoint validates the incoming Pydantic request, delegates to
`SearchService`, records a research event before responding, and returns a
``{"suggestions": [...]}`` response containing the matching media IDs,
or for ``/temporal`` the ranked `TemporalChain` objects.
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from ...schemas import (
    FacetedSearchRequest,
    TextSearchRequest,
    RFSearchRequest,
    TemporalSearchRequest,
)
from ...services.search_service import SearchService
from ...services.research_log import ResearchLog, SearchTimer
from ...core.exceptions import SearchError
from ..dependencies import get_search_service, get_research_log

router = APIRouter(prefix="/exq/search", tags=["search"])


@router.post("/clip")
async def clip_search(
    request: TextSearchRequest,
    search_service: SearchService = Depends(get_search_service),
    research_log: ResearchLog = Depends(get_research_log),
) -> Dict[str, Any]:
    """Search using CLIP text embeddings."""
    timer = SearchTimer()
    try:
        result = await search_service.search_text("clip", request)

        await research_log.log_query_execution(
            request.session_info.session, request.model_dump(mode="json"), result, result["suggestions"], timer,
        )

        return {"suggestions": result["suggestions"]}

    except SearchError as e:
        logger.exception("clip_search failed")
        await research_log.log_search_error(
            request.session_info.session, request.model_dump(mode="json"), e,
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("clip_search failed")
        await research_log.log_search_error(
            request.session_info.session, request.model_dump(mode="json"), e,
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@router.post("/text")
async def text_search(
    request: TextSearchRequest,
    search_service: SearchService = Depends(get_search_service),
    research_log: ResearchLog = Depends(get_research_log),
) -> Dict[str, Any]:
    """Search using sentence-transformers text embeddings."""
    timer = SearchTimer()
    try:
        result = await search_service.search_text("text", request)

        await research_log.log_query_execution(
            request.session_info.session, request.model_dump(mode="json"), result, result["suggestions"], timer,
        )

        return {"suggestions": result["suggestions"]}

    except SearchError as e:
        logger.exception("text_search failed")
        await research_log.log_search_error(
            request.session_info.session, request.model_dump(mode="json"), e,
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("text_search failed")
        await research_log.log_search_error(
            request.session_info.session, request.model_dump(mode="json"), e,
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@router.post("/rf")
async def rf_search(
    request: RFSearchRequest,
    search_service: SearchService = Depends(get_search_service),
    research_log: ResearchLog = Depends(get_research_log),
) -> Dict[str, Any]:
    """Search using relevance feedback (SVM)."""
    timer = SearchTimer()
    try:
        result = await search_service.search_rf(request)

        await research_log.log_query_execution(
            request.session_info.session, request.model_dump(mode="json"), result, result["suggestions"], timer,
        )

        return {"suggestions": result["suggestions"]}

    except SearchError as e:
        logger.exception("rf_search failed")
        await research_log.log_search_error(
            request.session_info.session, request.model_dump(mode="json"), e,
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("rf_search failed")
        await research_log.log_search_error(
            request.session_info.session, request.model_dump(mode="json"), e,
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@router.post("/temporal")
async def temporal_search(
    request: TemporalSearchRequest,
    search_service: SearchService = Depends(get_search_service),
    research_log: ResearchLog = Depends(get_research_log),
) -> Dict[str, Any]:
    """Search for chains of segments matching an ordered sequence of queries."""
    timer = SearchTimer()
    try:
        result = await search_service.search_temporal(request)
        # Using model_dump() to convert TemporalChain into a plain dict for audit logger
        chains = [chain.model_dump() for chain in result["suggestions"]]

        await research_log.log_query_execution(
            request.session_info.session, request.model_dump(mode="json"), result, chains, timer,
        )

        return {"suggestions": chains}

    except SearchError as e:
        logger.exception("temporal_search failed")
        await research_log.log_search_error(
            request.session_info.session, request.model_dump(mode="json"), e,
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("temporal_search failed")
        await research_log.log_search_error(
            request.session_info.session, request.model_dump(mode="json"), e,
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e


@router.post("/faceted")
async def faceted_search(
    request: FacetedSearchRequest,
    search_service: SearchService = Depends(get_search_service),
    research_log: ResearchLog = Depends(get_research_log),
) -> Dict[str, Any]:
    """Search using filters only"""
    timer = SearchTimer()
    try:
        result = await search_service.search_faceted(request)

        await research_log.log_query_execution(
            request.session_info.session, request.model_dump(mode="json"), result, result["suggestions"], timer,
        )

        return {"suggestions": result["suggestions"]}

    except SearchError as e:
        logger.exception("faceted_search failed")
        await research_log.log_search_error(
            request.session_info.session, request.model_dump(mode="json"), e,
        )
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("faceted_search failed")
        await research_log.log_search_error(
            request.session_info.session, request.model_dump(mode="json"), e,
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}") from e
