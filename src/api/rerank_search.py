import logging

from fastapi import APIRouter, HTTPException

from src.dependencies import PaperRrfSearchDep, RerankSearchDep
from src.domain.search.schemas import RerankSearchRequest, DocumentSearchResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rerank-search", tags=["rerank-search"])


@router.post("", response_model=DocumentSearchResult)
async def rerank_search(
    request: RerankSearchRequest,
    rerank_search_service: RerankSearchDep,
    paper_rrf_search: PaperRrfSearchDep,
) -> DocumentSearchResult:
    """Search OpenSearch and optionally rerank results with Jina."""
    if not paper_rrf_search.health_check():
        raise HTTPException(status_code=503, detail="Search service is currently unavailable")

    original_use_hybrid = rerank_search_service.config.use_hybrid
    original_rerank_enabled = rerank_search_service.config.rerank_enabled

    try:
        rerank_search_service.config.use_hybrid = request.use_hybrid
        rerank_search_service.config.rerank_enabled = request.rerank_enabled

        logger.info(
            "Rerank search: '%s' (hybrid: %s, rerank: %s, top_k: %s)",
            request.query,
            request.use_hybrid,
            request.rerank_enabled,
            request.top_k,
        )

        result = await rerank_search_service.search_rrf(query=request.query, top_k=request.top_k)
        logger.info(
            "Rerank search completed: before=%d, after=%d, rerank_applied=%s",
            len(result.before_rerank),
            len(result.after_rerank),
            result.rerank_applied,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Rerank search error: %s", e)
        raise HTTPException(status_code=500, detail=f"Rerank search failed: {str(e)}") from e
    finally:
        rerank_search_service.config.use_hybrid = original_use_hybrid
        rerank_search_service.config.rerank_enabled = original_rerank_enabled
