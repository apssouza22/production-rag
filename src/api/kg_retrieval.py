"""HTTP adapter for dual-level KG retrieval (Phase 2 debug endpoint only)."""

import logging

from fastapi import APIRouter, HTTPException, Request

from src.api.kg_retrieval_schemas import KgRetrievalDebugRequest, KgRetrievalDebugResponse
from src.dependencies import SettingsDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/kg-retrieval", tags=["kg-retrieval"])


@router.post("/debug", response_model=KgRetrievalDebugResponse)
async def kg_retrieval_debug(
    body: KgRetrievalDebugRequest,
    request: Request,
    settings: SettingsDep,
) -> KgRetrievalDebugResponse:
    """Run dual-level KG retrieval and return structured context (debug / eval only)."""
    if not settings.kg.enabled:
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph retrieval is disabled. Set KG__ENABLED=true and index a corpus slice first.",
        )

    service = getattr(request.app.state, "kg_retrieval_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="KG retrieval service is not initialized")

    logger.info("KG debug retrieval: %s...", body.query[:80])
    result = await service.retrieve(
        body.query,
        top_k_entities=body.top_k_entities,
        top_k_relations=body.top_k_relations,
    )
    return KgRetrievalDebugResponse.model_validate(result.model_dump())
