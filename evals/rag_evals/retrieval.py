"""Client for retrieving evaluation contexts via rerank_search_service.search."""

import logging

import httpx

from evals.rag_evals.config import RagEvalSettings
from evals.rag_evals.schemas import RerankSearchResponse

logger = logging.getLogger(__name__)


class RerankSearchClient:
    """Fetch retrieved contexts from the RAG API rerank-search endpoint."""

    def __init__(self, settings: RagEvalSettings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client
        self._endpoint = f"{settings.rag_api_base_url.rstrip('/')}/api/v1/rerank-search"

    async def search(self, query: str) -> RerankSearchResponse:
        """Call rerank_search_service.search through the HTTP API."""
        payload = {
            "query": query,
            "top_k": self.settings.rag_eval_top_k,
            "use_hybrid": self.settings.rag_eval_use_hybrid,
            "rerank_enabled": self.settings.rag_eval_rerank_enabled,
        }

        if self._client is not None:
            response = await self._client.post(self._endpoint, json=payload)
            response.raise_for_status()
            return RerankSearchResponse.model_validate(response.json())

        async with httpx.AsyncClient(timeout=self.settings.rag_eval_request_timeout) as client:
            response = await client.post(self._endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

        result = RerankSearchResponse.model_validate(data)
        logger.info(
            "Retrieved %d contexts for query preview=%r (rerank_applied=%s)",
            len(result.after_rerank),
            query[:80],
            result.rerank_applied,
        )
        return result

    @staticmethod
    def contexts_from_result(result: RerankSearchResponse) -> list[str]:
        """Extract chunk texts used as Ragas retrieved_contexts."""
        return [document.chunk_text for document in result.after_rerank if document.chunk_text.strip()]
