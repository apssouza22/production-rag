import logging
from typing import List

import httpx

from .jina import JinaRerankRequest, JinaRerankResponse, JinaRerankResult

logger = logging.getLogger(__name__)


class JinaRerankerClient:
    """Client for Jina AI reranker API.

    Uses cross-encoder reranking to improve retrieval precision after initial search.
    Documentation: https://jina.ai/reranker
    """

    def __init__(self, api_key: str, base_url: str = "https://api.jina.ai/v1"):
        """Initialize Jina reranker client.

        :param api_key: Jina API key
        :param base_url: API base URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("Jina reranker client initialized")

    @property
    def is_configured(self) -> bool:
        """Return whether the client has a valid API key."""
        return bool(self.api_key)

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int,
        model: str = "jina-reranker-v2-base-multilingual",
    ) -> List[JinaRerankResult]:
        """Rerank documents by relevance to the query.

        :param query: Search query
        :param documents: Candidate document texts to rerank
        :param top_n: Number of top results to return
        :param model: Jina reranker model name
        :returns: Reranked results ordered by relevance (highest first)
        """
        if not documents:
            return []

        if not self.is_configured:
            raise ValueError("Jina API key is not configured")

        request_data = JinaRerankRequest(
            model=model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
        )

        try:
            response = await self.client.post(
                f"{self.base_url}/rerank",
                headers=self.headers,
                json=request_data.model_dump(),
            )
            response.raise_for_status()

            result = JinaRerankResponse(**response.json())
            logger.info(
                "Reranked %d documents, returning top %d",
                len(documents),
                len(result.results),
            )
            return result.results

        except httpx.HTTPError as e:
            logger.error("Error reranking documents: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error in rerank: %s", e)
            raise

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
