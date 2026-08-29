import json
import logging
from typing import Any, List

import httpx
from pydantic import ValidationError

from .jina import JinaRerankRequest, JinaRerankResponse, JinaRerankResult

logger = logging.getLogger(__name__)

_RESPONSE_LOG_MAX_CHARS = 2000


def _summarize_rerank_response(response_data: dict[str, Any]) -> str:
    """Build a compact summary of a Jina rerank response for error logs."""
    summary: dict[str, Any] = {
        "model": response_data.get("model"),
        "usage": response_data.get("usage"),
        "results_count": len(response_data.get("results", [])),
    }
    results = response_data.get("results", [])
    if results:
        first = results[0]
        document = first.get("document")
        summary["first_result"] = {
            "keys": list(first.keys()),
            "index": first.get("index"),
            "index_type": type(first.get("index")).__name__,
            "score": first.get("relevance_score", first.get("score")),
            "score_type": type(first.get("relevance_score", first.get("score"))).__name__,
            "document_type": type(document).__name__,
            "document_preview": (
                f"{document[:120]}..." if isinstance(document, str) and len(document) > 120 else document
            ),
        }
    serialized = json.dumps(summary, default=str)
    if len(serialized) > _RESPONSE_LOG_MAX_CHARS:
        return f"{serialized[:_RESPONSE_LOG_MAX_CHARS]}... (truncated)"
    return serialized


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

        effective_top_n = min(top_n, len(documents))
        request_data = JinaRerankRequest(
            model=model,
            query=query,
            documents=documents,
            top_n=effective_top_n,
        )

        logger.debug(
            "Sending Jina rerank request: model=%s, top_n=%d, query_len=%d, "
            "num_documents=%d, document_lengths=%s",
            model,
            effective_top_n,
            len(query),
            len(documents),
            [len(doc) for doc in documents],
        )

        try:
            response = await self.client.post(
                f"{self.base_url}/rerank",
                headers=self.headers,
                json=request_data.model_dump(),
            )
            response.raise_for_status()

            response_data = response.json()
            try:
                result = JinaRerankResponse(**response_data)
            except ValidationError as e:
                logger.error(
                    "Jina rerank response validation failed: model=%s, top_n=%d, "
                    "num_documents=%d, query_preview=%r, http_status=%d, "
                    "validation_errors=%s, response_summary=%s",
                    model,
                    effective_top_n,
                    len(documents),
                    query[:100],
                    response.status_code,
                    e.errors(),
                    _summarize_rerank_response(response_data),
                )
                raise

            logger.info(
                "Reranked %d documents, returning top %d",
                len(documents),
                len(result.results),
            )
            return result.results

        except httpx.HTTPError as e:
            logger.error(
                "HTTP error reranking documents: model=%s, top_n=%d, num_documents=%d, "
                "query_preview=%r, error=%s",
                model,
                effective_top_n,
                len(documents),
                query[:100],
                e,
            )
            raise
        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                "Unexpected error in rerank: model=%s, top_n=%d, num_documents=%d, "
                "query_preview=%r, error_type=%s, error=%s",
                model,
                effective_top_n,
                len(documents),
                query[:100],
                type(e).__name__,
                e,
            )
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
