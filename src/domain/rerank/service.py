import logging
from typing import Literal

from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.domain.opensearch.client import OpenSearchClient

from .schemas import RerankSearchResult, SearchDocument

logger = logging.getLogger(__name__)


def _hit_to_document(hit: dict, rank: int) -> SearchDocument:
    return SearchDocument(
        arxiv_id=hit["arxiv_id"],
        chunk_text=hit["chunk_text"],
        title=hit.get("title", ""),
        authors=hit.get("authors", ""),
        section_name=hit.get("section_name", ""),
        score=hit.get("score", 0.0),
        rank=rank,
        chunk_id=hit.get("chunk_id"),
    )


class RerankSearchService:
    """Search OpenSearch and optionally rerank candidates with Jina."""

    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        embeddings_client: JinaEmbeddingsClient,
        reranker_client: JinaRerankerClient | None = None,
    ) -> None:
        self.opensearch_client = opensearch_client
        self.embeddings_client = embeddings_client
        self.reranker_client = reranker_client

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        use_hybrid: bool = True,
        rerank_enabled: bool = True,
        rerank_candidate_multiplier: int = 2,
        rerank_model: str = "jina-reranker-v2-base-multilingual",
    ) -> RerankSearchResult:
        """Retrieve documents from OpenSearch and optionally rerank them with Jina.

        :param query: Search query
        :param top_k: Number of documents to return after reranking
        :param use_hybrid: Use hybrid BM25 + vector search when True
        :param rerank_enabled: Apply Jina reranking when configured
        :param rerank_candidate_multiplier: Candidate pool size multiplier before reranking
        :param rerank_model: Jina reranker model name
        :returns: Retrieval result with before and after rerank document lists
        """
        search_mode: Literal["hybrid", "bm25"] = "hybrid" if use_hybrid else "bm25"
        should_rerank = (
            rerank_enabled
            and self.reranker_client is not None
            and self.reranker_client.is_configured
        )

        search_size = top_k
        if should_rerank:
            search_size = top_k * rerank_candidate_multiplier

        logger.info("Searching OpenSearch for query: %s...", query[:100])
        logger.debug(
            "Search mode: %s, top_k: %s, rerank: %s, search_size: %s",
            search_mode,
            top_k,
            should_rerank,
            search_size,
        )

        query_embedding = await self.embeddings_client.embed_query(query)
        search_results = self.opensearch_client.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=search_size,
            use_hybrid=use_hybrid,
        )

        hits = search_results.get("hits", [])
        before_rerank = [_hit_to_document(hit, rank) for rank, hit in enumerate(hits)]
        logger.info("Found %d documents from OpenSearch", len(before_rerank))

        if should_rerank and len(hits) > 1:
            try:
                logger.info("Reranking %d candidates with Jina reranker", len(hits))
                after_rerank = await self._rerank_hits(
                    query=query,
                    hits=hits,
                    top_n=top_k,
                    model=rerank_model,
                )
                logger.info("Reranking complete, returning top %d documents", len(after_rerank))
                return RerankSearchResult(
                    query=query,
                    search_mode=search_mode,
                    rerank_applied=True,
                    before_rerank=before_rerank,
                    after_rerank=after_rerank,
                )
            except Exception as e:
                logger.warning(
                    "Reranking failed, using original search order: query_preview=%r, "
                    "candidate_count=%d, top_k=%d, model=%s, error_type=%s, error=%s",
                    query[:100],
                    len(hits),
                    top_k,
                    rerank_model,
                    type(e).__name__,
                    e,
                )

        after_rerank = before_rerank[:top_k]
        return RerankSearchResult(
            query=query,
            search_mode=search_mode,
            rerank_applied=False,
            before_rerank=before_rerank,
            after_rerank=after_rerank,
        )

    async def _rerank_hits(
        self,
        query: str,
        hits: list[dict],
        top_n: int,
        model: str,
    ) -> list[SearchDocument]:
        documents = [hit["chunk_text"] for hit in hits]
        rerank_results = await self.reranker_client.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            model=model,
        )

        reranked_documents: list[SearchDocument] = []
        for new_rank, result in enumerate(rerank_results):
            hit = hits[result.index]
            document = _hit_to_document(hit, rank=new_rank)
            document.score = result.relevance_score
            document.rerank_score = result.relevance_score
            document.original_rank = result.index
            reranked_documents.append(document)

        return reranked_documents
