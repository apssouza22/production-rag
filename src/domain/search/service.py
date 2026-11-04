import logging
from typing import Literal

from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.domain.kg.dual_level_retrieval import DualLevelKGRetrievalService
from src.domain.kg.schemas import DualLevelKGResult
from src.domain.rrf import PaperRrfSearch

from .config import RerankSearchConfig
from .schemas import DocumentSearchResult, SearchDocument

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


class SearchService:
    """Search OpenSearch and optionally rerank candidates with Jina."""

    def __init__(
        self,
        paper_rrf_search: PaperRrfSearch,
        embeddings_client: JinaEmbeddingsClient,
        config: RerankSearchConfig,
        reranker_client: JinaRerankerClient | None = None,
        kg_retrieval_service: DualLevelKGRetrievalService | None = None,
    ) -> None:
        self.paper_rrf_search = paper_rrf_search
        self.embeddings_client = embeddings_client
        self.config = config
        self.reranker_client = reranker_client
        self.kg_retrieval_service = kg_retrieval_service

    def _should_rerank(self) -> bool:
        return (
            self.config.rerank_enabled
            and self.reranker_client is not None
            and self.reranker_client.is_configured
        )

    async def search_rrf(self, query: str, *, top_k: int) -> DocumentSearchResult:
        """Retrieve documents from OpenSearch and optionally rerank them with Jina.

        :param query: Search query
        :param top_k: Number of documents to return after reranking
        :returns: Retrieval result with before and after rerank document lists
        """
        use_hybrid = self.config.use_hybrid
        search_mode: Literal["hybrid", "bm25"] = "hybrid" if use_hybrid else "bm25"
        should_rerank = self._should_rerank()

        search_size = top_k
        if should_rerank:
            search_size = top_k * self.config.rerank_candidate_multiplier

        logger.info("Searching OpenSearch for query: %s...", query[:100])
        logger.debug(
            "Search mode: %s, top_k: %s, rerank: %s, search_size: %s",
            search_mode,
            top_k,
            should_rerank,
            search_size,
        )

        query_embedding = await self.embeddings_client.embed_query(query)
        search_results = self.paper_rrf_search.search_unified(
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
                    model=self.config.rerank_model,
                )
                logger.info("Reranking complete, returning top %d documents", len(after_rerank))
                return DocumentSearchResult(
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
                    self.config.rerank_model,
                    type(e).__name__,
                    e,
                )

        return DocumentSearchResult(
            query=query,
            search_mode=search_mode,
            rerank_applied=False,
            before_rerank=before_rerank,
            after_rerank=[],
        )

    async def dual_search(
        self,
        query: str,
        *,
        top_k_entities: int | None = None,
        top_k_relations: int | None = None,
    ) -> DualLevelKGResult:
        """Run dual-level KG retrieval (entity + relation paths, graph hydrate, source chunks).

        :param query: Natural-language query
        :param top_k_entities: Override default entity kNN size
        :param top_k_relations: Override default relation kNN size
        :returns: Structured KG hits and token-budgeted ``formatted_context``
        :raises RuntimeError: When ``kg_retrieval_service`` was not injected
        """
        if self.kg_retrieval_service is None:
            raise RuntimeError(
                "Dual-level KG retrieval is not configured. Enable KG__ENABLED and wire "
                "DualLevelKGRetrievalService into SearchService."
            )

        logger.info("Dual-level KG search for query: %s...", query[:100])
        result = await self.kg_retrieval_service.retrieve(
            query,
            top_k_entities=top_k_entities,
            top_k_relations=top_k_relations,
        )
        logger.info(
            "Dual-level KG search complete: entities=%d, relations=%d, chunks=%d",
            len(result.entities),
            len(result.relations),
            len(result.chunks),
        )
        return result

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
