"""Generic document chunk search: BM25, vector kNN, and native RRF hybrid retrieval."""

import logging
from typing import Any

from opensearchpy import OpenSearch

from src.config import Settings
from src.domain.opensearch.client import OpenSearchClient

from .chunk_index_schema import (
    DEFAULT_CHUNK_PROFILE,
    ChunkSearchProfile,
    flatten_hit_metadata,
)
from .indexing import SearchIndexing
from .query_builder import QueryBuilder

logger = logging.getLogger(__name__)


class HybridRrfSearch:
    """Hybrid chunk index (BM25 + vector fused via RRF pipeline) for any document type."""

    def __init__(
        self,
        opensearch: OpenSearchClient,
        settings: Settings,
        *,
        profile: ChunkSearchProfile | None = None,
        ingestion_name: str | None = None,
    ):
        self._opensearch = opensearch
        self.settings = settings
        self.profile = profile or DEFAULT_CHUNK_PROFILE
        self._indexing = SearchIndexing(opensearch, settings, ingestion_name=ingestion_name)

    @property
    def index_name(self) -> str:
        return self._indexing.index_name

    @property
    def ingestion_name(self) -> str | None:
        return self._indexing.ingestion_name

    @ingestion_name.setter
    def ingestion_name(self, value: str | None) -> None:
        self._indexing.ingestion_name = value

    @property
    def _rrf_pipeline_id(self) -> str:
        return self._indexing._rrf_pipeline_id

    @property
    def indexing(self) -> SearchIndexing:
        return self._indexing

    @property
    def client(self) -> OpenSearch:
        return self._opensearch.client

    def health_check(self) -> bool:
        return self._opensearch.health_check()

    def get_index_stats(self) -> dict[str, Any]:
        return self._opensearch.get_index_stats(self.index_name)

    def setup_indices(self, force: bool = False) -> dict[str, bool]:
        return self._indexing.setup_indices(force=force)

    def _build_search_filters(
        self,
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        if self.ingestion_name:
            filters.append({"term": {"ingestion_name": self.ingestion_name}})
        if categories:
            field = self.profile.term_filters.get("categories", "categories")
            filters.append({"terms": {field: categories}})
        return filters

    @staticmethod
    def _wrap_query_with_filters(query: dict[str, Any], filters: list[dict[str, Any]]) -> dict[str, Any]:
        if not filters:
            return query
        return {"bool": {"must": [query], "filter": filters}}

    @staticmethod
    def _flatten_results(results: dict[str, Any]) -> dict[str, Any]:
        hits = [flatten_hit_metadata(h) for h in results.get("hits", [])]
        return {**results, "hits": hits}

    def keyword_search(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        categories: list[str] | None = None,
        latest: bool = True,
    ) -> dict[str, Any]:
        """BM25 search for chunks."""
        return self._search_bm25_only(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            latest=latest,
        )

    def vector_search(
        self,
        query_embedding: list[float],
        size: int = 10,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Pure vector search on chunks."""
        filter_clause = self._build_search_filters(categories=categories)

        search_body: dict[str, Any] = {
            "size": size,
            "query": {"knn": {"embedding": {"vector": query_embedding, "k": size}}},
            "_source": {"excludes": ["embedding"]},
        }

        if filter_clause:
            search_body["query"] = self._wrap_query_with_filters(search_body["query"], filter_clause)

        results = self._opensearch.search(self.index_name, search_body)
        return self._flatten_results(results)

    def search_unified(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        size: int = 10,
        from_: int = 0,
        categories: list[str] | None = None,
        latest: bool = False,
        use_hybrid: bool = True,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Unified search supporting BM25, vector, and RRF hybrid modes."""
        try:
            if not query_embedding or not use_hybrid:
                return self._search_bm25_only(
                    query=query,
                    size=size,
                    from_=from_,
                    categories=categories,
                    latest=latest,
                )

            return self._search_hybrid_native(
                query=query,
                query_embedding=query_embedding,
                size=size,
                categories=categories,
                min_score=min_score,
            )

        except Exception as e:
            logger.error("Unified search error: %s", e)
            return {"total": 0, "hits": []}

    def _search_bm25_only(
        self,
        query: str,
        size: int,
        from_: int,
        categories: list[str] | None,
        latest: bool,
    ) -> dict[str, Any]:
        builder = QueryBuilder(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            ingestion_name=self.ingestion_name or None,
            latest_papers=latest,
            profile=self.profile,
        )
        search_body = builder.build()
        results = self._opensearch.search(self.index_name, search_body)
        logger.info("BM25 search for '%s...' returned %s results", query[:50], results["total"])
        return self._flatten_results(results)

    def _search_hybrid_native(
        self,
        query: str,
        query_embedding: list[float],
        size: int,
        categories: list[str] | None,
        min_score: float,
    ) -> dict[str, Any]:
        builder = QueryBuilder(
            query=query,
            size=size * 2,
            from_=0,
            categories=categories,
            ingestion_name=self.ingestion_name or None,
            latest_papers=False,
            profile=self.profile,
        )
        bm25_search_body = builder.build()

        bm25_query = bm25_search_body["query"]
        embedding_query: dict[str, Any] = {"knn": {"embedding": {"vector": query_embedding, "k": size * 2}}}

        filter_clauses = self._build_search_filters(categories=categories)
        if filter_clauses:
            embedding_query = self._wrap_query_with_filters(embedding_query, filter_clauses)

        hybrid_query: dict[str, Any] = {"hybrid": {"queries": [bm25_query, embedding_query]}}

        search_body = {
            "size": size,
            "query": hybrid_query,
            "_source": bm25_search_body["_source"],
            "highlight": bm25_search_body["highlight"],
        }

        results = self._opensearch.search(
            self.index_name,
            search_body,
            params={"search_pipeline": self._rrf_pipeline_id},
            min_score=min_score,
            total_from_filtered_count=True,
        )
        logger.info("Native hybrid search for '%s...' returned %s results", query[:50], results["total"])
        return self._flatten_results(results)

    def index_chunk(self, chunk_data: dict[str, Any], embedding: list[float]) -> bool:
        return self._indexing.index_chunk(chunk_data, embedding)

    def bulk_index_chunks(self, chunks: list[dict[str, Any]]) -> dict[str, int]:
        return self._indexing.bulk_index_chunks(chunks)

    def delete_document_chunks(self, document_id: str) -> bool:
        return self._indexing.delete_document_chunks(document_id)

    def get_chunks_by_document(self, document_id: str) -> list[dict[str, Any]]:
        return self._indexing.get_chunks_by_document(document_id)

    def get_chunks_by_paper(self, arxiv_id: str) -> list[dict[str, Any]]:
        return self._indexing.get_chunks_by_paper(arxiv_id)

    def get_chunks_by_chunk_keys(
        self,
        chunk_keys: list[str],
        *,
        batch_size: int = 50,
    ) -> list[dict[str, Any]]:
        return self._indexing.get_chunks_by_chunk_keys(chunk_keys, batch_size=batch_size)
