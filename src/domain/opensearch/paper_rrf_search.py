"""ArXiv paper chunk search: BM25, vector kNN, and native RRF hybrid retrieval."""

import logging
from typing import Any

from opensearchpy import OpenSearch

from src.config import Settings

from .client import OpenSearchClient
from .index_config_hybrid import ARXIV_PAPERS_CHUNKS_MAPPING, HYBRID_RRF_PIPELINE
from .query_builder import QueryBuilder

logger = logging.getLogger(__name__)


class PaperRrfSearch:
    """Hybrid chunk index for arXiv papers (BM25 + vector fused via RRF pipeline)."""

    def __init__(self, opensearch: OpenSearchClient, settings: Settings):
        self._opensearch = opensearch
        self.settings = settings
        self.index_name = f"{settings.opensearch.index_name}-{settings.opensearch.chunk_index_suffix}"
        self.ingestion_name = settings.opensearch.ingestion_name

    @property
    def client(self) -> OpenSearch:
        return self._opensearch.client

    def health_check(self) -> bool:
        return self._opensearch.health_check()

    def get_index_stats(self) -> dict[str, Any]:
        return self._opensearch.get_index_stats(self.index_name)

    def setup_indices(self, force: bool = False) -> dict[str, bool]:
        """Setup the hybrid search index and RRF pipeline."""
        hybrid_index = self._opensearch.create_index(
            self.index_name,
            ARXIV_PAPERS_CHUNKS_MAPPING,
            force=force,
        )
        pipeline_body = {
            "description": HYBRID_RRF_PIPELINE["description"],
            "phase_results_processors": HYBRID_RRF_PIPELINE["phase_results_processors"],
        }
        rrf_pipeline = self._opensearch.ensure_search_pipeline(
            HYBRID_RRF_PIPELINE["id"],
            pipeline_body,
            force=force,
        )
        return {"hybrid_index": hybrid_index, "rrf_pipeline": rrf_pipeline}

    def _resolve_ingestion_name(self, ingestion_name: str | None = None) -> str | None:
        if ingestion_name is not None:
            return ingestion_name or None
        return self.ingestion_name or None

    def _build_search_filters(
        self,
        categories: list[str] | None = None,
        ingestion_name: str | None = None,
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        resolved_ingestion = self._resolve_ingestion_name(ingestion_name)
        if resolved_ingestion:
            filters.append({"term": {"ingestion_name": resolved_ingestion}})
        if categories:
            filters.append({"terms": {"categories": categories}})
        return filters

    @staticmethod
    def _wrap_query_with_filters(query: dict[str, Any], filters: list[dict[str, Any]]) -> dict[str, Any]:
        if not filters:
            return query
        return {"bool": {"must": [query], "filter": filters}}

    def keyword_search(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        categories: list[str] | None = None,
        latest: bool = True,
    ) -> dict[str, Any]:
        """BM25 search for papers."""
        return self._search_bm25_only(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            ingestion_name=None,
            latest=latest,
        )

    def vector_search(
        self,
        query_embedding: list[float],
        size: int = 10,
        categories: list[str] | None = None,
        ingestion_name: str | None = None,
    ) -> dict[str, Any]:
        """Pure vector search on chunks."""
        filter_clause = self._build_search_filters(categories=categories, ingestion_name=ingestion_name)

        search_body: dict[str, Any] = {
            "size": size,
            "query": {"knn": {"embedding": {"vector": query_embedding, "k": size}}},
            "_source": {"excludes": ["embedding"]},
        }

        if filter_clause:
            search_body["query"] = self._wrap_query_with_filters(search_body["query"], filter_clause)

        return self._opensearch.search(self.index_name, search_body)

    def search_unified(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        size: int = 10,
        from_: int = 0,
        categories: list[str] | None = None,
        ingestion_name: str | None = None,
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
                    ingestion_name=ingestion_name,
                    latest=latest,
                )

            return self._search_hybrid_native(
                query=query,
                query_embedding=query_embedding,
                size=size,
                categories=categories,
                ingestion_name=ingestion_name,
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
        ingestion_name: str | None,
        latest: bool,
    ) -> dict[str, Any]:
        builder = QueryBuilder(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            ingestion_name=self._resolve_ingestion_name(ingestion_name),
            latest_papers=latest,
            search_chunks=True,
        )
        search_body = builder.build()
        results = self._opensearch.search(self.index_name, search_body)
        logger.info("BM25 search for '%s...' returned %s results", query[:50], results["total"])
        return results

    def _search_hybrid_native(
        self,
        query: str,
        query_embedding: list[float],
        size: int,
        categories: list[str] | None,
        ingestion_name: str | None,
        min_score: float,
    ) -> dict[str, Any]:
        builder = QueryBuilder(
            query=query,
            size=size * 2,
            from_=0,
            categories=categories,
            ingestion_name=self._resolve_ingestion_name(ingestion_name),
            latest_papers=False,
            search_chunks=True,
        )
        bm25_search_body = builder.build()

        bm25_query = bm25_search_body["query"]
        embedding_query: dict[str, Any] = {"knn": {"embedding": {"vector": query_embedding, "k": size * 2}}}

        filter_clauses = self._build_search_filters(categories=categories, ingestion_name=ingestion_name)
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
            params={"search_pipeline": HYBRID_RRF_PIPELINE["id"]},
            min_score=min_score,
            total_from_filtered_count=True,
        )
        logger.info("Native hybrid search for '%s...' returned %s results", query[:50], results["total"])
        return results

    def index_chunk(self, chunk_data: dict[str, Any], embedding: list[float]) -> bool:
        body = {**chunk_data, "embedding": embedding}
        return self._opensearch.index_document(self.index_name, body)

    def bulk_index_chunks(self, chunks: list[dict[str, Any]]) -> dict[str, int]:
        sources = []
        for chunk in chunks:
            chunk_data = chunk["chunk_data"].copy()
            chunk_data["embedding"] = chunk["embedding"]
            sources.append(chunk_data)
        return self._opensearch.bulk_index(self.index_name, sources)

    def delete_paper_chunks(self, arxiv_id: str, ingestion_name: str | None = None) -> bool:
        must_clauses: list[dict[str, Any]] = [{"term": {"arxiv_id": arxiv_id}}]
        resolved_ingestion = self._resolve_ingestion_name(ingestion_name)
        if resolved_ingestion:
            must_clauses.append({"term": {"ingestion_name": resolved_ingestion}})

        deleted = self._opensearch.delete_by_query(self.index_name, {"bool": {"must": must_clauses}})
        logger.info("Deleted %s chunks for paper %s", deleted, arxiv_id)
        return deleted > 0

    def get_chunks_by_paper(self, arxiv_id: str, ingestion_name: str | None = None) -> list[dict[str, Any]]:
        must_clauses: list[dict[str, Any]] = [{"term": {"arxiv_id": arxiv_id}}]
        resolved_ingestion = self._resolve_ingestion_name(ingestion_name)
        if resolved_ingestion:
            must_clauses.append({"term": {"ingestion_name": resolved_ingestion}})

        search_body = {
            "query": {"bool": {"must": must_clauses}},
            "size": 1000,
            "sort": [{"chunk_index": "asc"}],
            "_source": {"excludes": ["embedding"]},
        }

        try:
            response = self.client.search(index=self.index_name, body=search_body)
        except Exception as e:
            logger.error("Error getting chunks: %s", e)
            return []

        chunks = []
        for hit in response["hits"]["hits"]:
            chunk = hit["_source"]
            chunk["chunk_id"] = hit["_id"]
            chunks.append(chunk)
        return chunks

    def get_chunks_by_chunk_keys(
        self,
        chunk_keys: list[str],
        *,
        batch_size: int = 50,
        ingestion_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load chunk documents by stable keys arxiv_id:chunk_index."""
        if not chunk_keys:
            return []

        resolved_ingestion = self._resolve_ingestion_name(ingestion_name)
        results: list[dict[str, Any]] = []
        key_order = {key: idx for idx, key in enumerate(chunk_keys)}

        for offset in range(0, len(chunk_keys), batch_size):
            batch = chunk_keys[offset : offset + batch_size]
            should_clauses: list[dict[str, Any]] = []
            for key in batch:
                if ":" not in key:
                    continue
                arxiv_id, index_str = key.rsplit(":", 1)
                try:
                    chunk_index = int(index_str)
                except ValueError:
                    continue
                must = [
                    {"term": {"arxiv_id": arxiv_id}},
                    {"term": {"chunk_index": chunk_index}},
                ]
                if resolved_ingestion:
                    must.append({"term": {"ingestion_name": resolved_ingestion}})
                should_clauses.append({"bool": {"must": must}})

            if not should_clauses:
                continue

            search_body = {
                "query": {"bool": {"should": should_clauses, "minimum_should_match": 1}},
                "size": len(should_clauses),
                "_source": {"excludes": ["embedding"]},
            }
            response = self.client.search(index=self.index_name, body=search_body)
            for hit in response["hits"]["hits"]:
                chunk = hit["_source"]
                hit_arxiv_id = chunk.get("arxiv_id", "")
                hit_chunk_index = chunk.get("chunk_index", 0)
                chunk_key = f"{hit_arxiv_id}:{hit_chunk_index}"
                chunk["chunk_id"] = hit["_id"]
                chunk["chunk_key"] = chunk_key
                chunk["_sort_order"] = key_order.get(chunk_key, 9999)
                results.append(chunk)

        results.sort(key=lambda c: c.get("_sort_order", 9999))
        for chunk in results:
            chunk.pop("_sort_order", None)
        return results
