"""Unified OpenSearch client supporting both simple BM25 and hybrid search."""

import logging
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch

from src.config import Settings

from .index_config_hybrid import ARXIV_PAPERS_CHUNKS_MAPPING, HYBRID_RRF_PIPELINE
from .query_builder import QueryBuilder

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """OpenSearch client supporting BM25 and hybrid search with native RRF."""

    def __init__(self, host: str, settings: Settings):
        self.host = host
        self.settings = settings
        self.index_name = f"{settings.opensearch.index_name}-{settings.opensearch.chunk_index_suffix}"
        self.ingestion_name = settings.opensearch.ingestion_name

        self.client = OpenSearch(
            hosts=[host],
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )

        logger.info(f"OpenSearch client initialized with host: {host}")

    def health_check(self) -> bool:
        """Check if OpenSearch cluster is healthy."""
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for the hybrid index."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                return {"index_name": self.index_name, "exists": False, "document_count": 0}

            stats_response = self.client.indices.stats(index=self.index_name)
            index_stats = stats_response["indices"][self.index_name]["total"]

            return {
                "index_name": self.index_name,
                "exists": True,
                "document_count": index_stats["docs"]["count"],
                "deleted_count": index_stats["docs"]["deleted"],
                "size_in_bytes": index_stats["store"]["size_in_bytes"],
            }

        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"index_name": self.index_name, "exists": False, "document_count": 0, "error": str(e)}

    def setup_indices(self, force: bool = False) -> Dict[str, bool]:
        """Setup the hybrid search index and RRF pipeline."""
        results = {}
        results["hybrid_index"] = self._create_hybrid_index(force)
        results["rrf_pipeline"] = self._create_rrf_pipeline(force)
        return results

    def _create_hybrid_index(self, force: bool = False) -> bool:
        """Create hybrid index for all search types (BM25, vector, hybrid).

        :param force: If True, recreate index even if it exists
        :returns: True if created, False if already exists
        """
        try:
            if force and self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"Deleted existing hybrid index: {self.index_name}")

            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(index=self.index_name, body=ARXIV_PAPERS_CHUNKS_MAPPING)
                logger.info(f"Created hybrid index: {self.index_name}")
                return True

            logger.info(f"Hybrid index already exists: {self.index_name}")
            return False

        except Exception as e:
            # Handle race condition when multiple workers start simultaneously:
            # all check exists() -> False, all try to create, only one succeeds.
            if "resource_already_exists_exception" in str(e):
                logger.info(f"Hybrid index already exists (created by another worker): {self.index_name}")
                return False
            logger.error(f"Error creating hybrid index: {e}")
            raise

    def _create_rrf_pipeline(self, force: bool = False) -> bool:
        """Create RRF search pipeline for native hybrid search.

        :param force: If True, recreate pipeline even if it exists
        :returns: True if created, False if already exists
        """
        try:
            pipeline_id = HYBRID_RRF_PIPELINE["id"]

            if force:
                try:
                    self.client.ingest.get_pipeline(id=pipeline_id)
                    self.client.ingest.delete_pipeline(id=pipeline_id)
                    logger.info(f"Deleted existing RRF pipeline: {pipeline_id}")
                except Exception:
                    pass

            try:
                self.client.ingest.get_pipeline(id=pipeline_id)
                logger.info(f"RRF pipeline already exists: {pipeline_id}")
                return False
            except Exception:
                pass
            pipeline_body = {
                "description": HYBRID_RRF_PIPELINE["description"],
                "phase_results_processors": HYBRID_RRF_PIPELINE["phase_results_processors"],
            }

            self.client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_id}", body=pipeline_body)

            logger.info(f"Created RRF search pipeline: {pipeline_id}")
            return True

        except Exception as e:
            logger.error(f"Error creating RRF pipeline: {e}")
            raise

    def _resolve_ingestion_name(self, ingestion_name: Optional[str] = None) -> Optional[str]:
        """Return the ingestion partition to filter on, if any."""
        if ingestion_name is not None:
            return ingestion_name or None
        return self.ingestion_name or None

    def _build_search_filters(
        self,
        categories: Optional[List[str]] = None,
        ingestion_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        filters: List[Dict[str, Any]] = []
        resolved_ingestion = self._resolve_ingestion_name(ingestion_name)
        if resolved_ingestion:
            filters.append({"term": {"ingestion_name": resolved_ingestion}})
        if categories:
            filters.append({"terms": {"categories": categories}})
        return filters

    def _wrap_query_with_filters(self, query: Dict[str, Any], filters: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not filters:
            return query
        return {"bool": {"must": [query], "filter": filters}}

    def keyword_search(
        self, query: str, size: int = 10, from_: int = 0, categories: Optional[List[str]] = None, latest: bool = True
    ) -> Dict[str, Any]:
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
        query_embedding: List[float],
        size: int = 10,
        categories: Optional[List[str]] = None,
        ingestion_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pure vector search on chunks.

        :param ingestion_name:
        :param query_embedding: Query embedding vector
        :param size: Number of results
        :param categories: Optional category filter
        :returns: Search results
        """
        try:
            filter_clause = self._build_search_filters(categories=categories, ingestion_name=ingestion_name)

            search_body = {
                "size": size,
                "query": {"knn": {"embedding": {"vector": query_embedding, "k": size}}},
                "_source": {"excludes": ["embedding"]},
            }

            if filter_clause:
                search_body["query"] = self._wrap_query_with_filters(search_body["query"], filter_clause)

            response = self.client.search(index=self.index_name, body=search_body)

            results = {"total": response["hits"]["total"]["value"], "hits": []}

            for hit in response["hits"]["hits"]:
                chunk = hit["_source"]
                chunk["score"] = hit["_score"]
                chunk["chunk_id"] = hit["_id"]
                results["hits"].append(chunk)

            return results

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return {"total": 0, "hits": []}

    def search_unified(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        size: int = 10,
        from_: int = 0,
        categories: Optional[List[str]] = None,
        ingestion_name: Optional[str] = None,
        latest: bool = False,
        use_hybrid: bool = True,
        min_score: float = 0.0,
    ) -> Dict[str, Any]:
        """Unified search method supporting BM25, vector, and hybrid modes.

        :param ingestion_name:
        :param query: Text query for search
        :param query_embedding: Optional embedding for vector/hybrid search
        :param size: Number of results to return
        :param from_: Offset for pagination
        :param categories: Optional category filter
        :param latest: Sort by date instead of relevance
        :param use_hybrid: If True and embedding provided, use hybrid search
        :param min_score: Minimum score threshold
        :returns: Search results
        """
        try:
            # If no embedding provided or hybrid disabled, use BM25 only
            if not query_embedding or not use_hybrid:
                return self._search_bm25_only(
                    query=query,
                    size=size,
                    from_=from_,
                    categories=categories,
                    ingestion_name=ingestion_name,
                    latest=latest,
                )

            # Use native OpenSearch hybrid search with RRF pipeline (reciprocal rank fusion for hybrid search)
            return self._search_hybrid_native(
                query=query,
                query_embedding=query_embedding,
                size=size,
                categories=categories,
                ingestion_name=ingestion_name,
                min_score=min_score,
            )

        except Exception as e:
            logger.error(f"Unified search error: {e}")
            return {"total": 0, "hits": []}

    def _search_bm25_only(
        self,
        query: str,
        size: int,
        from_: int,
        categories: Optional[List[str]],
        ingestion_name: Optional[str],
        latest: bool,
    ) -> Dict[str, Any]:
        """Pure BM25 search implementation."""
        builder = QueryBuilder(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            ingestion_name=self._resolve_ingestion_name(ingestion_name),
            latest_papers=latest,
            search_chunks=True,  # Enable chunk search mode
        )
        search_body = builder.build()

        response = self.client.search(index=self.index_name, body=search_body)

        results = {"total": response["hits"]["total"]["value"], "hits": []}

        for hit in response["hits"]["hits"]:
            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]

            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]

            results["hits"].append(chunk)

        logger.info(f"BM25 search for '{query[:50]}...' returned {results['total']} results")
        return results

    def _search_hybrid_native(
        self,
        query: str,
        query_embedding: List[float],
        size: int,
        categories: Optional[List[str]],
        ingestion_name: Optional[str],
        min_score: float,
    ) -> Dict[str, Any]:
        """Native OpenSearch hybrid search with RRF pipeline (reciprocal rank fusion for hybrid search)."""
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
        embedding_query = {"knn": {"embedding": {"vector": query_embedding, "k": size * 2}}}

        # A hybrid query must stay at the top level, so filters are applied to each
        # sub-query instead. The BM25 side is already filtered by the query builder.
        filter_clauses = self._build_search_filters(categories=categories, ingestion_name=ingestion_name)
        if filter_clauses:
            embedding_query = self._wrap_query_with_filters(embedding_query, filter_clauses)

        hybrid_query: Dict[str, Any] = {"hybrid": {"queries": [bm25_query, embedding_query]}}

        search_body = {
            "size": size,
            "query": hybrid_query,
            "_source": bm25_search_body["_source"],
            "highlight": bm25_search_body["highlight"],
        }

        # Execute search with RRF pipeline
        response = self.client.search(
            index=self.index_name, body=search_body, params={"search_pipeline": HYBRID_RRF_PIPELINE["id"]}
        )

        results = {"total": response["hits"]["total"]["value"], "hits": []}

        for hit in response["hits"]["hits"]:
            if hit["_score"] < min_score:
                continue

            chunk = hit["_source"]
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]

            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]

            results["hits"].append(chunk)

        results["total"] = len(results["hits"])
        logger.info(f"Native hybrid search for '{query[:50]}...' returned {results['total']} results")
        return results

    def index_chunk(self, chunk_data: Dict[str, Any], embedding: List[float]) -> bool:
        """Index a single chunk with its embedding.

        :param chunk_data: Chunk data dictionary
        :param embedding: Embedding vector
        :returns: True if successful
        """
        try:
            chunk_data["embedding"] = embedding

            response = self.client.index(index=self.index_name, body=chunk_data, refresh=True)

            return response["result"] in ["created", "updated"]

        except Exception as e:
            logger.error(f"Error indexing chunk: {e}")
            return False

    def bulk_index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Bulk index multiple chunks with embeddings.

        :param chunks: List of dicts with 'chunk_data' and 'embedding'
        :returns: Statistics
        """
        from opensearchpy import helpers

        try:
            actions = []
            for chunk in chunks:
                chunk_data = chunk["chunk_data"].copy()
                chunk_data["embedding"] = chunk["embedding"]

                action = {"_index": self.index_name, "_source": chunk_data}
                actions.append(action)

            success, failed = helpers.bulk(self.client, actions, refresh=True)

            logger.info(f"Bulk indexed {success} chunks, {len(failed)} failed")
            return {"success": success, "failed": len(failed)}

        except Exception as e:
            logger.error(f"Bulk chunk indexing error: {e}")
            raise

    def delete_paper_chunks(self, arxiv_id: str, ingestion_name: Optional[str] = None) -> bool:
        """Delete all chunks for a specific paper.

        :param arxiv_id: ArXiv ID of the paper
        :param ingestion_name: Limit deletion to this ingestion partition (defaults to configured name)
        :returns: True if deletion was successful
        """
        try:
            must_clauses: List[Dict[str, Any]] = [{"term": {"arxiv_id": arxiv_id}}]
            resolved_ingestion = self._resolve_ingestion_name(ingestion_name)
            if resolved_ingestion:
                must_clauses.append({"term": {"ingestion_name": resolved_ingestion}})

            response = self.client.delete_by_query(
                index=self.index_name,
                body={"query": {"bool": {"must": must_clauses}}},
                refresh=True,
            )

            deleted = response.get("deleted", 0)
            logger.info(f"Deleted {deleted} chunks for paper {arxiv_id}")
            return deleted > 0

        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False

    def get_chunks_by_paper(self, arxiv_id: str, ingestion_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all chunks for a specific paper.

        :param arxiv_id: ArXiv ID of the paper
        :param ingestion_name: Limit results to this ingestion partition (defaults to configured name)
        :returns: List of chunks sorted by chunk_index
        """
        try:
            must_clauses: List[Dict[str, Any]] = [{"term": {"arxiv_id": arxiv_id}}]
            resolved_ingestion = self._resolve_ingestion_name(ingestion_name)
            if resolved_ingestion:
                must_clauses.append({"term": {"ingestion_name": resolved_ingestion}})

            search_body = {
                "query": {"bool": {"must": must_clauses}},
                "size": 1000,
                "sort": [{"chunk_index": "asc"}],
                "_source": {"excludes": ["embedding"]},
            }

            response = self.client.search(index=self.index_name, body=search_body)

            chunks = []
            for hit in response["hits"]["hits"]:
                chunk = hit["_source"]
                chunk["chunk_id"] = hit["_id"]
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error getting chunks: {e}")
            return []

    def get_chunks_by_chunk_keys(
        self,
        chunk_keys: list[str],
        *,
        batch_size: int = 50,
        ingestion_name: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        """Load chunk documents by stable keys arxiv_id:chunk_index."""
        if not chunk_keys:
            return []

        resolved_ingestion = self._resolve_ingestion_name(ingestion_name)
        results: list[Dict[str, Any]] = []
        key_order = {key: idx for idx, key in enumerate(chunk_keys)}

        for offset in range(0, len(chunk_keys), batch_size):
            batch = chunk_keys[offset : offset + batch_size]
            should_clauses: List[Dict[str, Any]] = []
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
                arxiv_id = chunk.get("arxiv_id", "")
                chunk_index = chunk.get("chunk_index", 0)
                chunk_key = f"{arxiv_id}:{chunk_index}"
                chunk["chunk_id"] = hit["_id"]
                chunk["chunk_key"] = chunk_key
                chunk["_sort_order"] = key_order.get(chunk_key, 9999)
                results.append(chunk)

        results.sort(key=lambda c: c.get("_sort_order", 9999))
        for chunk in results:
            chunk.pop("_sort_order", None)
        return results
