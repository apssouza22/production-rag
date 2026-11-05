"""Low-level OpenSearch transport client (connection, indices, search, writes)."""

import logging
from typing import Any

from opensearchpy import OpenSearch

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """Parameterised OpenSearch wrapper without domain-specific index semantics."""

    def __init__(self, host: str):
        self.host = host
        self.client = OpenSearch(
            hosts=[host],
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False,
        )
        logger.info("OpenSearch client initialized with host: %s", host)

    def health_check(self) -> bool:
        """Check if OpenSearch cluster is healthy."""
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return False

    def index_exists(self, index: str) -> bool:
        return bool(self.client.indices.exists(index=index))

    def get_index_stats(self, index: str) -> dict[str, Any]:
        """Get statistics for an index."""
        try:
            if not self.index_exists(index):
                return {"index_name": index, "exists": False, "document_count": 0}

            stats_response = self.client.indices.stats(index=index)
            index_stats = stats_response["indices"][index]["total"]

            return {
                "index_name": index,
                "exists": True,
                "document_count": index_stats["docs"]["count"],
                "deleted_count": index_stats["docs"]["deleted"],
                "size_in_bytes": index_stats["store"]["size_in_bytes"],
            }

        except Exception as e:
            logger.error("Error getting index stats: %s", e)
            return {"index_name": index, "exists": False, "document_count": 0, "error": str(e)}

    def create_index(self, index: str, body: dict[str, Any], *, force: bool = False) -> bool:
        """Create an index. Returns True if created, False if already exists."""
        try:
            if force and self.index_exists(index):
                self.client.indices.delete(index=index)
                logger.info("Deleted existing index: %s", index)

            if not self.index_exists(index):
                self.client.indices.create(index=index, body=body)
                logger.info("Created index: %s", index)
                return True

            logger.info("Index already exists: %s", index)
            return False

        except Exception as e:
            if "resource_already_exists_exception" in str(e):
                logger.info("Index already exists (created by another worker): %s", index)
                return False
            logger.error("Error creating index: %s", e)
            raise

    def ensure_index_mapping(self, index: str, body: dict[str, Any]) -> None:
        """Apply mapping properties to an existing index (adds new fields only)."""
        if not self.index_exists(index):
            return

        properties = body.get("mappings", {}).get("properties")
        if not properties:
            return

        try:
            self.client.indices.put_mapping(index=index, body={"properties": properties})
            logger.info("Updated index mapping for %s", index)
        except Exception as e:
            logger.error("Failed to update index mapping for %s: %s", index, e)
            raise

    def ensure_search_pipeline(
        self,
        pipeline_id: str,
        body: dict[str, Any],
        *,
        force: bool = False,
    ) -> bool:
        """Create a search pipeline if missing. Returns True if created."""
        try:
            if force:
                try:
                    self.client.ingest.get_pipeline(id=pipeline_id)
                    self.client.ingest.delete_pipeline(id=pipeline_id)
                    logger.info("Deleted existing search pipeline: %s", pipeline_id)
                except Exception:
                    pass

            try:
                self.client.ingest.get_pipeline(id=pipeline_id)
                logger.info("Search pipeline already exists: %s", pipeline_id)
                return False
            except Exception:
                pass

            self.client.transport.perform_request("PUT", f"/_search/pipeline/{pipeline_id}", body=body)
            logger.info("Created search pipeline: %s", pipeline_id)
            return True

        except Exception as e:
            logger.error("Error creating search pipeline: %s", e)
            raise

    @staticmethod
    def normalize_search_hits(
        response: dict[str, Any],
        *,
        min_score: float = 0.0,
        total_from_filtered_count: bool = False,
    ) -> dict[str, Any]:
        """Map an OpenSearch search response to ``{total, hits}`` with score and chunk_id."""
        results: dict[str, Any] = {"total": response["hits"]["total"]["value"], "hits": []}

        for hit in response["hits"]["hits"]:
            if hit["_score"] < min_score:
                continue

            chunk = dict(hit["_source"])
            chunk["score"] = hit["_score"]
            chunk["chunk_id"] = hit["_id"]

            if "highlight" in hit:
                chunk["highlights"] = hit["highlight"]

            results["hits"].append(chunk)

        if total_from_filtered_count:
            results["total"] = len(results["hits"])

        return results

    def search(
        self,
        index: str,
        body: dict[str, Any],
        *,
        params: dict[str, Any] | None = None,
        min_score: float = 0.0,
        total_from_filtered_count: bool = False,
    ) -> dict[str, Any]:
        """Execute search and return normalised hit list."""
        try:
            response = self.client.search(index=index, body=body, params=params or {})
            return self.normalize_search_hits(
                response,
                min_score=min_score,
                total_from_filtered_count=total_from_filtered_count,
            )
        except Exception as e:
            logger.error("Search error: %s", e)
            return {"total": 0, "hits": []}

    def index_document(self, index: str, body: dict[str, Any], *, refresh: bool = True) -> bool:
        try:
            response = self.client.index(index=index, body=body, refresh=refresh)
            return response["result"] in ["created", "updated"]
        except Exception as e:
            logger.error("Error indexing document: %s", e)
            return False

    def bulk_index(self, index: str, sources: list[dict[str, Any]], *, refresh: bool = True) -> dict[str, int]:
        from opensearchpy import helpers

        try:
            actions = [{"_index": index, "_source": source} for source in sources]
            success, failed = helpers.bulk(self.client, actions, refresh=refresh)
            logger.info("Bulk indexed %s documents, %s failed", success, len(failed))
            return {"success": success, "failed": len(failed)}
        except Exception as e:
            logger.error("Bulk indexing error: %s", e)
            raise

    def delete_by_query(self, index: str, query: dict[str, Any], *, refresh: bool = True) -> int:
        try:
            response = self.client.delete_by_query(index=index, body={"query": query}, refresh=refresh)
            return int(response.get("deleted", 0))
        except Exception as e:
            logger.error("Delete by query error: %s", e)
            return 0

    def count(self, index: str, body: dict[str, Any] | None = None) -> int:
        response = self.client.count(index=index, body=body or {})
        return int(response["count"])
