import logging
from typing import Any

from opensearchpy import OpenSearch, helpers

from src.config import Settings
from src.domain.kg.index_config import KG_ENTITY_MAPPING, KG_RELATION_MAPPING

logger = logging.getLogger(__name__)


class KgSearchClient:
    """OpenSearch client for KG entity and relation indices (parallel to ``arxiv-papers-chunks``).

    Document IDs: ``entity_id`` for nodes, ``source->target`` for edges. All queries
    filter on ``ingestion_name`` so dev/staging partitions do not cross-contaminate.
    """

    def __init__(self, client: OpenSearch, settings: Settings, ingestion_name: str):
        self.client = client
        self.settings = settings
        self.ingestion_name = ingestion_name
        base = settings.opensearch.index_name
        self.entity_index = f"{base}-{settings.kg.entity_index_suffix}"
        self.relation_index = f"{base}-{settings.kg.relation_index_suffix}"
        self.vector_dimension = settings.opensearch.vector_dimension

    def setup_indices(self, force: bool = False) -> dict[str, bool]:
        return {
            "entities": self._ensure_index(self.entity_index, KG_ENTITY_MAPPING, force),
            "relations": self._ensure_index(self.relation_index, KG_RELATION_MAPPING, force),
        }

    def _ensure_index(self, index_name: str, body: dict[str, Any], force: bool) -> bool:
        if force and self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)
        if not self.client.indices.exists(index=index_name):
            mapping = {
                "settings": body["settings"],
                "mappings": {
                    "dynamic": body["mappings"]["dynamic"],
                    "properties": {k: (v.copy() if k == "embedding" else v) for k, v in body["mappings"]["properties"].items()},
                },
            }
            mapping["mappings"]["properties"]["embedding"]["dimension"] = self.vector_dimension
            self.client.indices.create(index=index_name, body=mapping)
            logger.info("Created KG index %s", index_name)
            return True
        return False

    def bulk_upsert_entities(self, docs: list[dict[str, Any]]) -> int:
        return self._bulk_upsert(self.entity_index, docs, id_field="entity_id")

    def bulk_upsert_relations(self, docs: list[dict[str, Any]]) -> int:
        return self._bulk_upsert(self.relation_index, docs, id_field="relation_id")

    def _bulk_upsert(self, index: str, docs: list[dict[str, Any]], id_field: str) -> int:
        if not docs:
            return 0
        actions = []
        for doc in docs:
            doc_id = doc[id_field]
            actions.append(
                {
                    "_index": index,
                    "_id": doc_id,
                    "_source": doc,
                }
            )
        success, _failed = helpers.bulk(self.client, actions, refresh=True)
        return int(success)

    def search_entities(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        query_text: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._vector_search_rrf(
            self.entity_index,
            query_embedding,
            top_k,
            query_text,
            bm25_fields=["entity_name^2", "description"],
        )

    def search_relations(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        query_text: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._vector_search_rrf(
            self.relation_index,
            query_embedding,
            top_k,
            query_text,
            bm25_fields=["keywords^2", "description"],
        )

    @property
    def _rrf_pipeline_id(self) -> str:
        return self.settings.opensearch.rrf_pipeline_name

    def _recall_window_size(self, top_k: int) -> int:
        mult = self.settings.opensearch.hybrid_search_size_multiplier
        return max(top_k * mult, top_k)

    def _vector_search_rrf(
        self,
        index: str,
        query_embedding: list[float],
        top_k: int,
        query_text: str | None,
        *,
        bm25_fields: list[str],
    ) -> list[dict[str, Any]]:
        """kNN-only, or BM25 + kNN fused via the cluster RRF search pipeline when ``query_text`` is set."""
        text = (query_text or "").strip()
        if not text:
            return self._knn_search(index, query_embedding, top_k)
        return self._hybrid_rrf_search(index, query_embedding, top_k, text, bm25_fields=bm25_fields)

    def _ingestion_filter_clause(self) -> dict[str, Any]:
        return {"term": {"ingestion_name": self.ingestion_name}}

    def _bm25_subquery(self, query_text: str, bm25_fields: list[str]) -> dict[str, Any]:
        return {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": bm25_fields,
                            "type": "best_fields",
                            "operator": "or",
                            "fuzziness": "AUTO",
                            "prefix_length": 2,
                        }
                    }
                ],
                "filter": [self._ingestion_filter_clause()],
            }
        }

    def _knn_subquery(self, query_embedding: list[float], recall_k: int) -> dict[str, Any]:
        return {
            "bool": {
                "must": [
                    self._ingestion_filter_clause(),
                    {"knn": {"embedding": {"vector": query_embedding, "k": recall_k}}},
                ]
            }
        }

    def _hybrid_rrf_search(
        self,
        index: str,
        query_embedding: list[float],
        top_k: int,
        query_text: str,
        *,
        bm25_fields: list[str],
    ) -> list[dict[str, Any]]:
        recall_k = self._recall_window_size(top_k)
        body: dict[str, Any] = {
            "size": top_k,
            "query": {
                "hybrid": {
                    "queries": [
                        self._bm25_subquery(query_text, bm25_fields),
                        self._knn_subquery(query_embedding, recall_k),
                    ]
                }
            },
            "_source": {"excludes": ["embedding"]},
        }
        response = self.client.search(
            index=index,
            body=body,
            params={"search_pipeline": self._rrf_pipeline_id},
        )
        return self._hits_to_sources(response)

    def _knn_search(self, index: str, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"ingestion_name": self.ingestion_name}},
                        {"knn": {"embedding": {"vector": query_embedding, "k": top_k}}},
                    ]
                }
            },
            "_source": {"excludes": ["embedding"]},
        }
        response = self.client.search(index=index, body=body)
        return self._hits_to_sources(response)

    @staticmethod
    def _hits_to_sources(response: dict[str, Any]) -> list[dict[str, Any]]:
        hits: list[dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = dict(hit.get("_source", {}))
            source["score"] = hit.get("_score", 0.0)
            hits.append(source)
        return hits

    def delete_vectors_for_arxiv(self, entity_ids: list[str], relation_ids: list[str]) -> None:
        for index, ids in ((self.entity_index, entity_ids), (self.relation_index, relation_ids)):
            if not ids:
                continue
            self.client.delete_by_query(
                index=index,
                body={"query": {"ids": {"values": ids}}},
                refresh=True,
            )
