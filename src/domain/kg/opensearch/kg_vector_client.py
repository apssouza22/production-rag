import logging
from typing import Any

from opensearchpy import OpenSearch, helpers
from src.config import Settings
from src.domain.kg.opensearch.index_config import KG_ENTITY_MAPPING, KG_RELATION_MAPPING

logger = logging.getLogger(__name__)


class KgVectorClient:
    """OpenSearch kNN stores for entity and relation embeddings (parallel to ``arxiv-papers-chunks``).

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

    def knn_search_entities(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        return self._knn_search(self.entity_index, query_embedding, top_k)

    def knn_search_relations(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        return self._knn_search(self.relation_index, query_embedding, top_k)

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
        }
        response = self.client.search(index=index, body=body)
        hits = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
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
