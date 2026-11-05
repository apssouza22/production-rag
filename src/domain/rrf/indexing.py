"""OpenSearch chunk index setup, writes, and document lookups for hybrid RRF search."""

import logging
from typing import Any

from opensearchpy import OpenSearch

from src.config import Settings
from src.domain.opensearch.client import OpenSearchClient

from .chunk_index_schema import ChunkIndexDocument, build_hybrid_chunk_mapping, flatten_hit_metadata
from .pipeline import hybrid_rrf_pipeline_body

logger = logging.getLogger(__name__)


class SearchIndexing:
    """Hybrid chunk index lifecycle: mapping, pipeline, bulk writes, and chunk fetches."""

    def __init__(
        self,
        opensearch: OpenSearchClient,
        settings: Settings,
        *,
        ingestion_name: str | None = None,
    ) -> None:
        self._opensearch = opensearch
        self.settings = settings
        self.index_name = f"{settings.opensearch.index_name}-{settings.opensearch.chunk_index_suffix}"
        self.ingestion_name = (
            settings.opensearch.ingestion_name if ingestion_name is None else ingestion_name
        )
        self._rrf_pipeline_id = settings.opensearch.rrf_pipeline_name

    @property
    def client(self) -> OpenSearch:
        return self._opensearch.client

    def setup_indices(self, force: bool = False) -> dict[str, bool]:
        """Setup the hybrid search index and RRF pipeline."""
        mapping = build_hybrid_chunk_mapping(self.settings.opensearch)
        hybrid_index = self._opensearch.create_index(
            self.index_name,
            mapping,
            force=force,
        )
        if not hybrid_index:
            self._opensearch.ensure_index_mapping(self.index_name, mapping)
        pipeline_body = hybrid_rrf_pipeline_body(self.settings.opensearch)
        rrf_pipeline = self._opensearch.ensure_search_pipeline(
            self._rrf_pipeline_id,
            pipeline_body,
            force=force,
        )
        return {"hybrid_index": hybrid_index, "rrf_pipeline": rrf_pipeline}

    def _prepare_chunk_source(self, chunk_data: dict[str, Any], embedding: list[float]) -> dict[str, Any]:
        doc = ChunkIndexDocument.from_index_payload(chunk_data)
        return doc.to_opensearch_source(embedding=embedding)

    def index_chunk(self, chunk_data: dict[str, Any], embedding: list[float]) -> bool:
        body = self._prepare_chunk_source(chunk_data, embedding)
        return self._opensearch.index_document(self.index_name, body)

    def bulk_index_chunks(self, chunks: list[dict[str, Any]]) -> dict[str, int]:
        sources = []
        for chunk in chunks:
            sources.append(self._prepare_chunk_source(chunk["chunk_data"], chunk["embedding"]))
        return self._opensearch.bulk_index(self.index_name, sources)

    def delete_document_chunks(self, document_id: str) -> bool:
        must_clauses: list[dict[str, Any]] = [{"term": {"document_id": document_id}}]
        if self.ingestion_name:
            must_clauses.append({"term": {"ingestion_name": self.ingestion_name}})

        deleted = self._opensearch.delete_by_query(self.index_name, {"bool": {"must": must_clauses}})
        logger.info("Deleted %s chunks for document %s", deleted, document_id)
        return deleted > 0

    def get_chunks_by_document(self, document_id: str) -> list[dict[str, Any]]:
        must_clauses: list[dict[str, Any]] = [{"term": {"document_id": document_id}}]
        if self.ingestion_name:
            must_clauses.append({"term": {"ingestion_name": self.ingestion_name}})

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
            chunk = flatten_hit_metadata(dict(hit["_source"]))
            chunk["chunk_id"] = hit["_id"]
            chunks.append(chunk)
        return chunks

    def get_chunks_by_paper(self, arxiv_id: str) -> list[dict[str, Any]]:
        return self.get_chunks_by_document(arxiv_id)

    def get_chunks_by_chunk_keys(
        self,
        chunk_keys: list[str],
        *,
        batch_size: int = 50,
    ) -> list[dict[str, Any]]:
        """Load chunk documents by stable keys document_id:chunk_index."""
        if not chunk_keys:
            return []

        results: list[dict[str, Any]] = []
        key_order = {key: idx for idx, key in enumerate(chunk_keys)}

        for offset in range(0, len(chunk_keys), batch_size):
            batch = chunk_keys[offset : offset + batch_size]
            should_clauses: list[dict[str, Any]] = []
            for key in batch:
                if ":" not in key:
                    continue
                document_id, index_str = key.rsplit(":", 1)
                try:
                    chunk_index = int(index_str)
                except ValueError:
                    continue
                must = [
                    {"term": {"document_id": document_id}},
                    {"term": {"chunk_index": chunk_index}},
                ]
                if self.ingestion_name:
                    must.append({"term": {"ingestion_name": self.ingestion_name}})
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
                chunk = flatten_hit_metadata(dict(hit["_source"]))
                hit_document_id = chunk.get("document_id", "")
                hit_chunk_index = chunk.get("chunk_index", 0)
                chunk_key = f"{hit_document_id}:{hit_chunk_index}"
                chunk["chunk_id"] = hit["_id"]
                chunk["chunk_key"] = chunk_key
                chunk["_sort_order"] = key_order.get(chunk_key, 9999)
                results.append(chunk)

        results.sort(key=lambda c: c.get("_sort_order", 9999))
        for chunk in results:
            chunk.pop("_sort_order", None)
        return results
