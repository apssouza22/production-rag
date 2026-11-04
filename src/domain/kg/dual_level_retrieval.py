import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from src.config import KGSettings
from src.domain.db.interfaces.base import BaseDatabase
from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.kg.context_format import format_kg_context
from src.domain.kg.extraction import KgExtractionService
from src.domain.kg.kg_vector_client import KgVectorClient
from src.domain.kg.models import KgEdge, KgNode
from src.domain.kg.repository import KgGraphRepository
from src.domain.kg.schemas import (
    DualLevelKGResult,
    KgChunkHit,
    KgEntityHit,
    KgRelationHit,
)
from src.domain.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


@dataclass
class _ChunkKeyCollector:
    """Tracks chunk keys in discovery order (entity paths before relation paths)."""

    _order: dict[str, int] = field(default_factory=dict)
    _counter: int = 0

    def register_entity_keys(self, keys: list[str] | None) -> None:
        for key in keys or []:
            if key not in self._order:
                self._order[key] = self._counter
                self._counter += 1

    def register_relation_keys(self, keys: list[str] | None) -> None:
        for key in keys or []:
            if key not in self._order:
                self._order[key] = self._counter + 1000
                self._counter += 1

    def sorted_keys(self) -> list[str]:
        return sorted(self._order.keys(), key=lambda k: self._order[k])


class DualLevelKGRetrievalService:
    """Query-time LightRAG KG path (no raw-query chunk kNN).

    Flow: dual keywords → embed low-level → entity kNN → graph hydrate + degree rank
    → embed high-level → relation kNN → load edges → resolve ``source_chunk_keys``
    → fetch chunk text from the hybrid index → token-budgeted ``formatted_context``.
    """

    def __init__(
        self,
        database: BaseDatabase,
        opensearch_client: OpenSearchClient,
        kg_vector_client: KgVectorClient,
        extraction_service: KgExtractionService,
        embeddings_client: JinaEmbeddingsClient,
        kg_settings: KGSettings,
        ingestion_name: str,
    ) -> None:
        self._database = database
        self._opensearch_client = opensearch_client
        self._kg_vector_client = kg_vector_client
        self._extraction_service = extraction_service
        self._embeddings_client = embeddings_client
        self._kg_settings = kg_settings
        self._ingestion_name = ingestion_name

    async def retrieve(
        self,
        query: str,
        *,
        top_k_entities: int | None = None,
        top_k_relations: int | None = None,
    ) -> DualLevelKGResult:
        top_k_entities = top_k_entities or self._kg_settings.top_k_entities
        top_k_relations = top_k_relations or self._kg_settings.top_k_relations

        hl, ll, should_abort = await self._resolve_query_keywords(query)
        if should_abort:
            return DualLevelKGResult(query=query, high_level_keywords=hl, low_level_keywords=ll)
        chunk_keys = _ChunkKeyCollector()

        entity_hits = await self._retrieve_entities(ll, top_k_entities, chunk_keys)
        relation_hits = await self._retrieve_relations(hl, top_k_relations, chunk_keys)

        entity_hits.sort(key=lambda e: e.rank, reverse=True)
        relation_hits.sort(key=lambda r: r.rank, reverse=True)

        chunk_hits = self._load_chunk_hits(chunk_keys.sorted_keys())
        formatted = format_kg_context(
            entity_hits,
            relation_hits,
            chunk_hits,
            max_entity_tokens=self._kg_settings.max_entity_tokens,
            max_relation_tokens=self._kg_settings.max_relation_tokens,
            max_total_tokens=self._kg_settings.max_total_tokens,
        )

        return DualLevelKGResult(
            query=query,
            high_level_keywords=hl,
            low_level_keywords=ll,
            entities=entity_hits,
            relations=relation_hits,
            chunks=chunk_hits,
            formatted_context=formatted,
        )

    async def _resolve_query_keywords(self, query: str) -> tuple[list[str], list[str], bool]:
        """Extract dual keywords; third value is ``True`` when retrieval should return without search."""
        keywords = await self._extraction_service.extract_keywords(query)
        hl = keywords.high_level_keywords
        ll = keywords.low_level_keywords

        if hl or ll:
            return hl, ll, False

        stripped = query.strip()
        if len(stripped) < 50:
            return [], [stripped], False
        return hl, ll, True

    async def _retrieve_entities(
        self,
        low_level_keywords: list[str],
        top_k: int,
        chunk_keys: _ChunkKeyCollector,
    ) -> list[KgEntityHit]:
        if not low_level_keywords:
            return []

        embedding = await self._embeddings_client.embed_query(", ".join(low_level_keywords))
        vector_hits = self._kg_vector_client.knn_search_entities(embedding, top_k)
        entity_ids = [hit["entity_id"] for hit in vector_hits if hit.get("entity_id")]

        with self._graph_repository() as repo:
            nodes = repo.get_nodes_by_entity_ids(entity_ids)
            node_map = {n.entity_id: n for n in nodes}
            node_counts = repo.node_counts(entity_ids)

        hits: list[KgEntityHit] = []
        for hit in vector_hits:
            entity_id = hit.get("entity_id")
            if not entity_id or entity_id not in node_map:
                continue
            node = node_map[entity_id]
            hits.append(self._entity_hit_from_node(node, node_counts.get(entity_id, 0)))
            chunk_keys.register_entity_keys(node.source_chunk_keys)
        return hits

    async def _retrieve_relations(
        self,
        high_level_keywords: list[str],
        top_k: int,
        chunk_keys: _ChunkKeyCollector,
    ) -> list[KgRelationHit]:
        if not high_level_keywords:
            return []

        embedding = await self._embeddings_client.embed_query(", ".join(high_level_keywords))
        vector_hits = self._kg_vector_client.knn_search_relations(embedding, top_k)
        pairs = [
            (hit["source_entity_id"], hit["target_entity_id"])
            for hit in vector_hits
            if hit.get("source_entity_id") and hit.get("target_entity_id")
        ]

        with self._graph_repository() as repo:
            edges = repo.get_edges_by_pairs(pairs)
            edge_map = {(e.source_entity_id, e.target_entity_id): e for e in edges}

        hits: list[KgRelationHit] = []
        for hit in vector_hits:
            pair = (hit.get("source_entity_id"), hit.get("target_entity_id"))
            if not pair[0] or not pair[1] or pair not in edge_map:
                continue
            edge = edge_map[pair]
            hits.append(self._relation_hit_from_edge(edge))
            chunk_keys.register_relation_keys(edge.source_chunk_keys)
        return hits

    def _load_chunk_hits(self, chunk_keys: list[str]) -> list[KgChunkHit]:
        os_chunks = self._opensearch_client.get_chunks_by_chunk_keys(
            chunk_keys,
            ingestion_name=self._ingestion_name,
        )
        return [self._chunk_hit_from_opensearch_doc(chunk) for chunk in os_chunks]

    @contextmanager
    def _graph_repository(self) -> Iterator[KgGraphRepository]:
        with self._database.get_session() as session:
            yield KgGraphRepository(session, self._ingestion_name)

    @staticmethod
    def _entity_hit_from_node(node: KgNode, degree: int) -> KgEntityHit:
        return KgEntityHit(
            entity_id=node.entity_id,
            entity_name=node.entity_name,
            entity_type=node.entity_type,
            description=node.description,
            rank=degree,
            source_chunk_keys=list(node.source_chunk_keys or []),
        )

    @staticmethod
    def _relation_hit_from_edge(edge: KgEdge) -> KgRelationHit:
        return KgRelationHit(
            source_entity_id=edge.source_entity_id,
            target_entity_id=edge.target_entity_id,
            description=edge.description,
            keywords=edge.keywords,
            weight=edge.weight,
            rank=int(edge.weight),
            source_chunk_keys=list(edge.source_chunk_keys or []),
        )

    @staticmethod
    def _chunk_hit_from_opensearch_doc(chunk: dict[str, Any]) -> KgChunkHit:
        return KgChunkHit(
            chunk_key=chunk.get("chunk_key") or f"{chunk.get('arxiv_id')}:{chunk.get('chunk_index')}",
            arxiv_id=chunk.get("arxiv_id", ""),
            chunk_index=int(chunk.get("chunk_index", 0)),
            chunk_text=chunk.get("chunk_text", ""),
            title=chunk.get("title", ""),
            authors=chunk.get("authors", ""),
            section_name=chunk.get("section_title") or chunk.get("section_name") or "",
        )
