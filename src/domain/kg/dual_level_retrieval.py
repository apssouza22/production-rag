import logging

from src.config import KGSettings
from src.domain.db.interfaces.base import BaseDatabase
from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.kg.extraction import KgExtractionService
from src.domain.kg.graph.repository import KgGraphRepository
from src.domain.kg.indexing_service import relation_vector_id
from src.domain.kg.opensearch.kg_vector_client import KgVectorClient
from src.domain.kg.schemas import (
    DualLevelKGResult,
    KgChunkHit,
    KgEntityHit,
    KgRelationHit,
    format_kg_context,
)
from src.domain.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


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
    ):
        self.database = database
        self.opensearch_client = opensearch_client
        self.kg_vector_client = kg_vector_client
        self.extraction_service = extraction_service
        self.embeddings_client = embeddings_client
        self.kg_settings = kg_settings
        self.ingestion_name = ingestion_name

    async def retrieve(
        self,
        query: str,
        *,
        top_k_entities: int | None = None,
        top_k_relations: int | None = None,
    ) -> DualLevelKGResult:
        top_k_entities = top_k_entities or self.kg_settings.top_k_entities
        top_k_relations = top_k_relations or self.kg_settings.top_k_relations

        keywords = await self.extraction_service.extract_keywords(query)
        hl = keywords.high_level_keywords
        ll = keywords.low_level_keywords

        if not hl and not ll:
            if len(query.strip()) < 50:
                ll = [query.strip()]
            else:
                return DualLevelKGResult(query=query, high_level_keywords=hl, low_level_keywords=ll)

        entity_hits: list[KgEntityHit] = []
        relation_hits: list[KgRelationHit] = []
        chunk_key_order: dict[str, int] = {}
        order_counter = 0

        if ll:
            ll_text = ", ".join(ll)
            ll_embedding = await self.embeddings_client.embed_query(ll_text)
            vector_hits = self.kg_vector_client.knn_search_entities(ll_embedding, top_k_entities)
            entity_ids = [hit["entity_id"] for hit in vector_hits if hit.get("entity_id")]

            with self.database.get_session() as session:
                repo = KgGraphRepository(session, self.ingestion_name)
                nodes = repo.get_nodes_by_entity_ids(entity_ids)
                node_map = {n.entity_id: n for n in nodes}
                degrees = repo.node_degrees(entity_ids)

            for rank, hit in enumerate(vector_hits, start=1):
                entity_id = hit.get("entity_id")
                if not entity_id or entity_id not in node_map:
                    continue
                node = node_map[entity_id]
                entity_hits.append(
                    KgEntityHit(
                        entity_id=entity_id,
                        entity_name=node.entity_name,
                        entity_type=node.entity_type,
                        description=node.description,
                        rank=degrees.get(entity_id, 0),
                        source_chunk_keys=list(node.source_chunk_keys or []),
                    )
                )
                for key in node.source_chunk_keys or []:
                    if key not in chunk_key_order:
                        chunk_key_order[key] = order_counter
                        order_counter += 1

        if hl:
            hl_text = ", ".join(hl)
            hl_embedding = await self.embeddings_client.embed_query(hl_text)
            vector_hits = self.kg_vector_client.knn_search_relations(hl_embedding, top_k_relations)
            pairs = [
                (hit["source_entity_id"], hit["target_entity_id"])
                for hit in vector_hits
                if hit.get("source_entity_id") and hit.get("target_entity_id")
            ]

            with self.database.get_session() as session:
                repo = KgGraphRepository(session, self.ingestion_name)
                edges = repo.get_edges_by_pairs(pairs)
                edge_map = {(e.source_entity_id, e.target_entity_id): e for e in edges}

            for hit in vector_hits:
                pair = (hit.get("source_entity_id"), hit.get("target_entity_id"))
                if not pair[0] or not pair[1] or pair not in edge_map:
                    continue
                edge = edge_map[pair]
                relation_hits.append(
                    KgRelationHit(
                        source_entity_id=edge.source_entity_id,
                        target_entity_id=edge.target_entity_id,
                        description=edge.description,
                        keywords=edge.keywords,
                        weight=edge.weight,
                        rank=int(edge.weight),
                        source_chunk_keys=list(edge.source_chunk_keys or []),
                    )
                )
                for key in edge.source_chunk_keys or []:
                    if key not in chunk_key_order:
                        chunk_key_order[key] = order_counter + 1000
                        order_counter += 1

        entity_hits.sort(key=lambda e: e.rank, reverse=True)
        relation_hits.sort(key=lambda r: r.rank, reverse=True)

        chunk_keys = sorted(chunk_key_order.keys(), key=lambda k: chunk_key_order[k])
        os_chunks = self.opensearch_client.get_chunks_by_chunk_keys(
            chunk_keys,
            ingestion_name=self.ingestion_name,
        )
        chunk_hits = [
            KgChunkHit(
                chunk_key=chunk.get("chunk_key") or f"{chunk.get('arxiv_id')}:{chunk.get('chunk_index')}",
                arxiv_id=chunk.get("arxiv_id", ""),
                chunk_index=int(chunk.get("chunk_index", 0)),
                chunk_text=chunk.get("chunk_text", ""),
                title=chunk.get("title", ""),
                authors=chunk.get("authors", ""),
                section_name=chunk.get("section_title") or chunk.get("section_name") or "",
            )
            for chunk in os_chunks
        ]

        formatted = format_kg_context(
            entity_hits,
            relation_hits,
            chunk_hits,
            max_entity_tokens=self.kg_settings.max_entity_tokens,
            max_relation_tokens=self.kg_settings.max_relation_tokens,
            max_total_tokens=self.kg_settings.max_total_tokens,
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
