import logging
from typing import Any

from src.config import Settings
from src.domain.db.interfaces.base import BaseDatabase
from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.kg.extraction import KgExtractionService
from src.domain.kg.graph import KgGraphRepository
from src.domain.kg.graph.repository import KgGraphRepository
from src.domain.kg.merge import make_chunk_key, normalize_entity_id
from src.domain.kg.opensearch.kg_vector_client import KgVectorClient
from src.domain.kg.schemas import ChunkExtractionResult
from src.domain.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


def relation_vector_id(source_entity_id: str, target_entity_id: str) -> str:
    return f"{source_entity_id}->{target_entity_id}"


class KgIndexingService:
    """Offline pipeline: OpenSearch chunks → LLM extract → Postgres graph → entity/relation vectors.

    Expects hybrid chunks to already exist for each ``arxiv_id``. Writes graph state
    in PostgreSQL and embedding rows in separate OpenSearch indices (not the chunk index).
    """

    def __init__(
        self,
        database: BaseDatabase,
        opensearch_client: OpenSearchClient,
        kg_vector_client: KgVectorClient,
        extraction_service: KgExtractionService,
        embeddings_client: JinaEmbeddingsClient,
        ingestion_name: str,
    ):
        self.database = database
        self.opensearch_client = opensearch_client
        self.kg_vector_client = kg_vector_client
        self.extraction_service = extraction_service
        self.embeddings_client = embeddings_client
        self.ingestion_name = ingestion_name

    async def index_arxiv_ids(self, arxiv_ids: list[str], *, replace: bool = False) -> dict[str, Any]:
        stats = {
            "papers": 0,
            "chunks_processed": 0,
            "chunks_skipped": 0,
            "chunks_failed": 0,
            "entities_upserted": 0,
            "relations_upserted": 0,
        }
        self.kg_vector_client.setup_indices()

        for arxiv_id in arxiv_ids:
            arxiv_id = arxiv_id.strip()
            if not arxiv_id:
                continue
            stats["papers"] += 1
            paper_stats = await self._index_paper(arxiv_id, replace=replace)
            for key in ("chunks_processed", "chunks_skipped", "chunks_failed", "entities_upserted", "relations_upserted"):
                stats[key] += paper_stats.get(key, 0)

        return stats

    async def _index_paper(self, arxiv_id: str, *, replace: bool) -> dict[str, int]:
        stats = {
            "chunks_processed": 0,
            "chunks_skipped": 0,
            "chunks_failed": 0,
            "entities_upserted": 0,
            "relations_upserted": 0,
        }
        chunks = self.opensearch_client.get_chunks_by_paper(arxiv_id, ingestion_name=self.ingestion_name)
        if not chunks:
            logger.warning("No chunks in OpenSearch for %s", arxiv_id)
            return stats

        entity_docs: list[dict[str, Any]] = []
        relation_docs: list[dict[str, Any]] = []
        entity_ids_seen: set[str] = set()
        relation_ids_seen: set[str] = set()

        with self.database.get_session() as session:
            repo = KgGraphRepository(session, self.ingestion_name)
            if replace:
                repo.delete_graph_for_arxiv(arxiv_id)
                repo.commit()

            for chunk in chunks:
                chunk_index = int(chunk.get("chunk_index", 0))
                chunk_key = make_chunk_key(arxiv_id, chunk_index)
                existing = repo.get_chunk_extraction(chunk_key)
                if existing and existing.status == "done" and not replace:
                    stats["chunks_skipped"] += 1
                    continue

                chunk_text = chunk.get("chunk_text") or ""
                if not chunk_text.strip():
                    repo.mark_chunk_extraction(
                        chunk_key=chunk_key,
                        arxiv_id=arxiv_id,
                        chunk_index=chunk_index,
                        status="done",
                    )
                    stats["chunks_skipped"] += 1
                    continue

                try:
                    chunk_extraction = await self.extraction_service.extract_from_chunk(
                        chunk_text,
                        section_title=chunk.get("section_title") or chunk.get("section_name") or "",
                    )
                    name_to_id: dict[str, str] = {}
                    await self._handle_entitites(arxiv_id, chunk_extraction, chunk_key, entity_docs, entity_ids_seen, name_to_id, repo, stats)

                    for relation in chunk_extraction.relations:
                        source_id = name_to_id.get(relation.source_entity) or normalize_entity_id(relation.source_entity)
                        target_id = name_to_id.get(relation.target_entity) or normalize_entity_id(relation.target_entity)
                        if source_id == target_id:
                            continue
                        repo.upsert_node(
                            entity_name=relation.source_entity,
                            entity_type="UNKNOWN",
                            description="",
                            chunk_key=chunk_key,
                            arxiv_id=arxiv_id,
                        )
                        repo.upsert_node(
                            entity_name=relation.target_entity,
                            entity_type="UNKNOWN",
                            description="",
                            chunk_key=chunk_key,
                            arxiv_id=arxiv_id,
                        )
                        repo.upsert_edge(
                            source_entity_id=source_id,
                            target_entity_id=target_id,
                            description=relation.description,
                            keywords=relation.keywords,
                            weight=relation.weight,
                            chunk_key=chunk_key,
                        )
                        stats["relations_upserted"] += 1
                        rel_id = relation_vector_id(source_id, target_id)
                        if rel_id not in relation_ids_seen:
                            relation_ids_seen.add(rel_id)
                            relation_docs.append(
                                {
                                    "relation_id": rel_id,
                                    "source_entity_id": source_id,
                                    "target_entity_id": target_id,
                                    "description": relation.description,
                                    "keywords": relation.keywords,
                                    "weight": relation.weight,
                                    "ingestion_name": self.ingestion_name,
                                }
                            )

                    repo.mark_chunk_extraction(
                        chunk_key=chunk_key,
                        arxiv_id=arxiv_id,
                        chunk_index=chunk_index,
                        status="done",
                    )
                    stats["chunks_processed"] += 1
                except Exception as exc:
                    logger.exception("KG extraction failed for %s: %s", chunk_key, exc)
                    repo.mark_chunk_extraction(
                        chunk_key=chunk_key,
                        arxiv_id=arxiv_id,
                        chunk_index=chunk_index,
                        status="failed",
                        error=str(exc),
                    )
                    stats["chunks_failed"] += 1

            repo.commit()

        await self._embed_and_index_vectors(entity_docs, relation_docs)
        return stats

    async def _handle_entitites(self, arxiv_id: str, chunk_extraction: ChunkExtractionResult, chunk_key: str, entity_docs: list[dict[str, Any]], entity_ids_seen: set[str], name_to_id: dict[str, str],
        repo: KgGraphRepository, stats: dict[str, int]):
        for entity in chunk_extraction.entities:
            entity_id = repo.upsert_node(
                entity_name=entity.entity_name,
                entity_type=entity.entity_type,
                description=entity.description,
                chunk_key=chunk_key,
                arxiv_id=arxiv_id,
            )
            name_to_id[entity.entity_name] = entity_id
            name_to_id[normalize_entity_id(entity.entity_name)] = entity_id
            stats["entities_upserted"] += 1
            if entity_id not in entity_ids_seen:
                entity_ids_seen.add(entity_id)
                entity_docs.append(
                    {
                        "entity_id": entity_id,
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "description": entity.description,
                        "ingestion_name": self.ingestion_name,
                    }
                )

    async def _embed_and_index_vectors(
        self,
        entity_docs: list[dict[str, Any]],
        relation_docs: list[dict[str, Any]],
    ) -> None:
        if entity_docs:
            texts = [f"{d['entity_name']}\n{d['description']}" for d in entity_docs]
            embeddings = await self.embeddings_client.embed_passages(texts, batch_size=32)
            for doc, embedding in zip(entity_docs, embeddings):
                doc["embedding"] = embedding
            self.kg_vector_client.bulk_upsert_entities(entity_docs)

        if relation_docs:
            texts = [
                f"{d['source_entity_id']} -> {d['target_entity_id']}\n{d['description']}\n{d['keywords']}" for d in relation_docs
            ]
            embeddings = await self.embeddings_client.embed_passages(texts, batch_size=32)
            for doc, embedding in zip(relation_docs, embeddings):
                doc["embedding"] = embedding
            self.kg_vector_client.bulk_upsert_relations(relation_docs)
