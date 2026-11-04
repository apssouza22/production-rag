import logging
from dataclasses import dataclass, field
from typing import Any

from src.domain.db.interfaces.base import BaseDatabase
from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.kg.extraction import KgExtractionService
from src.domain.kg.kg_vector_client import KgVectorClient
from src.domain.kg.merge import make_chunk_key, normalize_entity_id, relation_vector_id
from src.domain.kg.repository import KgGraphRepository
from src.domain.kg.schemas import ChunkExtractionResult, ExtractedRelation
from src.domain.opensearch.paper_rrf_search import PaperRrfSearch

logger = logging.getLogger(__name__)


@dataclass
class KgPaperIndexingStats:
    chunks_processed: int = 0
    chunks_skipped: int = 0
    chunks_failed: int = 0
    entities_upserted: int = 0
    relations_upserted: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "chunks_processed": self.chunks_processed,
            "chunks_skipped": self.chunks_skipped,
            "chunks_failed": self.chunks_failed,
            "entities_upserted": self.entities_upserted,
            "relations_upserted": self.relations_upserted,
        }


@dataclass
class KgBatchIndexingStats:
    papers: int = 0
    chunks_processed: int = 0
    chunks_skipped: int = 0
    chunks_failed: int = 0
    entities_upserted: int = 0
    relations_upserted: int = 0

    def absorb_paper(self, paper_stats: KgPaperIndexingStats) -> None:
        self.chunks_processed += paper_stats.chunks_processed
        self.chunks_skipped += paper_stats.chunks_skipped
        self.chunks_failed += paper_stats.chunks_failed
        self.entities_upserted += paper_stats.entities_upserted
        self.relations_upserted += paper_stats.relations_upserted

    def as_dict(self) -> dict[str, Any]:
        return {
            "papers": self.papers,
            "chunks_processed": self.chunks_processed,
            "chunks_skipped": self.chunks_skipped,
            "chunks_failed": self.chunks_failed,
            "entities_upserted": self.entities_upserted,
            "relations_upserted": self.relations_upserted,
        }


@dataclass
class _PendingVectorIndex:
    """Unique entity/relation documents collected during one paper before embedding."""

    ingestion_name: str
    entity_docs: list[dict[str, Any]] = field(default_factory=list)
    relation_docs: list[dict[str, Any]] = field(default_factory=list)
    _entity_ids: set[str] = field(default_factory=set, repr=False)
    _relation_ids: set[str] = field(default_factory=set, repr=False)

    def track_entity(
        self,
        *,
        entity_id: str,
        entity_name: str,
        entity_type: str,
        description: str,
    ) -> None:
        if entity_id in self._entity_ids:
            return
        self._entity_ids.add(entity_id)
        self.entity_docs.append(
            {
                "entity_id": entity_id,
                "entity_name": entity_name,
                "entity_type": entity_type,
                "description": description,
                "ingestion_name": self.ingestion_name,
            }
        )

    def track_relation(
        self,
        *,
        source_entity_id: str,
        target_entity_id: str,
        description: str,
        keywords: str,
        weight: float,
    ) -> None:
        rel_id = relation_vector_id(source_entity_id, target_entity_id)
        if rel_id in self._relation_ids:
            return
        self._relation_ids.add(rel_id)
        self.relation_docs.append(
            {
                "relation_id": rel_id,
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
                "description": description,
                "keywords": keywords,
                "weight": weight,
                "ingestion_name": self.ingestion_name,
            }
        )


@dataclass
class _PaperIndexingContext:
    arxiv_id: str
    repo: KgGraphRepository
    stats: KgPaperIndexingStats
    vectors: _PendingVectorIndex
    name_to_id: dict[str, str] = field(default_factory=dict)


class KgIndexingService:
    """Offline pipeline: OpenSearch chunks → LLM extract → Postgres graph → entity/relation vectors.

    Expects hybrid chunks to already exist for each ``arxiv_id``. Writes graph state
    in PostgreSQL and embedding rows in separate OpenSearch indices (not the chunk index).
    """

    def __init__(
        self,
        database: BaseDatabase,
        paper_rrf_search: PaperRrfSearch,
        kg_vector_client: KgVectorClient,
        extraction_service: KgExtractionService,
        embeddings_client: JinaEmbeddingsClient,
        ingestion_name: str,
    ):
        self.database = database
        self.paper_rrf_search = paper_rrf_search
        self.kg_vector_client = kg_vector_client
        self.extraction_service = extraction_service
        self.embeddings_client = embeddings_client
        self.ingestion_name = ingestion_name

    async def index_arxiv_ids(self, arxiv_ids: list[str], *, replace: bool = False) -> dict[str, Any]:
        stats = KgBatchIndexingStats()
        self.kg_vector_client.setup_indices()

        for raw_id in arxiv_ids:
            arxiv_id = raw_id.strip()
            if not arxiv_id:
                continue
            stats.papers += 1
            paper_stats = await self._index_paper(arxiv_id, replace=replace)
            stats.absorb_paper(paper_stats)

        return stats.as_dict()

    async def _index_paper(self, arxiv_id: str, *, replace: bool) -> KgPaperIndexingStats:
        stats = KgPaperIndexingStats()
        chunks = self.paper_rrf_search.get_chunks_by_paper(arxiv_id, ingestion_name=self.ingestion_name)
        if not chunks:
            logger.warning("No chunks in OpenSearch for %s", arxiv_id)
            return stats

        vectors = _PendingVectorIndex(ingestion_name=self.ingestion_name)

        with self.database.get_session() as session:
            repo = KgGraphRepository(session, self.ingestion_name)
            if replace:
                repo.delete_graph_for_arxiv(arxiv_id)
                repo.commit()

            ctx = _PaperIndexingContext(
                arxiv_id=arxiv_id,
                repo=repo,
                stats=stats,
                vectors=vectors,
            )

            for chunk in chunks:
                await self._process_chunk(ctx, chunk, replace=replace)

            repo.commit()

        await self._embed_and_index_vectors(vectors.entity_docs, vectors.relation_docs)
        return stats

    async def _process_chunk(
        self,
        ctx: _PaperIndexingContext,
        chunk: dict[str, Any],
        *,
        replace: bool,
    ) -> None:
        chunk_index = int(chunk.get("chunk_index", 0))
        chunk_key = make_chunk_key(ctx.arxiv_id, chunk_index)

        if self._should_skip_chunk(ctx.repo, chunk_key, replace=replace):
            ctx.stats.chunks_skipped += 1
            return

        chunk_text = chunk.get("chunk_text") or ""
        if not chunk_text.strip():
            self._mark_chunk_done(ctx, chunk_key, chunk_index)
            ctx.stats.chunks_skipped += 1
            return

        try:
            extraction = await self.extraction_service.extract_from_chunk(
                chunk_text,
                section_title=_chunk_section_title(chunk),
            )
            self._persist_entities(ctx, chunk_key, extraction)
            self._persist_relations(ctx, chunk_key, extraction.relations)
            self._mark_chunk_done(ctx, chunk_key, chunk_index)
            ctx.stats.chunks_processed += 1
        except Exception as exc:
            logger.exception("KG extraction failed for %s: %s", chunk_key, exc)
            ctx.repo.mark_chunk_extraction(
                chunk_key=chunk_key,
                arxiv_id=ctx.arxiv_id,
                chunk_index=chunk_index,
                status="failed",
                error=str(exc),
            )
            ctx.stats.chunks_failed += 1

    def _should_skip_chunk(self, repo: KgGraphRepository, chunk_key: str, *, replace: bool) -> bool:
        existing = repo.get_chunk_extraction(chunk_key)
        return bool(existing and existing.status == "done" and not replace)

    def _mark_chunk_done(self, ctx: _PaperIndexingContext, chunk_key: str, chunk_index: int) -> None:
        ctx.repo.mark_chunk_extraction(
            chunk_key=chunk_key,
            arxiv_id=ctx.arxiv_id,
            chunk_index=chunk_index,
            status="done",
        )

    def _persist_entities(
        self,
        ctx: _PaperIndexingContext,
        chunk_key: str,
        extraction: ChunkExtractionResult,
    ) -> None:
        for entity in extraction.entities:
            entity_id = ctx.repo.upsert_node(
                entity_name=entity.entity_name,
                entity_type=entity.entity_type,
                description=entity.description,
                chunk_key=chunk_key,
                arxiv_id=ctx.arxiv_id,
            )
            ctx.name_to_id[entity.entity_name] = entity_id
            ctx.name_to_id[normalize_entity_id(entity.entity_name)] = entity_id
            ctx.stats.entities_upserted += 1
            ctx.vectors.track_entity(
                entity_id=entity_id,
                entity_name=entity.entity_name,
                entity_type=entity.entity_type,
                description=entity.description,
            )

    def _persist_relations(
        self,
        ctx: _PaperIndexingContext,
        chunk_key: str,
        relations: list[ExtractedRelation],
    ) -> None:
        for relation in relations:
            source_id = self._resolve_entity_id(ctx.name_to_id, relation.source_entity)
            target_id = self._resolve_entity_id(ctx.name_to_id, relation.target_entity)
            if source_id == target_id:
                continue

            self._ensure_relation_endpoint_nodes(ctx, chunk_key, relation)
            ctx.repo.upsert_edge(
                source_entity_id=source_id,
                target_entity_id=target_id,
                description=relation.description,
                keywords=relation.keywords,
                weight=relation.weight,
                chunk_key=chunk_key,
            )
            ctx.stats.relations_upserted += 1
            ctx.vectors.track_relation(
                source_entity_id=source_id,
                target_entity_id=target_id,
                description=relation.description,
                keywords=relation.keywords,
                weight=relation.weight,
            )

    def _ensure_relation_endpoint_nodes(
        self,
        ctx: _PaperIndexingContext,
        chunk_key: str,
        relation: ExtractedRelation,
    ) -> None:
        for entity_name in (relation.source_entity, relation.target_entity):
            ctx.repo.upsert_node(
                entity_name=entity_name,
                entity_type="UNKNOWN",
                description="",
                chunk_key=chunk_key,
                arxiv_id=ctx.arxiv_id,
            )

    @staticmethod
    def _resolve_entity_id(name_to_id: dict[str, str], entity_name: str) -> str:
        return name_to_id.get(entity_name) or normalize_entity_id(entity_name)

    async def _embed_and_index_vectors(
        self,
        entity_docs: list[dict[str, Any]],
        relation_docs: list[dict[str, Any]],
    ) -> None:
        if entity_docs:
            texts = [f"{doc['entity_name']}\n{doc['description']}" for doc in entity_docs]
            embeddings = await self.embeddings_client.embed_passages(texts, batch_size=32)
            for doc, embedding in zip(entity_docs, embeddings):
                doc["embedding"] = embedding
            self.kg_vector_client.bulk_upsert_entities(entity_docs)

        if relation_docs:
            texts = [
                f"{doc['source_entity_id']} -> {doc['target_entity_id']}\n{doc['description']}\n{doc['keywords']}"
                for doc in relation_docs
            ]
            embeddings = await self.embeddings_client.embed_passages(texts, batch_size=32)
            for doc, embedding in zip(relation_docs, embeddings):
                doc["embedding"] = embedding
            self.kg_vector_client.bulk_upsert_relations(relation_docs)


def _chunk_section_title(chunk: dict[str, Any]) -> str:
    return chunk.get("section_title") or chunk.get("section_name") or ""


__all__ = ["KgIndexingService", "KgBatchIndexingStats", "KgPaperIndexingStats"]
