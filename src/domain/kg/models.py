"""SQLAlchemy persistence for the table-graph KG (no AGE/Cypher).

Chunk linkage uses keys ``{document_id}:{chunk_index}`` stored in JSONB on nodes and edges.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from src.domain.db.interfaces.postgresql import Base


class KgNode(Base):
    """Entity node: merged descriptions and provenance across chunks and source documents."""

    __tablename__ = "kg_nodes"
    __table_args__ = (
        UniqueConstraint("entity_id", "ingestion_name", name="uq_kg_nodes_entity_ingestion"),
        Index("ix_kg_nodes_entity_ingestion", "entity_id", "ingestion_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[str] = mapped_column(String(512), nullable=False)
    entity_name: Mapped[str] = mapped_column(String(512), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(128), nullable=False, default="UNKNOWN")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_chunk_keys: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    document_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    ingestion_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class KgEdge(Base):
    """Directed relation between two ``entity_id`` values with optional thematic keywords."""

    __tablename__ = "kg_edges"
    __table_args__ = (
        UniqueConstraint(
            "source_entity_id",
            "target_entity_id",
            "ingestion_name",
            name="uq_kg_edges_src_tgt_ingestion",
        ),
        Index("ix_kg_edges_source", "source_entity_id"),
        Index("ix_kg_edges_target", "target_entity_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_entity_id: Mapped[str] = mapped_column(String(512), nullable=False)
    target_entity_id: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    keywords: Mapped[str] = mapped_column(Text, nullable=False, default="")
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    source_chunk_keys: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    ingestion_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class KgChunkExtraction(Base):
    """Per-chunk indexing status so CLI runs are idempotent (skip ``done`` unless ``--replace``)."""

    __tablename__ = "kg_chunk_extractions"

    chunk_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    ingestion_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
