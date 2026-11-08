"""One-shot PostgreSQL column renames for KG tables (arxiv → document naming)."""

import logging

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def apply_kg_schema_migrations(engine: Engine) -> None:
    """Rename legacy arXiv columns if present (idempotent)."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    if "kg_nodes" in tables:
        node_cols = {c["name"] for c in inspector.get_columns("kg_nodes")}
        if "arxiv_ids" in node_cols and "document_ids" not in node_cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE kg_nodes RENAME COLUMN arxiv_ids TO document_ids"))
            logger.info("Migrated kg_nodes.arxiv_ids → document_ids")

    if "kg_chunk_extractions" in tables:
        chunk_cols = {c["name"] for c in inspector.get_columns("kg_chunk_extractions")}
        if "arxiv_id" in chunk_cols and "document_id" not in chunk_cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE kg_chunk_extractions RENAME COLUMN arxiv_id TO document_id"))
            logger.info("Migrated kg_chunk_extractions.arxiv_id → document_id")
