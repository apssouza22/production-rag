from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

from src.domain.kg.merge import merge_descriptions, merge_unique_strings, normalize_entity_id
from src.domain.kg.models import KgChunkExtraction, KgEdge, KgNode


class KgGraphRepository:
    """Postgres CRUD for the table graph scoped by ``ingestion_name``.

    Upserts merge duplicate entities/relations across chunks without an extra LLM
    summarize step (descriptions are appended and de-duplicated in Python).
    """

    def __init__(self, session: Session, ingestion_name: str):
        self.session = session
        self.ingestion_name = ingestion_name

    def upsert_node(
        self,
        *,
        entity_name: str,
        entity_type: str,
        description: str,
        chunk_key: str,
        document_id: str,
    ) -> str:
        entity_id = normalize_entity_id(entity_name)
        stmt = select(KgNode).where(
            KgNode.entity_id == entity_id,
            KgNode.ingestion_name == self.ingestion_name,
        )
        node = self.session.scalar(stmt)
        if node is None:
            node = KgNode(
                entity_id=entity_id,
                entity_name=entity_name.strip(),
                entity_type=entity_type or "UNKNOWN",
                description=description or "",
                source_chunk_keys=[chunk_key],
                document_ids=[document_id],
                ingestion_name=self.ingestion_name,
            )
            self.session.add(node)
        else:
            node.entity_name = entity_name.strip() or node.entity_name
            node.entity_type = entity_type or node.entity_type
            node.description = merge_descriptions(node.description, description)
            node.source_chunk_keys = merge_unique_strings(list(node.source_chunk_keys or []), [chunk_key])
            node.document_ids = merge_unique_strings(list(node.document_ids or []), [document_id])
            node.updated_at = datetime.now(timezone.utc)
        return entity_id

    def upsert_edge(
        self,
        *,
        source_entity_id: str,
        target_entity_id: str,
        description: str,
        keywords: str,
        weight: float,
        chunk_key: str,
    ) -> None:
        stmt = select(KgEdge).where(
            KgEdge.source_entity_id == source_entity_id,
            KgEdge.target_entity_id == target_entity_id,
            KgEdge.ingestion_name == self.ingestion_name,
        )
        edge = self.session.scalar(stmt)
        if edge is None:
            edge = KgEdge(
                source_entity_id=source_entity_id,
                target_entity_id=target_entity_id,
                description=description or "",
                keywords=keywords or "",
                weight=weight,
                source_chunk_keys=[chunk_key],
                ingestion_name=self.ingestion_name,
            )
            self.session.add(edge)
        else:
            edge.description = merge_descriptions(edge.description, description)
            if keywords and keywords not in edge.keywords:
                edge.keywords = f"{edge.keywords} {keywords}".strip() if edge.keywords else keywords
            edge.weight = max(edge.weight, weight)
            edge.source_chunk_keys = merge_unique_strings(list(edge.source_chunk_keys or []), [chunk_key])
            edge.updated_at = datetime.now(timezone.utc)

    def get_nodes_by_entity_ids(self, entity_ids: Iterable[str]) -> list[KgNode]:
        ids = list(entity_ids)
        if not ids:
            return []
        stmt = select(KgNode).where(
            KgNode.entity_id.in_(ids),
            KgNode.ingestion_name == self.ingestion_name,
        )
        return list(self.session.scalars(stmt))

    def get_incident_edges_for_entity_ids(
        self,
        entity_ids: Iterable[str],
        *,
        max_edges: int,
    ) -> list[KgEdge]:
        """Edges with at least one endpoint in ``entity_ids``, capped and ordered by weight."""
        ids = list(entity_ids)
        if not ids or max_edges <= 0:
            return []
        stmt = (
            select(KgEdge)
            .where(
                KgEdge.ingestion_name == self.ingestion_name,
                or_(
                    KgEdge.source_entity_id.in_(ids),
                    KgEdge.target_entity_id.in_(ids),
                ),
            )
            .order_by(KgEdge.weight.desc())
            .limit(max_edges)
        )
        return list(self.session.scalars(stmt))

    def get_edges_by_pairs(self, pairs: list[tuple[str, str]]) -> list[KgEdge]:
        if not pairs:
            return []
        results: list[KgEdge] = []
        for source_id, target_id in pairs:
            stmt = select(KgEdge).where(
                KgEdge.source_entity_id == source_id,
                KgEdge.target_entity_id == target_id,
                KgEdge.ingestion_name == self.ingestion_name,
            )
            edge = self.session.scalar(stmt)
            if edge:
                results.append(edge)
        return results

    def node_counts(self, entity_ids: Iterable[str]) -> dict[str, int]:
        ids = list(entity_ids)
        if not ids:
            return {}
        out: dict[str, int] = {}
        stmt = (
            select(KgEdge.source_entity_id, func.count())
            .where(
                KgEdge.ingestion_name == self.ingestion_name,
                KgEdge.source_entity_id.in_(ids),
            )
            .group_by(KgEdge.source_entity_id)
        )
        for source_id, count in self.session.execute(stmt):
            out[source_id] = int(count)
        stmt2 = (
            select(KgEdge.target_entity_id, func.count())
            .where(
                KgEdge.ingestion_name == self.ingestion_name,
                KgEdge.target_entity_id.in_(ids),
            )
            .group_by(KgEdge.target_entity_id)
        )
        for target_id, count in self.session.execute(stmt2):
            out[target_id] = out.get(target_id, 0) + int(count)
        return out

    def mark_chunk_extraction(
        self,
        *,
        chunk_key: str,
        document_id: str,
        chunk_index: int,
        status: str,
        error: str | None = None,
    ) -> None:
        row = self.session.get(KgChunkExtraction, chunk_key)
        if row is None:
            row = KgChunkExtraction(
                chunk_key=chunk_key,
                document_id=document_id,
                chunk_index=chunk_index,
                ingestion_name=self.ingestion_name,
            )
            self.session.add(row)
        row.status = status
        row.error = error
        row.extracted_at = datetime.now(timezone.utc) if status == "done" else row.extracted_at

    def get_chunk_extraction(self, chunk_key: str) -> KgChunkExtraction | None:
        return self.session.get(KgChunkExtraction, chunk_key)

    def delete_graph_for_document(self, document_id: str) -> None:
        chunk_keys_stmt = select(KgChunkExtraction.chunk_key).where(
            KgChunkExtraction.document_id == document_id,
            KgChunkExtraction.ingestion_name == self.ingestion_name,
        )
        chunk_keys = set(self.session.scalars(chunk_keys_stmt))
        if not chunk_keys:
            return

        nodes = self.session.scalars(select(KgNode).where(KgNode.ingestion_name == self.ingestion_name))
        for node in nodes:
            if document_id not in (node.document_ids or []):
                continue
            keys = [k for k in (node.source_chunk_keys or []) if k not in chunk_keys]
            doc_ids = [d for d in (node.document_ids or []) if d != document_id]
            if not keys and not doc_ids:
                self.session.delete(node)
            else:
                node.source_chunk_keys = keys
                node.document_ids = doc_ids

        edges = self.session.scalars(select(KgEdge).where(KgEdge.ingestion_name == self.ingestion_name))
        for edge in edges:
            keys = [k for k in (edge.source_chunk_keys or []) if k not in chunk_keys]
            if not keys:
                self.session.delete(edge)
            else:
                edge.source_chunk_keys = keys

        self.session.execute(
            delete(KgChunkExtraction).where(
                KgChunkExtraction.document_id == document_id,
                KgChunkExtraction.ingestion_name == self.ingestion_name,
            )
        )

    def commit(self) -> None:
        self.session.commit()

    def get_subgraph_for_visualization(
        self,
        *,
        document_id: str | None = None,
        max_nodes: int = 300,
        max_edges: int = 1000,
    ) -> tuple[list[KgNode], list[KgEdge]]:
        """Return a bounded set of nodes and only edges whose endpoints are both in that set.

        Ordering prefers recently updated nodes so active extractions surface first.

        :param document_id: When set, keep only nodes whose ``document_ids`` JSONB array contains this id.
        :param max_nodes: Maximum number of nodes (caller should cap; typically 300–500).
        :param max_edges: Maximum number of edges after endpoint filtering.
        """
        conditions = [KgNode.ingestion_name == self.ingestion_name]
        if document_id is not None and document_id.strip():
            conditions.append(KgNode.document_ids.contains([document_id.strip()]))

        stmt = select(KgNode).where(*conditions).order_by(KgNode.updated_at.desc()).limit(max_nodes)

        nodes = list(self.session.scalars(stmt))
        if not nodes:
            return [], []

        entity_ids = [n.entity_id for n in nodes]
        edge_stmt = (
            select(KgEdge)
            .where(
                KgEdge.ingestion_name == self.ingestion_name,
                KgEdge.source_entity_id.in_(entity_ids),
                KgEdge.target_entity_id.in_(entity_ids),
            )
            .order_by(KgEdge.updated_at.desc())
            .limit(max_edges)
        )
        edges = list(self.session.scalars(edge_stmt))
        return nodes, edges
