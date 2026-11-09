"""Post-retrieval parent-section assembly from child chunks sharing a section_breadcrumb."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .chunk_index_schema import flatten_hit_metadata
from .indexing import MAX_CHUNKS_PER_DOCUMENT_QUERY, SearchIndexing

logger = logging.getLogger(__name__)

ChunkHit = dict[str, Any]


@dataclass(frozen=True, slots=True)
class SectionKey:
    """Identifies all child chunks belonging to one indexed section."""

    document_id: str
    section_breadcrumb: str

    @classmethod
    def from_hit(cls, hit: ChunkHit) -> SectionKey | None:
        breadcrumb = ParentSectionExpander.breadcrumb_for_hit(hit)
        if breadcrumb is None:
            return None
        document_id = str(hit.get("document_id", "")).strip()
        if not document_id:
            return None
        return cls(document_id=document_id, section_breadcrumb=breadcrumb)


class ParentSectionAssembler:
    """Merges ordered sibling chunk bodies into one parent-section text."""

    @staticmethod
    def assemble(siblings: list[ChunkHit]) -> str:
        ordered = sorted(siblings, key=lambda chunk: chunk.get("chunk_index", 0))
        texts = [str(chunk.get("chunk_text", "")) for chunk in ordered]
        if not texts:
            return ""
        if len(texts) == 1:
            return texts[0]

        common = ParentSectionAssembler._longest_common_prefix(texts)
        parts = [texts[0]]
        for text in texts[1:]:
            suffix = text[len(common) :] if text.startswith(common) else text
            parts.append(suffix)
        return "".join(parts)

    @staticmethod
    def _longest_common_prefix(strings: list[str]) -> str:
        if not strings:
            return ""
        prefix = strings[0]
        for text in strings[1:]:
            limit = min(len(prefix), len(text))
            idx = 0
            while idx < limit and prefix[idx] == text[idx]:
                idx += 1
            prefix = prefix[:idx]
            if not prefix:
                break
        return prefix


class ParentSectionExpander:
    """Expands retrieval hits to parent-section text via core section_breadcrumb siblings."""

    def __init__(self, indexing: SearchIndexing) -> None:
        self._indexing = indexing

    def get_chunks_by_section(self, document_id: str, section_breadcrumb: str) -> list[ChunkHit]:
        """Return all chunks for a section, sorted by chunk_index."""
        if not section_breadcrumb.strip():
            return []

        must_clauses: list[dict[str, Any]] = [
            {"term": {"document_id": document_id}},
            {"term": {"section_breadcrumb": section_breadcrumb}},
        ]
        if self._indexing.ingestion_name:
            must_clauses.append({"term": {"ingestion_name": self._indexing.ingestion_name}})

        search_body = {
            "query": {"bool": {"must": must_clauses}},
            "size": MAX_CHUNKS_PER_DOCUMENT_QUERY,
            "sort": [{"chunk_index": "asc"}],
            "_source": {"excludes": ["embedding"]},
        }

        try:
            response = self._indexing.client.search(index=self._indexing.index_name, body=search_body)
        except Exception as e:
            logger.error("Error getting section chunks: %s", e)
            return []

        chunks: list[ChunkHit] = []
        for hit in response["hits"]["hits"]:
            chunk = flatten_hit_metadata(dict(hit["_source"]))
            hit_document_id = chunk.get("document_id", "")
            hit_chunk_index = chunk.get("chunk_index", 0)
            chunk["chunk_id"] = hit["_id"]
            chunk["chunk_key"] = f"{hit_document_id}:{hit_chunk_index}"
            chunks.append(chunk)
        return chunks

    def expand(self, hits: list[ChunkHit]) -> list[ChunkHit]:
        if not hits:
            return []

        groups = self._group_hit_indices_by_section(hits)
        assembled_by_key = self._load_assembled_parents(groups)
        best_index_by_key = self._best_hit_index_per_section(hits, groups, assembled_by_key)
        return self._build_result_list(hits, assembled_by_key, best_index_by_key)

    @staticmethod
    def breadcrumb_for_hit(hit: ChunkHit) -> str | None:
        """Return the core index section_breadcrumb when present and non-empty."""
        breadcrumb = hit.get("section_breadcrumb")
        if breadcrumb is None:
            return None
        text = str(breadcrumb).strip()
        return text or None

    def _group_hit_indices_by_section(self, hits: list[ChunkHit]) -> dict[SectionKey, list[int]]:
        groups: dict[SectionKey, list[int]] = {}
        for idx, hit in enumerate(hits):
            key = SectionKey.from_hit(hit)
            if key is None:
                continue
            groups.setdefault(key, []).append(idx)
        return groups

    def _load_assembled_parents(self, groups: dict[SectionKey, list[int]]) -> dict[SectionKey, str]:
        assembled: dict[SectionKey, str] = {}
        for key in groups:
            siblings = self.get_chunks_by_section(key.document_id, key.section_breadcrumb)
            if len(siblings) <= 1:
                continue
            assembled[key] = ParentSectionAssembler.assemble(siblings)
        return assembled

    @staticmethod
    def _best_hit_index_per_section(
        hits: list[ChunkHit],
        groups: dict[SectionKey, list[int]],
        assembled_by_key: dict[SectionKey, str],
    ) -> dict[SectionKey, int]:
        winners: dict[SectionKey, int] = {}
        for key, indices in groups.items():
            if key not in assembled_by_key:
                continue
            winners[key] = max(indices, key=lambda i: (ParentSectionExpander._score_for_hit(hits[i]), -i))
        return winners

    @staticmethod
    def _build_result_list(
        hits: list[ChunkHit],
        assembled_by_key: dict[SectionKey, str],
        best_index_by_key: dict[SectionKey, int],
    ) -> list[ChunkHit]:
        result: list[ChunkHit] = []
        emitted: set[SectionKey] = set()

        for idx, hit in enumerate(hits):
            key = SectionKey.from_hit(hit)
            if key is None or key not in assembled_by_key:
                copy = dict(hit)
                copy.setdefault("parent_expansion_applied", False)
                result.append(copy)
                continue

            if best_index_by_key.get(key) != idx or key in emitted:
                continue

            emitted.add(key)
            result.append(ParentSectionExpander._apply_expansion(hit, assembled_by_key[key]))

        return result

    @staticmethod
    def _score_for_hit(hit: ChunkHit) -> float:
        score = hit.get("score")
        if score is None:
            return 0.0
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _apply_expansion(hit: ChunkHit, assembled_text: str) -> ChunkHit:
        return {
            **hit,
            "matched_child_chunk_index": hit.get("chunk_index"),
            "matched_child_chunk_text": hit.get("chunk_text"),
            "chunk_text": assembled_text,
            "parent_expansion_applied": True,
        }


def section_breadcrumb_for_hit(hit: ChunkHit) -> str | None:
    return ParentSectionExpander.breadcrumb_for_hit(hit)


def assemble_parent_text(siblings: list[ChunkHit]) -> str:
    return ParentSectionAssembler.assemble(siblings)


