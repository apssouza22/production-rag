"""Persist parsed PDF content as Markdown and JSON sidecars."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .artifacts import ParsedPaperArtifact, artifact_stem
from .models import PaperSection, ParsedPaper, PdfContent

logger = logging.getLogger(__name__)


@dataclass
class ArtifactWriteResult:
    """Outcome of persisting Markdown and JSON sidecars."""

    artifacts: dict[str, ParsedPaperArtifact] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text beside its destination and atomically replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = temporary_file.name

        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


class MarkdownWriter:
    """Write Markdown and structured sidecars for parsed papers."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    def write_all(
        self,
        parsed_papers: Mapping[str, ParsedPaper],
        *,
        errors: Iterable[str] = (),
    ) -> ArtifactWriteResult:
        """Persist artifacts for every successfully parsed paper."""
        result = ArtifactWriteResult(errors=list(errors))
        for arxiv_id, parsed_paper in parsed_papers.items():
            self._try_write_paper(arxiv_id, parsed_paper, result)
        return result

    def write_paper(
        self,
        arxiv_id: str,
        parsed_paper: ParsedPaper
    ) -> ParsedPaperArtifact:
        """Write Markdown and a structured JSON sidecar for one parsed paper."""
        pdf_content = parsed_paper.pdf_content
        if pdf_content is None:
            raise ValueError(f"Parsed paper {arxiv_id} has no PDF content")

        markdown_body = self.create_markdown_content(pdf_content)
        stem = artifact_stem(arxiv_id)
        markdown_path = self._output_dir / f"{stem}.md"
        sidecar_path = self._output_dir / f"{stem}.json"

        payload = parsed_paper.model_dump(mode="json")
        serialized_pdf_content = payload.get("pdf_content")
        if isinstance(serialized_pdf_content, dict):
            serialized_pdf_content.pop("markdown", None)
            metadata = serialized_pdf_content.get("metadata")
            if isinstance(metadata, dict):
                sidecar_metadata = dict(metadata)
                sidecar_metadata.pop("docling_markdown_fallback", None)
                serialized_pdf_content["metadata"] = sidecar_metadata

        _atomic_write_text(markdown_path, markdown_body)
        _atomic_write_text(sidecar_path, json.dumps(payload, indent=2, sort_keys=True))

        return {
            "markdown_path": str(markdown_path.resolve()),
            "sidecar_path": str(sidecar_path.resolve()),
        }

    @staticmethod
    def create_markdown_from_sections(sections: list[PaperSection]) -> str:
        """Build a Markdown document from extracted sections, preserving inline figures and tables."""
        parts: list[str] = []
        for section in sections:
            if section.title == "Content" and section.breadcrumb is None:
                if section.content.strip():
                    parts.append(section.content.strip())
                continue

            heading_level = min(max(section.level, 1), 6)
            parts.append(f"{'#' * heading_level} {section.title}")
            if section.content.strip():
                parts.append(section.content.strip())

        if not parts:
            return ""
        return "\n\n".join(parts).rstrip() + "\n"

    @staticmethod
    def create_markdown_content(pdf_content: PdfContent) -> str:
        """Choose the Markdown body to persist for one parsed paper."""
        assembled = MarkdownWriter.create_markdown_from_sections(pdf_content.sections)
        if assembled.strip():
            return assembled

        docling_fallback = pdf_content.metadata.get("docling_markdown_fallback")
        if isinstance(docling_fallback, str) and docling_fallback.strip():
            return docling_fallback

        if pdf_content.markdown and pdf_content.markdown.strip():
            return pdf_content.markdown

        raise ValueError("Parsed paper has no Markdown content")

    def _try_write_paper(
        self,
        arxiv_id: str,
        parsed_paper: ParsedPaper,
        result: ArtifactWriteResult,
    ) -> None:
        try:
            result.artifacts[arxiv_id] = self.write_paper(arxiv_id, parsed_paper)
        except Exception as exc:
            error = f"Failed to write parsed artifacts for {arxiv_id}: {exc}"
            logger.exception(error)
            result.errors.append(error)
