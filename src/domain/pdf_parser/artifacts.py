"""Filesystem artifacts produced and consumed by PDF parsing."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import TypedDict

from .models import ParsedPaper, PaperSection, PdfContent


class ParsedPaperArtifact(TypedDict):
    """Paths needed to persist one parsed paper."""

    markdown_path: str
    sidecar_path: str


def artifact_stem(arxiv_id: str) -> str:
    """Return a stable, filesystem-safe artifact stem for an arXiv ID."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", arxiv_id).strip("._-") or "paper"
    digest = hashlib.sha256(arxiv_id.encode("utf-8")).hexdigest()[:12]
    return f"{sanitized}-{digest}"


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


def assemble_markdown_from_sections(sections: list[PaperSection]) -> str:
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


def resolve_markdown_for_write(pdf_content: PdfContent) -> str:
    """Choose the Markdown body to persist for one parsed paper."""
    assembled = assemble_markdown_from_sections(pdf_content.sections)
    if assembled.strip():
        return assembled

    docling_fallback = pdf_content.metadata.get("docling_markdown_fallback")
    if isinstance(docling_fallback, str) and docling_fallback.strip():
        return docling_fallback

    if pdf_content.markdown and pdf_content.markdown.strip():
        return pdf_content.markdown

    raise ValueError("Parsed paper has no Markdown content")


def write_parsed_paper_artifacts(
    output_dir: Path,
    parsed_paper: ParsedPaper,
    *,
    markdown: str | None = None,
) -> ParsedPaperArtifact:
    """Write Markdown and a structured JSON sidecar for one parsed paper."""
    pdf_content = parsed_paper.pdf_content
    if pdf_content is None:
        raise ValueError(f"Parsed paper {parsed_paper.arxiv_metadata.arxiv_id} has no PDF content")

    markdown_body = markdown if markdown is not None else resolve_markdown_for_write(pdf_content)

    stem = artifact_stem(parsed_paper.arxiv_metadata.arxiv_id)
    markdown_path = output_dir / f"{stem}.md"
    sidecar_path = output_dir / f"{stem}.json"

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


def load_parsed_paper_artifact(sidecar_path: str | Path) -> ParsedPaper:
    """Load and validate one structured parsed-paper sidecar."""
    return ParsedPaper.model_validate_json(Path(sidecar_path).read_text(encoding="utf-8"))
