"""Filesystem artifacts shared between PDF parsing and persistence tasks."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import TypedDict

from src.domain.pdf_parser.models import ParsedPaper


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


def write_parsed_paper_artifacts(output_dir: Path, parsed_paper: ParsedPaper) -> ParsedPaperArtifact:
    """Write Markdown and a structured JSON sidecar for one parsed paper."""
    pdf_content = parsed_paper.pdf_content
    if pdf_content is None or pdf_content.markdown is None:
        raise ValueError(f"Parsed paper {parsed_paper.arxiv_metadata.arxiv_id} has no Markdown content")

    stem = artifact_stem(parsed_paper.arxiv_metadata.arxiv_id)
    markdown_path = output_dir / f"{stem}.md"
    sidecar_path = output_dir / f"{stem}.json"

    payload = parsed_paper.model_dump(mode="json")
    serialized_pdf_content = payload.get("pdf_content")
    if isinstance(serialized_pdf_content, dict):
        serialized_pdf_content.pop("markdown", None)

    _atomic_write_text(markdown_path, pdf_content.markdown)
    _atomic_write_text(sidecar_path, json.dumps(payload, indent=2, sort_keys=True))

    return {
        "markdown_path": str(markdown_path.resolve()),
        "sidecar_path": str(sidecar_path.resolve()),
    }


def load_parsed_paper_artifact(sidecar_path: str | Path) -> ParsedPaper:
    """Load and validate one structured parsed-paper sidecar."""
    return ParsedPaper.model_validate_json(Path(sidecar_path).read_text(encoding="utf-8"))
