"""Filesystem artifacts produced and consumed by PDF parsing."""

import hashlib
import re
from pathlib import Path
from typing import TypedDict

from .models import ParsedPaper


class ParsedPaperArtifact(TypedDict):
    """Paths needed to persist one parsed paper."""

    markdown_path: str
    sidecar_path: str


def artifact_stem(arxiv_id: str) -> str:
    """Return a stable, filesystem-safe artifact stem for an arXiv ID."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", arxiv_id).strip("._-") or "paper"
    digest = hashlib.sha256(arxiv_id.encode("utf-8")).hexdigest()[:12]
    return f"{sanitized}-{digest}"


def load_parsed_paper_artifact(sidecar_path: str | Path) -> ParsedPaper:
    """Load and validate one structured parsed-paper sidecar."""
    return ParsedPaper.model_validate_json(Path(sidecar_path).read_text(encoding="utf-8"))
