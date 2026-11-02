from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.domain.arxiv.paper import ArxivPaper


class DownloadResults(BaseModel):
    """Output of the ingestion download step (metadata + local PDF paths)."""

    papers_fetched: int = 0
    pdfs_downloaded: int = 0
    errors: list[str] = Field(default_factory=list)
    papers: list[ArxivPaper] = Field(default_factory=list)
    downloaded_paths: dict[str, str] = Field(default_factory=dict)
    processing_time: float = 0.0
    date: str | None = None

    @classmethod
    def from_xcom(cls, payload: dict[str, Any] | None) -> DownloadResults | None:
        """Deserialize Airflow XCom payload; ``None`` when the upstream task did not push results."""
        if payload is None:
            return None
        return cls.model_validate(payload)


class PersistResults(BaseModel):
    """Output of the ingestion persist step (PostgreSQL upserts)."""

    pdfs_parsed: int = 0
    papers_stored: int = 0
    errors: list[str] = Field(default_factory=list)
    date: str | None = None
    processing_time: float = 0.0

    @classmethod
    def from_xcom(cls, payload: dict[str, Any] | None) -> PersistResults | None:
        """Deserialize Airflow XCom payload; ``None`` when the upstream task did not push results."""
        if payload is None:
            return None
        return cls.model_validate(payload)
