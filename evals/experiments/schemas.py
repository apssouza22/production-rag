"""Schemas for the local-PDF retrieval experiment."""

from pydantic import BaseModel, ConfigDict, Field


class ExperimentQuery(BaseModel):
    """One generated question with its reference answer and originating paper."""

    user_input: str = Field(min_length=1, description="Question to evaluate")
    reference: str = Field(min_length=1, description="Ground-truth answer for reference-based metrics")
    arxiv_id: str = Field(default="", description="Paper the question was generated from")


class GeneratedQuestion(BaseModel):
    """Structured output item returned by the question-generation model."""

    question: str = Field(description="A self-contained question answerable from the excerpts")
    reference: str = Field(description="The ground-truth answer, grounded in the excerpts")


class GeneratedQuestions(BaseModel):
    """Structured output wrapper returned by the question-generation model."""

    questions: list[GeneratedQuestion] = Field(default_factory=list)


class KgIngestionSummary(BaseModel):
    """Stats returned by ``KgIndexingService.index_arxiv_ids``."""

    papers: int = 0
    chunks_processed: int = 0
    chunks_skipped: int = 0
    chunks_failed: int = 0
    entities_upserted: int = 0
    relations_upserted: int = 0


class ExperimentIngestionResult(BaseModel):
    """Outcome of hybrid chunk ingest plus optional KG backfill for one experiment run."""

    model_config = ConfigDict(frozen=True)

    skipped: bool = Field(description="True when hybrid chunk indexing was skipped (partition already had chunks)")
    chunk_count: int
    ingestion_name: str
    pdfs_dir: str
    arxiv_ids: list[str]
    papers_stored: int | None = None
    pdfs_parsed: int | None = None
    chunks_indexed: int | None = None
    kg: KgIngestionSummary | None = None

    def model_dump_for_runner(self) -> dict[str, object]:
        """JSON-serializable dict preserving the shape expected by ``ExperimentRunner``."""
        return self.model_dump(mode="json", exclude_none=True)
