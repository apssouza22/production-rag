"""Schemas for the local-PDF retrieval experiment."""

from pydantic import BaseModel, ConfigDict, Field


class ExperimentQuery(BaseModel):
    """One generated question with its reference answer and originating paper."""

    user_input: str = Field(
        min_length=1,
        max_length=2000,
        description="Evaluation question; must be answerable using only the paired reference and retrieved contexts.",
    )
    reference: str = Field(
        min_length=1,
        max_length=8000,
        description=(
            "Gold answer for metrics; every claim should appear in or follow from the source excerpts "
            "used when this question was created."
        ),
    )
    arxiv_id: str = Field(
        default="",
        max_length=32,
        description="arXiv id of the paper whose excerpts ground this question-reference pair (may be empty for synthetic sets).",
    )


class GeneratedQuestion(BaseModel):
    """Structured output item returned by the question-generation model."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "Self-contained question with enough context that someone with only paper excerpts can answer; "
            "no references to 'the text above' or unseen material."
        ),
    )
    reference: str = Field(
        min_length=1,
        max_length=8000,
        description=(
            "Short gold answer fully supported by the provided excerpts only; prefer precise wording "
            "and citations to chunk content where appropriate."
        ),
    )


class GeneratedQuestions(BaseModel):
    """Structured output wrapper returned by the question-generation model."""

    questions: list[GeneratedQuestion] = Field(
        default_factory=list,
        description=(
            "Batch of question-reference pairs, each grounded in the same excerpt bundle passed to the generator; "
            "omit pairs if the excerpts do not support a clean QA item."
        ),
    )


class KgIngestionSummary(BaseModel):
    """Stats returned by ``KgIndexingService.index_document_ids``."""

    documents: int = 0
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
