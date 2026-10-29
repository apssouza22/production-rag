"""Normalized schemas shared by all evaluation implementations."""

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, model_validator


class EvalSample(BaseModel):
    """One normalized example produced by any evaluation dataset."""

    input: str | None = Field(default=None, validation_alias=AliasChoices("input", "user_input"))
    output: str | None = Field(default=None, validation_alias=AliasChoices("output", "response"))
    reference: str | None = None
    contexts: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("contexts", "retrieved_contexts"),
    )
    source_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def move_source_fields_to_metadata(cls, value: Any) -> Any:
        if isinstance(value, dict) and "rerank_applied" in value:
            normalized = dict(value)
            metadata = dict(normalized.get("metadata") or {})
            metadata.setdefault("rerank_applied", bool(normalized.pop("rerank_applied")))
            normalized["metadata"] = metadata
            return normalized
        return value

    @property
    def user_input(self) -> str:
        return self.input or ""

    @property
    def response(self) -> str:
        return self.output or ""

    @property
    def retrieved_contexts(self) -> list[str]:
        return self.contexts

    @property
    def rerank_applied(self) -> bool:
        return bool(self.metadata.get("rerank_applied", False))


class MetricResult(BaseModel):
    """Normalized result returned by every evaluation metric."""

    name: str
    value: float | None = None
    reasoning: str | None = None
    error: str | None = None


class ScoreSchema(BaseModel):
    """Structured output returned by an LLM-based metric."""

    score: float = Field(description="provide a score between 0 and 1")
    reasoning: str = Field(description="provide a one sentence reasoning")
