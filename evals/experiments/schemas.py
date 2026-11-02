"""Schemas for the local-PDF retrieval experiment."""

from pydantic import BaseModel, Field


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
