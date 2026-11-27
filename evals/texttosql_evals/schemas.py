"""Schemas for text-to-SQL evaluation."""

from pydantic import BaseModel, Field


class EvalQuery(BaseModel):
    """A natural-language question with gold SQL for execution-accuracy scoring."""

    user_input: str = Field(min_length=1, description="Natural-language question for the agent")
    reference_sql: str = Field(min_length=1, description="Read-only SQL whose result is the gold answer")
    id: str | None = Field(default=None, description="Optional stable identifier for reports")