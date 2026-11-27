from pydantic import BaseModel, Field, model_validator


class SqlGenerationOutput(BaseModel):
    """Structured output for SQL generation (execute query or finish with an answer)."""

    thought: str = Field(description="Brief reasoning about tables, joins, filters, and approach before acting")
    query: str | None = Field(
        default=None,
        description="PostgreSQL SELECT (or WITH ... SELECT) to run when the database must be queried",
    )
    final_answer: str | None = Field(
        default=None,
        description="Natural-language answer to the user when no further SQL is required",
    )

    @model_validator(mode="after")
    def validate_action(self) -> "SqlGenerationOutput":
        has_query = bool(self.query and self.query.strip())
        has_answer = bool(self.final_answer and self.final_answer.strip())
        if has_query == has_answer:
            raise ValueError("Provide exactly one of `query` or `final_answer` (not both, not neither).")
        return self


class SqlCheckOutput(BaseModel):
    """Structured output for the SQL review step before execution."""

    thought: str = Field(description="Review notes, mistakes found, and changes made")
    query: str = Field(description="Corrected or unchanged SQL query to execute")