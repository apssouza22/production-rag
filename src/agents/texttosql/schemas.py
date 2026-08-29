from typing import List, Optional

from pydantic import BaseModel, Field


class TextToSQLRequest(BaseModel):
    """Request model for text-to-SQL question answering."""

    query: str = Field(..., description="Natural-language question about the database", min_length=1, max_length=1000)
    model: str = Field("gpt-4o-mini", description="LLM model to use for SQL generation")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How many papers have been processed into the database?",
                "model": "gpt-4o-mini",
            }
        }


class TextToSQLResponse(BaseModel):
    """Response model for text-to-SQL question answering."""

    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Natural-language answer based on SQL results")
    sql_queries: List[str] = Field(default_factory=list, description="SQL queries executed by the agent")
    reasoning_steps: List[str] = Field(default_factory=list, description="Agent workflow steps")
    execution_time: float = Field(..., description="Total execution time in seconds")
    trace_id: Optional[str] = Field(None, description="Langfuse trace ID when tracing is enabled")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How many papers have been processed into the database?",
                "answer": "There are 42 papers in the database.",
                "sql_queries": ["SELECT COUNT(*) FROM papers;"],
                "reasoning_steps": [
                    "Listed available database tables",
                    "Fetched relevant table schemas",
                    "Executed 1 SQL query",
                    "Generated natural-language answer from query results",
                ],
                "execution_time": 3.42,
                "trace_id": "abc123-def456-ghi789",
            }
        }
