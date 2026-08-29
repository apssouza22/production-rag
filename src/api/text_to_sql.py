from fastapi import APIRouter, HTTPException

from src.dependencies import TextToSQLDep
from src.agents.texttosql.schemas import TextToSQLRequest, TextToSQLResponse

router = APIRouter(prefix="/api/v1", tags=["text-to-sql"])


@router.post("/ask-sql", response_model=TextToSQLResponse)
async def ask_sql(
    request: TextToSQLRequest,
    text_to_sql: TextToSQLDep,
) -> TextToSQLResponse:
    """
    Ask a natural-language question about the PostgreSQL database.

    The agent follows the LangGraph SQL workflow:
    1. Lists available tables
    2. Fetches relevant schemas
    3. Generates a read-only SQL query
    4. Validates the query before execution
    5. Returns a natural-language answer
    """
    try:
        result = await text_to_sql.ask(
            query=request.query,
            model=request.model,
        )

        return TextToSQLResponse(
            query=result["query"],
            answer=result["answer"],
            sql_queries=result.get("sql_queries", []),
            reasoning_steps=result.get("reasoning_steps", []),
            execution_time=result.get("execution_time", 0.0),
            trace_id=result.get("trace_id"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing SQL question: {exc}") from exc
