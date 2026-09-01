from fastapi import APIRouter, HTTPException

from src.dependencies import KnowledgeRouterDep
from src.agents.knowledgerouter.schemas import KnowledgeRouterRequest, KnowledgeRouterResponse

router = APIRouter(prefix="/api/v1", tags=["knowledge-router"])


@router.post("/ask-router", response_model=KnowledgeRouterResponse)
async def ask_router(
    request: KnowledgeRouterRequest,
    knowledge_router: KnowledgeRouterDep,
) -> KnowledgeRouterResponse:
    """
    Route a question to the best knowledge source(s) and synthesize the answer.

    The router follows the LangGraph multi-agent pattern:
    1. Classifies the query into targeted sub-questions per knowledge source
    2. Routes to specialized agents in parallel (documents and/or database)
    3. Synthesizes results into a single coherent answer

    Available sources:
    - documents: Agentic RAG over arXiv paper content (concepts, explanations)
    - database: Text-to-SQL over PostgreSQL metadata (counts, listings, filters)
    """
    try:
        result = await knowledge_router.ask(query=request.query)

        trajectory = result.get("trajectory")
        if isinstance(trajectory, dict) and "summary" in trajectory:
            trajectory = trajectory["summary"]

        return KnowledgeRouterResponse(
            query=result["query"],
            answer=result["answer"],
            classifications=result.get("classifications", []),
            agent_results=result.get("agent_results", []),
            reasoning_steps=result.get("reasoning_steps", []),
            execution_time=result.get("execution_time", 0.0),
            trace_id=result.get("trace_id"),
            trajectory=trajectory,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing routed question: {exc}") from exc
