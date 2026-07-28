from typing import Optional

from src.config import get_settings
from src.domain.agents.fusionsearch.agentic_rag import AgenticRAGService
from src.domain.agents.texttosql.service import TextToSQLService
from src.domain.langfuse.client import LangfuseTracer
from src.domain.llm.protocol import LLMClient

from .config import KnowledgeRouterConfig
from .service import KnowledgeRouterService


def make_knowledge_router_service(
    agentic_rag_service: AgenticRAGService,
    text_to_sql_service: TextToSQLService,
    llm_client: LLMClient,
    langfuse_tracer: Optional[LangfuseTracer] = None,
    model: Optional[str] = None,
) -> KnowledgeRouterService:
    """Create a configured KnowledgeRouterService instance."""
    settings = get_settings()
    agent_config = KnowledgeRouterConfig(
        model=model or settings.agent_model,
        router_model=model or settings.agent_model,
    )

    return KnowledgeRouterService(
        agentic_rag_service=agentic_rag_service,
        text_to_sql_service=text_to_sql_service,
        llm_client=llm_client,
        langfuse_tracer=langfuse_tracer,
        agent_config=agent_config,
    )
