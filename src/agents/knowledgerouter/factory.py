from typing import Optional

from src.agents.fusionsearch.agentic_rag import AgenticRAGService
from src.agents.texttosql.service import TextToSQLService
from src.config import get_settings
from src.platform.langfuse.client import LangfuseTracer
from src.platform.llm.chat_model import make_chat_model

from .config import KnowledgeRouterConfig
from .service import KnowledgeRouterService


def make_knowledge_router_service(
    agentic_rag_service: AgenticRAGService,
    text_to_sql_service: TextToSQLService,
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
        router_model=make_chat_model(
            settings,
            agent_config.router_model,
            temperature=agent_config.temperature,
        ),
        synthesis_model=make_chat_model(
            settings,
            agent_config.model,
            temperature=agent_config.temperature,
        ),
        langfuse_tracer=langfuse_tracer,
        agent_config=agent_config,
    )
