from typing import Optional

from src.config import get_settings
from src.domain.langfuse.client import LangfuseTracer
from src.domain.llm.protocol import LLMClient

from .config import TextToSQLConfig
from .service import TextToSQLService


def make_text_to_sql_service(
    llm_client: LLMClient,
    langfuse_tracer: Optional[LangfuseTracer] = None,
    model: Optional[str] = None,
    top_k: int = 5,
    include_tables: Optional[list[str]] = None,
) -> TextToSQLService:
    """Create a configured TextToSQLService instance."""
    settings = get_settings()
    agent_config = TextToSQLConfig(
        model=model or settings.agent_model,
        top_k=top_k,
        include_tables=include_tables or ["papers"],
    )

    return TextToSQLService(
        llm_client=llm_client,
        langfuse_tracer=langfuse_tracer,
        agent_config=agent_config,
    )
