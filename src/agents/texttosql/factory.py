from typing import Optional

from src.config import get_settings
from src.platform.langfuse.client import LangfuseTracer
from src.platform.llm.chat_model import make_chat_model

from .config import TextToSQLConfig
from .service import TextToSQLService


def make_text_to_sql_service(
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
        chat_model=make_chat_model(settings, agent_config.model, temperature=agent_config.temperature),
        langfuse_tracer=langfuse_tracer,
        agent_config=agent_config,
    )
