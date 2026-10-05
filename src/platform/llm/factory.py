from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings, get_settings
from src.platform.llm.chat_model import check_openai_health, make_chat_model

__all__ = ["make_chat_model", "check_openai_health"]


def make_default_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Create the default agent chat model from application settings."""
    resolved = settings or get_settings()
    return make_chat_model(resolved, resolved.agent_model, temperature=0.0)
