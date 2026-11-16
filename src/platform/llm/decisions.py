from langchain_openai.decisions import OpenAIDecisions
from pydantic import SecretStr

from src.config import Settings
from src.platform.llm.chat_model import normalize_model

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def resolve_decisions_model(settings: Settings, model: str | None = None) -> str:
    """Return the decisions model name, preferring explicit overrides then settings."""
    if model is not None and model.strip():
        return normalize_model(model)
    if settings.decisions_model.strip():
        return normalize_model(settings.decisions_model)
    return normalize_model(settings.agent_model)


def make_openai_decisions(
    settings: Settings,
    model: str | None = None,
    *,
    api_key: str | None = None,
) -> OpenAIDecisions:
    """Create a LangChain OpenAI Decisions runnable aligned with application OpenAI settings."""
    resolved_model = resolve_decisions_model(settings, model)
    resolved_key = api_key or settings.openai_api_key
    base_url = (settings.openai_base_url or _DEFAULT_BASE_URL).rstrip("/")
    return OpenAIDecisions(
        model=resolved_model,
        api_key=SecretStr(resolved_key) if resolved_key else None,
        base_url=base_url,
        timeout=float(settings.llm_request_timeout),
    )
