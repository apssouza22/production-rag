from typing import Literal

from src.config import Settings, get_settings
from src.platform.bifrost.factory import make_bifrost_client
from src.platform.llm.protocol import LlmProviderClient
from src.platform.ollama.factory import make_ollama_client

AgentKey = Literal["agent_1", "agent_2"]


def _make_llm_provider(settings: Settings, api_key: str | None = None) -> LlmProviderClient:
    if settings.llm_provider == "bifrost":
        return make_bifrost_client(settings, api_key=api_key)
    return make_ollama_client(settings)


def make_llm_client(settings: Settings | None = None, api_key: str | None = None) -> LlmProviderClient:
    """Create the configured LLM provider (Ollama direct or Bifrost gateway)."""
    resolved_settings = settings or get_settings()
    return _make_llm_provider(resolved_settings, api_key=api_key)


def make_agent_llm_client(agent: AgentKey, settings: Settings | None = None) -> LlmProviderClient:
    """Create an LLM client using the Bifrost virtual key assigned to a LangGraph agent."""
    resolved_settings = settings or get_settings()
    api_key_by_agent = {
        "agent_1": resolved_settings.bifrost_api_key_agent_1,
        "agent_2": resolved_settings.bifrost_api_key_agent_2,
    }
    return make_llm_client(resolved_settings, api_key=api_key_by_agent[agent])
