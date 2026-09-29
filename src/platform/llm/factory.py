from typing import Literal

from src.config import Settings, get_settings
from src.platform.bifrost.factory import make_bifrost_client
from src.platform.llm.protocol import LlmProviderClient

AgentKey = Literal["agent_1", "agent_2"]


def make_llm_client(settings: Settings | None = None, api_key: str | None = None) -> LlmProviderClient:
    """Create the Bifrost-backed LLM client."""
    resolved_settings = settings or get_settings()
    return make_bifrost_client(resolved_settings, api_key=api_key)


def make_agent_llm_client(agent: AgentKey, settings: Settings | None = None) -> LlmProviderClient:
    """Create an LLM client using the Bifrost virtual key assigned to a LangGraph agent."""
    resolved_settings = settings or get_settings()
    api_key_by_agent = {
        "agent_1": resolved_settings.bifrost_api_key_agent_1,
        "agent_2": resolved_settings.bifrost_api_key_agent_2,
    }
    return make_llm_client(resolved_settings, api_key=api_key_by_agent[agent])
