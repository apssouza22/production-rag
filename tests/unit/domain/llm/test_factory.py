from src.domain.bifrost.client import BifrostClient
from src.domain.llm.factory import make_agent_llm_client, make_llm_client
from src.domain.ollama.client import OllamaClient
from src.config import Settings


def test_make_llm_client_uses_ollama_by_default():
    settings = Settings(llm_provider="ollama")
    client = make_llm_client(settings)
    assert isinstance(client, OllamaClient)


def test_make_llm_client_uses_bifrost_when_configured():
    settings = Settings(llm_provider="bifrost")
    client = make_llm_client(settings)
    assert isinstance(client, BifrostClient)


def test_make_agent_llm_client_uses_agent_specific_api_key():
    settings = Settings(
        llm_provider="bifrost",
        bifrost_api_key_agent_1="sk-bf-agent-1-dev",
        bifrost_api_key_agent_2="sk-bf-agent-2-dev",
    )
    agent_1_client = make_agent_llm_client("agent_1", settings)
    agent_2_client = make_agent_llm_client("agent_2", settings)

    assert isinstance(agent_1_client, BifrostClient)
    assert isinstance(agent_2_client, BifrostClient)
    assert agent_1_client.api_key == "sk-bf-agent-1-dev"
    assert agent_2_client.api_key == "sk-bf-agent-2-dev"
