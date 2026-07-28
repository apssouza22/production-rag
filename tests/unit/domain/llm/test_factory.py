from src.domain.bifrost.client import BifrostClient
from src.domain.llm.factory import make_llm_client
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
