from src.config import Settings, get_settings
from src.domain.bifrost.factory import make_bifrost_client
from src.domain.llm.protocol import LLMClient
from src.domain.ollama.factory import make_ollama_client


def make_llm_client(settings: Settings | None = None) -> LLMClient:
    """Create the configured LLM client (Ollama direct or Bifrost gateway)."""
    resolved_settings = settings or get_settings()

    if resolved_settings.llm_provider == "bifrost":
        return make_bifrost_client(resolved_settings)

    return make_ollama_client(resolved_settings)
