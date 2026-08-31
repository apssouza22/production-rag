from src.config import Settings, get_settings
from src.platform.ollama.client import OllamaClient


def make_ollama_client(settings: Settings | None = None) -> OllamaClient:
    """Create and return an Ollama client instance."""
    resolved_settings = settings or get_settings()
    return OllamaClient(resolved_settings)
