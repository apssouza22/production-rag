"""Ollama direct LLM client package."""

__all__ = ["OllamaClient"]


def __getattr__(name: str):
    if name == "OllamaClient":
        from .client import OllamaClient

        return OllamaClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
