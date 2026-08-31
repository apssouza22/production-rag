"""Ollama direct LLM client package."""

__all__ = [
    "OllamaClient",
    "OllamaConnectionError",
    "OllamaException",
    "OllamaTimeoutError",
]


def __getattr__(name: str):
    if name == "OllamaClient":
        from .client import OllamaClient

        return OllamaClient
    if name in {"OllamaConnectionError", "OllamaException", "OllamaTimeoutError"}:
        from . import exceptions

        return getattr(exceptions, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
