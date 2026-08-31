"""Shared LLM abstractions and provider factory."""

__all__ = ["LlmProviderClient", "make_llm_client"]


def __getattr__(name: str):
    if name == "LlmProviderClient":
        from src.platform.llm.protocol import LlmProviderClient

        return LlmProviderClient
    if name == "make_llm_client":
        from src.platform.llm.factory import make_llm_client

        return make_llm_client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
