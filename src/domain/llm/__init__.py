"""Shared LLM abstractions and provider factory."""

__all__ = ["LLMClient", "make_llm_client"]


def __getattr__(name: str):
    if name == "LLMClient":
        from src.domain.llm.protocol import RagService

        return RagService
    if name == "make_llm_client":
        from src.domain.llm.factory import make_llm_client

        return make_llm_client
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
