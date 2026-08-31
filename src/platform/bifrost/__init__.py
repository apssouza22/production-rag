"""Bifrost gateway LLM client package."""

__all__ = ["BifrostClient"]


def __getattr__(name: str):
    if name == "BifrostClient":
        from .client import BifrostClient

        return BifrostClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
