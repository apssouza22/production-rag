"""Backward-compatible re-exports; prefer ``src.platform.decisions``."""

from src.platform.decisions.factory import make_openai_decisions, resolve_decisions_model

__all__ = ["make_openai_decisions", "resolve_decisions_model"]