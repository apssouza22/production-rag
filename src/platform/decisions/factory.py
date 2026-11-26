"""Factory functions for decision backends and legacy OpenAI runnables."""

from typing import Literal

from src.config import Settings

from .exceptions import DecisionsProviderError
from .protocol import DecisionBackend
from .providers.openai import (
    OpenAIDecisionBackend,
    make_openai_decisions,
    resolve_decisions_model,
)
from .providers.typesafe import TypeSafeDecisionBackend, make_typesafe_classifier

DecisionsProvider = Literal["openai", "typesafe"]

_SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"openai", "typesafe"})


def make_decision_backend(
    settings: Settings,
    model: str | None = None,
) -> DecisionBackend:
    """Select and construct the configured decision backend.

    Args:
        settings: Application settings, including ``decisions_provider``.
        model: Optional model override; when omitted, :func:`resolve_decisions_model` applies.

    Returns:
        A backend that accepts portable ``langchain_core.decisions`` questions.

    Raises:
        DecisionsProviderError: When the provider is unknown or its integration is missing.
    """
    provider = settings.decisions_provider
    if provider not in _SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(_SUPPORTED_PROVIDERS))
        raise DecisionsProviderError(
            provider,
            f"Unsupported decisions_provider {provider!r}. Supported: {supported}.",
        )
    if provider == "openai":
        resolved_model = resolve_decisions_model(settings, model)
        return OpenAIDecisionBackend(make_openai_decisions(settings, resolved_model))
    if provider == "typesafe":
        return TypeSafeDecisionBackend(make_typesafe_classifier(settings))
    raise DecisionsProviderError(provider, "Unreachable provider branch.")


__all__ = [
    "DecisionsProvider",
    "make_decision_backend",
    "make_openai_decisions",
    "resolve_decisions_model",
]
