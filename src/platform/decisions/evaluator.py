"""Template for evaluators that prefer typed decisions with structured chat fallback."""

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.config import ReasoningEffort, Settings
from src.platform.llm.chat_model import make_chat_model

from .client import DecisionsClient

TResult = TypeVar("TResult")


class StructuredDecisionEvaluator(ABC, Generic[TResult]):
    """Try the configured decision backend first; on failure use structured chat output.

    Subclasses implement provider-specific mapping in
    :meth:`evaluate_with_decisions` using portable
    :mod:`langchain_core.decisions` questions via :class:`DecisionsClient`.
    """

    def __init__(
        self,
        settings: Settings,
        model_name: str,
        *,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        """Wire settings, optional reasoning effort, and a decisions client."""
        self._settings = settings
        self._model_name = model_name
        self._reasoning_effort = reasoning_effort
        self._decisions = DecisionsClient(settings, model_name)
        self._logger = logging.getLogger(self.__class__.__module__)

    @property
    @abstractmethod
    def component_name(self) -> str:
        """Short label for log lines (e.g. ``'Guardrail'``)."""

    async def run(self) -> TResult:
        """Run the decisions path, falling back to structured chat on any error."""
        try:
            return await self.evaluate_with_decisions()
        except Exception as exc:
            self._logger.warning(
                "%s decisions call failed (%s), falling back to structured output",
                self.component_name,
                exc,
            )
            return await self.evaluate_with_structured_output()

    @abstractmethod
    async def evaluate_with_decisions(self) -> TResult:
        """Map a normalized decisions response to the domain result type."""

    @abstractmethod
    async def evaluate_with_structured_output(self) -> TResult:
        """Legacy structured chat completion path."""

    def _chat_model(self):
        return make_chat_model(
            self._settings,
            self._model_name,
            temperature=0.0,
            reasoning_effort=self._reasoning_effort,
        )
