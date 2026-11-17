"""Template for evaluators that prefer OpenAI Decisions with structured chat fallback."""

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.config import ReasoningEffort, Settings
from src.platform.llm.chat_model import make_chat_model

from .client import DecisionsClient

TResult = TypeVar("TResult")


class StructuredDecisionEvaluator(ABC, Generic[TResult]):
    """Try OpenAI Decisions first; on failure use ``with_structured_output``."""

    def __init__(
        self,
        settings: Settings,
        model_name: str,
        *,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        self._settings = settings
        self._model_name = model_name
        self._reasoning_effort = reasoning_effort
        self._decisions = DecisionsClient(settings, model_name)
        self._logger = logging.getLogger(self.__class__.__module__)

    @property
    @abstractmethod
    def component_name(self) -> str:
        """Short label for log lines (e.g. 'Guardrail')."""

    async def run(self) -> TResult:
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
        """Map a Decisions API response to the domain result type."""

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
