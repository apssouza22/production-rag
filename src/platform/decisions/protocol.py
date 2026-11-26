"""Contracts for decision-model providers."""

from typing import Protocol

from langchain_core.decisions import Question, State

from .response import NormalizedDecisionResponse


class DecisionBackend(Protocol):
    """Portable async facade over a LangChain decision runnable.

    Implementations translate portable :mod:`langchain_core.decisions` questions
    into provider wire formats, invoke the underlying runnable, and normalize
    the provider response for :class:`~src.platform.decisions.client.DecisionsClient`.
    """

    async def ainvoke(
        self,
        state: State,
        questions: dict[str, Question],
    ) -> NormalizedDecisionResponse:
        """Evaluate ``state`` against ``questions`` and return normalized answers.

        Args:
            state: Text, messages, or structured JSON the model should read.
            questions: Portable question definitions keyed by answer ID.

        Returns:
            Binary probabilities, categorical choices, and any refusals in a
            single normalized structure.
        """
