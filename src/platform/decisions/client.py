"""Application-facing client for typed decision models."""

import warnings

from langchain_core.decisions import Question, State

from src.config import Settings

from .factory import make_decision_backend, resolve_decisions_model
from .protocol import DecisionBackend
from .response import NormalizedDecisionResponse


class DecisionsClient:
    """Invoke configured decision backends using portable LangChain question types.

    Callers pass :class:`~langchain_core.decisions.Noul` and
    :class:`~langchain_core.decisions.Choice`. Provider translation happens inside the
    injected :class:`~src.platform.decisions.protocol.DecisionBackend`.
    """

    def __init__(
        self,
        settings: Settings,
        chat_model_name: str,
        *,
        backend: DecisionBackend | None = None,
    ) -> None:
        """Bind a backend from settings or accept a pre-built backend (tests).

        Args:
            settings: Application configuration, including provider credentials.
            chat_model_name: Chat model name used to resolve the decisions model when unset.
            backend: Optional backend override; defaults to :func:`make_decision_backend`.
        """
        self._settings = settings
        self._chat_model_name = chat_model_name
        self._backend = backend or make_decision_backend(
            settings,
            resolve_decisions_model(settings, chat_model_name),
        )

    async def invoke(
        self,
        state: State,
        questions: dict[str, Question],
    ) -> NormalizedDecisionResponse:
        """Evaluate ``state`` against ``questions`` and return normalized answers.

        Args:
            state: Natural language or structured context for the model.
            questions: Portable questions keyed by the answer IDs callers will read.

        Returns:
            Provider-agnostic probabilities, choices, and refusal metadata.
        """
        return await self._backend.ainvoke(state, questions)

    @staticmethod
    def refusal_names(response: NormalizedDecisionResponse) -> set[str]:
        """Return question IDs the backend refused to answer."""
        return response.refusal_names()

    @classmethod
    def require_noul(cls, response: NormalizedDecisionResponse, name: str) -> float:
        """Return the yes-probability for a binary question or raise ``ValueError``.

        Args:
            response: Outcome from :meth:`invoke`.
            name: Question ID supplied in the ``questions`` mapping.

        Returns:
            A probability in ``[0, 1]`` representing confidence the condition is true.
        """
        if name in cls.refusal_names(response) or name not in response.nouls:
            raise ValueError(f"Decisions refused or omitted noul '{name}'")
        return float(response.nouls[name])

    @classmethod
    def require_choice(
        cls,
        response: NormalizedDecisionResponse,
        name: str,
        allowed: set[str],
    ) -> tuple[str, float]:
        """Return a categorical choice and confidence, validating membership.

        Args:
            response: Outcome from :meth:`invoke`.
            name: Question ID supplied in the ``questions`` mapping.
            allowed: Labels the application accepts; raises when the model picks another.

        Returns:
            The selected label and its confidence in ``[0, 1]``.
        """
        if name in cls.refusal_names(response) or name not in response.choices:
            raise ValueError(f"Decisions refused or omitted choice '{name}'")
        answer = response.choices[name]
        choice = str(answer.choice)
        if choice not in allowed:
            raise ValueError(f"Unexpected choice: {choice}")
        return choice, float(answer.confidence)
