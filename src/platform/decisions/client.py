"""Thin OO wrapper around LangChain ``OpenAIDecisions``."""

from typing import Any

from langchain_openai.decisions import Choice, DecisionResponse, OpenAIDecisions, Predicate

from src.config import Settings

from .factory import make_openai_decisions, resolve_decisions_model


class DecisionsClient:
    """Invoke the OpenAI Decisions API using application settings."""

    def __init__(self, settings: Settings, chat_model_name: str) -> None:
        self._settings = settings
        self._chat_model_name = chat_model_name
        self._runnable: OpenAIDecisions = make_openai_decisions(
            settings,
            resolve_decisions_model(settings, chat_model_name),
        )

    async def invoke(
        self,
        decision_input: Any,
        questions: dict[str, Predicate | Choice],
    ) -> DecisionResponse:
        return await self._runnable.ainvoke({"input": decision_input, "questions": questions})

    @staticmethod
    def refusal_names(response: DecisionResponse) -> set[str]:
        return set(response.refusals.keys()) if response.refusals else set()

    @classmethod
    def require_predicate(cls, response: DecisionResponse, name: str) -> float:
        if name in cls.refusal_names(response) or name not in response.predicates:
            raise ValueError(f"Decisions refused or omitted predicate '{name}'")
        return float(response.predicates[name].probability)

    @classmethod
    def require_choice(
        cls,
        response: DecisionResponse,
        name: str,
        allowed: set[str],
    ) -> tuple[str, float]:
        if name in cls.refusal_names(response) or name not in response.choices:
            raise ValueError(f"Decisions refused or omitted choice '{name}'")
        answer = response.choices[name]
        choice = str(answer.choice)
        if choice not in allowed:
            raise ValueError(f"Unexpected choice: {choice}")
        return choice, float(answer.confidence)
