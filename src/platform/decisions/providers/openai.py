"""OpenAI Decisions backend using :mod:`langchain_openai.decisions`."""

from langchain_core.decisions import Question, State
from langchain_openai.decisions import OpenAIDecisions
from pydantic import SecretStr

from src.config import Settings
from src.platform.llm.chat_model import normalize_model

from ..adapters.questions import to_openai_questions
from ..adapters.responses import from_openai_decision_response
from ..response import NormalizedDecisionResponse

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def resolve_decisions_model(settings: Settings, model: str | None = None) -> str:
    """Resolve the decisions model name from overrides and application settings."""
    if model is not None and model.strip():
        return normalize_model(model)
    if settings.decisions_model.strip():
        return normalize_model(settings.decisions_model)
    return normalize_model(settings.agent_model)


def make_openai_decisions(
    settings: Settings,
    model: str | None = None,
    *,
    api_key: str | None = None,
) -> OpenAIDecisions:
    """Create a LangChain OpenAI Decisions runnable aligned with OpenAI settings."""
    resolved_model = resolve_decisions_model(settings, model)
    resolved_key = api_key or settings.openai_api_key
    base_url = (settings.openai_base_url or _DEFAULT_BASE_URL).rstrip("/")
    return OpenAIDecisions(
        model=resolved_model,
        api_key=SecretStr(resolved_key) if resolved_key else None,
        base_url=base_url,
        timeout=float(settings.llm_request_timeout),
    )


class OpenAIDecisionBackend:
    """Invoke OpenAI's Decisions API with portable question types.

    This backend converts portable ``Noul`` and ``Choice`` questions to OpenAI's
    ``Predicate`` and ``choices`` fields, calls ``ainvoke`` with the ``input`` key,
    and maps predicate probabilities into normalized ``nouls``.
    """

    def __init__(self, runnable: OpenAIDecisions) -> None:
        self._runnable = runnable

    async def ainvoke(
        self,
        state: State,
        questions: dict[str, Question],
    ) -> NormalizedDecisionResponse:
        """Run OpenAI Decisions and return a normalized response."""
        provider_questions = to_openai_questions(questions)
        raw = await self._runnable.ainvoke({"input": state, "questions": provider_questions})
        return from_openai_decision_response(raw)
