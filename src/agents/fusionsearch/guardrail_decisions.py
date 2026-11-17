"""Guardrail scoring via OpenAI Decisions API."""

from langchain_openai.decisions import Predicate

from src.config import ReasoningEffort, Settings
from src.platform.decisions import DecisionsClient, StructuredDecisionEvaluator

from .models import GuardrailScoring
from .prompts import GUARDRAIL_IN_SCOPE_INSTRUCTIONS, GUARDRAIL_PROMPT


class GuardrailEvaluator(StructuredDecisionEvaluator[GuardrailScoring]):
    def __init__(
        self,
        query: str,
        settings: Settings,
        model_name: str,
        *,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        super().__init__(settings, model_name, reasoning_effort=reasoning_effort)
        self._query = query

    @property
    def component_name(self) -> str:
        return "Guardrail"

    async def evaluate_with_decisions(self) -> GuardrailScoring:
        response = await self._decisions.invoke(
            self._query,
            {"in_scope": Predicate(instructions=GUARDRAIL_IN_SCOPE_INSTRUCTIONS)},
        )
        probability = DecisionsClient.require_predicate(response, "in_scope")
        score = max(0, min(100, round(probability * 100)))
        return GuardrailScoring(
            score=score,
            reason=f"Decisions in-scope probability {probability:.2f}",
        )

    async def evaluate_with_structured_output(self) -> GuardrailScoring:
        structured_llm = self._chat_model().with_structured_output(GuardrailScoring)
        return await structured_llm.ainvoke(GUARDRAIL_PROMPT.format(question=self._query))


async def decide_guardrail_scoring(
    query: str,
    settings: Settings,
    model_name: str,
    *,
    reasoning_effort: ReasoningEffort | None = None,
) -> GuardrailScoring:
    """Score guardrail relevance via Decisions, falling back to structured chat output."""
    return await GuardrailEvaluator(
        query,
        settings,
        model_name,
        reasoning_effort=reasoning_effort,
    ).run()
