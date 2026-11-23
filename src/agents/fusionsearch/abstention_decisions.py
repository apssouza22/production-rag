"""Abstention judge via the configured decisions backend."""

from langchain_core.decisions import Choice

from src.config import ReasoningEffort, Settings
from src.platform.decisions import DecisionsClient, StructuredDecisionEvaluator

from .models import AnswerJudgement
from .prompts import (
    ABSTENTION_JUDGE_PROMPT,
    ABSTENTION_VERDICT_CHOICES,
    ABSTENTION_VERDICT_INSTRUCTIONS,
)


class AbstentionJudgeEvaluator(StructuredDecisionEvaluator[AnswerJudgement]):
    def __init__(
        self,
        question: str,
        context: str,
        answer: str,
        settings: Settings,
        model_name: str,
        *,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        super().__init__(settings, model_name, reasoning_effort=reasoning_effort)
        self._question = question
        self._context = context
        self._answer = answer
        self._allowed_verdicts = set(ABSTENTION_VERDICT_CHOICES.keys())

    @property
    def component_name(self) -> str:
        return "Abstention judge"

    def _decision_input(self) -> str:
        return f"Question:\n{self._question}\n\nRetrieved context:\n{self._context}\n\nGenerated answer:\n{self._answer}"

    async def evaluate_with_decisions(self) -> AnswerJudgement:
        response = await self._decisions.invoke(
            self._decision_input(),
            {
                "verdict": Choice(
                    instructions=ABSTENTION_VERDICT_INSTRUCTIONS,
                    criteria=ABSTENTION_VERDICT_CHOICES,
                )
            },
        )
        verdict, confidence = DecisionsClient.require_choice(response, "verdict", self._allowed_verdicts)
        return AnswerJudgement(
            verdict=verdict,  # type: ignore[arg-type]
            confidence=confidence,
            reasoning=f"Decisions verdict {verdict} with confidence {confidence:.2f}",
        )

    async def evaluate_with_structured_output(self) -> AnswerJudgement:
        structured_llm = self._chat_model().with_structured_output(AnswerJudgement)
        prompt = ABSTENTION_JUDGE_PROMPT.format(
            question=self._question,
            context=self._context,
            answer=self._answer,
        )
        return await structured_llm.ainvoke(prompt)


async def decide_answer_judgement(
    question: str,
    context: str,
    answer: str,
    settings: Settings,
    model_name: str,
    *,
    reasoning_effort: ReasoningEffort | None = None,
) -> AnswerJudgement:
    """Judge answer grounding via Decisions, falling back to structured chat output."""
    return await AbstentionJudgeEvaluator(
        question,
        context,
        answer,
        settings,
        model_name,
        reasoning_effort=reasoning_effort,
    ).run()
