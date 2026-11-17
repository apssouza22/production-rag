"""Document grading via OpenAI Decisions API."""

from langchain_openai.decisions import Predicate

from src.config import ReasoningEffort, Settings
from src.platform.decisions import DecisionsClient, StructuredDecisionEvaluator
from .models import GradeDocuments
from .prompts import GRADE_DOCUMENTS_PROMPT, GRADE_RELEVANCE_INSTRUCTIONS, GRADE_RELEVANCE_THRESHOLD


class DocumentGradingEvaluator(StructuredDecisionEvaluator[GradeDocuments]):
    def __init__(
        self,
        question: str,
        context: str,
        settings: Settings,
        model_name: str,
        *,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        super().__init__(settings, model_name, reasoning_effort=reasoning_effort)
        self._question = question
        self._context = context

    @property
    def component_name(self) -> str:
        return "Document grading"

    def _decision_input(self) -> str:
        return f"User question:\n{self._question}\n\nRetrieved documents:\n{self._context}"

    async def evaluate_with_decisions(self) -> GradeDocuments:
        response = await self._decisions.invoke(
            self._decision_input(),
            {"relevant": Predicate(instructions=GRADE_RELEVANCE_INSTRUCTIONS)},
        )
        probability = DecisionsClient.require_predicate(response, "relevant")
        binary_score = "yes" if probability >= GRADE_RELEVANCE_THRESHOLD else "no"
        return GradeDocuments(
            binary_score=binary_score,
            reasoning=f"Decisions relevance probability {probability:.2f}",
        )

    async def evaluate_with_structured_output(self) -> GradeDocuments:
        structured_llm = self._chat_model().with_structured_output(GradeDocuments)
        prompt = GRADE_DOCUMENTS_PROMPT.format(context=self._context, question=self._question)
        return await structured_llm.ainvoke(prompt)


async def decide_grade_documents(
    question: str,
    context: str,
    settings: Settings,
    model_name: str,
    *,
    reasoning_effort: ReasoningEffort | None = None,
) -> GradeDocuments:
    """Grade retrieval relevance via Decisions, falling back to structured chat output."""
    return await DocumentGradingEvaluator(
        question,
        context,
        settings,
        model_name,
        reasoning_effort=reasoning_effort,
    ).run()
