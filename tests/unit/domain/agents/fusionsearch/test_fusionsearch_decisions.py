from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_openai.decisions import ChoiceAnswer, DecisionResponse, PredicateAnswer
from src.agents.fusionsearch.abstention_decisions import decide_answer_judgement
from src.agents.fusionsearch.grading_decisions import decide_grade_documents
from src.agents.fusionsearch.guardrail_decisions import decide_guardrail_scoring
from src.agents.fusionsearch.models import AnswerJudgement, GradeDocuments, GuardrailScoring
from src.config import Settings


def _predicate_response(name: str, probability: float) -> DecisionResponse:
    return DecisionResponse(
        model="gpt-test",
        answers={name: PredicateAnswer(type="predicate", probability=probability)},
    )


def _choice_response(name: str, choice: str, confidence: float) -> DecisionResponse:
    answer = ChoiceAnswer(
        type="choice",
        choice=choice,
        probabilities={choice: confidence},
        confidence=confidence,
    )
    return DecisionResponse(model="gpt-test", answers={name: answer})


@pytest.mark.asyncio
async def test_decide_guardrail_scoring_from_predicate():
    settings = Settings(openai_api_key="k", agent_model="gpt-6-luna")
    mock_decisions = Mock()
    mock_decisions.ainvoke = AsyncMock(return_value=_predicate_response("in_scope", 0.85))

    with patch(
        "src.platform.decisions.client.make_openai_decisions",
        return_value=mock_decisions,
    ):
        result = await decide_guardrail_scoring("What is BERT?", settings, "gpt-6-luna")

    assert result == GuardrailScoring(score=85, reason="Decisions in-scope probability 0.85")


@pytest.mark.asyncio
async def test_decide_grade_documents_no_when_below_threshold():
    settings = Settings(openai_api_key="k", agent_model="gpt-6-luna")
    mock_decisions = Mock()
    mock_decisions.ainvoke = AsyncMock(return_value=_predicate_response("relevant", 0.4))

    with patch(
        "src.platform.decisions.client.make_openai_decisions",
        return_value=mock_decisions,
    ):
        result = await decide_grade_documents("q", "ctx", settings, "gpt-6-luna")

    assert result.binary_score == "no"


@pytest.mark.asyncio
async def test_decide_answer_judgement_maps_choice():
    settings = Settings(openai_api_key="k", agent_model="gpt-6-luna")
    mock_decisions = Mock()
    mock_decisions.ainvoke = AsyncMock(return_value=_choice_response("verdict", "supported", 0.91))

    with patch(
        "src.platform.decisions.client.make_openai_decisions",
        return_value=mock_decisions,
    ):
        result = await decide_answer_judgement("q", "ctx", "ans", settings, "gpt-6-luna")

    assert result.verdict == "supported"
    assert result.confidence == 0.91