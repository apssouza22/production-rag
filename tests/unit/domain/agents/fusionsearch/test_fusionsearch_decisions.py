from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.agents.fusionsearch.abstention_decisions import decide_answer_judgement
from src.agents.fusionsearch.grading_decisions import decide_grade_documents
from src.agents.fusionsearch.guardrail_decisions import decide_guardrail_scoring
from src.agents.fusionsearch.models import AnswerJudgement, GradeDocuments, GuardrailScoring
from src.config import Settings
from src.platform.decisions.response import NormalizedChoiceAnswer, NormalizedDecisionResponse


def _noul_response(name: str, probability: float) -> NormalizedDecisionResponse:
    return NormalizedDecisionResponse(model="gpt-test", nouls={name: probability})


def _choice_response(name: str, choice: str, confidence: float) -> NormalizedDecisionResponse:
    return NormalizedDecisionResponse(
        model="gpt-test",
        choices={name: NormalizedChoiceAnswer(choice=choice, confidence=confidence)},
    )


def _mock_backend(return_value: NormalizedDecisionResponse) -> Mock:
    backend = Mock()
    backend.ainvoke = AsyncMock(return_value=return_value)
    return backend


@pytest.mark.asyncio
async def test_decide_guardrail_scoring_uses_decisions():
    settings = Settings(openai_api_key="k", agent_model="gpt-6-luna")

    with patch(
        "src.platform.decisions.client.make_decision_backend",
        return_value=_mock_backend(_noul_response("in_scope", 0.85)),
    ):
        result = await decide_guardrail_scoring("What is BERT?", settings, "gpt-6-luna")

    assert result == GuardrailScoring(score=85, reason="Decisions in-scope probability 0.85")


@pytest.mark.asyncio
async def test_decide_grade_documents_no_when_below_threshold():
    settings = Settings(openai_api_key="k", agent_model="gpt-6-luna")

    with patch(
        "src.platform.decisions.client.make_decision_backend",
        return_value=_mock_backend(_noul_response("relevant", 0.4)),
    ):
        result = await decide_grade_documents("q", "ctx", settings, "gpt-6-luna")

    assert result.binary_score == "no"


@pytest.mark.asyncio
async def test_decide_answer_judgement_maps_choice():
    settings = Settings(openai_api_key="k", agent_model="gpt-6-luna")

    with patch(
        "src.platform.decisions.client.make_decision_backend",
        return_value=_mock_backend(_choice_response("verdict", "supported", 0.91)),
    ):
        result = await decide_answer_judgement("q", "ctx", "ans", settings, "gpt-6-luna")

    assert result.verdict == "supported"
    assert result.confidence == 0.91
