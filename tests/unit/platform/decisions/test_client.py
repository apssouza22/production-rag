import pytest
from langchain_openai.decisions import DecisionResponse, PredicateAnswer, RefusalAnswer

from src.platform.decisions import DecisionsClient


def test_require_predicate_reads_probability():
    response = DecisionResponse(
        model="gpt-test",
        answers={"relevant": PredicateAnswer(type="predicate", probability=0.75)},
    )
    assert DecisionsClient.require_predicate(response, "relevant") == 0.75


def test_require_predicate_raises_on_refusal():
    response = DecisionResponse(
        model="gpt-test",
        answers={"relevant": RefusalAnswer(type="refusal")},
    )
    with pytest.raises(ValueError, match="refused"):
        DecisionsClient.require_predicate(response, "relevant")