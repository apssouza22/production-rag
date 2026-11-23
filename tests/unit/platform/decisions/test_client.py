import pytest
from src.platform.decisions import DecisionsClient, NormalizedDecisionResponse


def test_require_noul_reads_probability() -> None:
    response = NormalizedDecisionResponse(model="gpt-test", nouls={"relevant": 0.75})
    assert DecisionsClient.require_noul(response, "relevant") == 0.75


def test_require_noul_raises_on_refusal() -> None:
    response = NormalizedDecisionResponse(model="gpt-test", refusals=frozenset({"relevant"}))
    with pytest.raises(ValueError, match="refused"):
        DecisionsClient.require_noul(response, "relevant")


def test_require_choice_validates_membership() -> None:
    from src.platform.decisions.response import NormalizedChoiceAnswer

    response = NormalizedDecisionResponse(
        model="gpt-test",
        choices={"verdict": NormalizedChoiceAnswer(choice="supported", confidence=0.91)},
    )
    choice, confidence = DecisionsClient.require_choice(response, "verdict", {"supported"})
    assert choice == "supported"
    assert confidence == 0.91
