from langchain_core.decisions import Choice, ClassifierResponse, Noul
from langchain_core.decisions import NoulAnswer as CoreNoulAnswer
from langchain_openai.decisions import ChoiceAnswer, DecisionResponse, PredicateAnswer, RefusalAnswer
from src.platform.decisions.adapters.questions import to_openai_questions
from src.platform.decisions.adapters.responses import from_classifier_response, from_openai_decision_response
from src.platform.decisions.response import NormalizedDecisionResponse


def test_to_openai_questions_maps_noul_and_choice() -> None:
    questions = {
        "relevant": Noul(instructions="Is this relevant?"),
        "route": Choice(instructions="Pick team", criteria={"billing": "Invoices"}),
    }
    converted = to_openai_questions(questions)
    assert converted["relevant"].type == "predicate"
    assert converted["route"].choices == {"billing": "Invoices"}


def test_from_openai_predicate_answer() -> None:
    response = DecisionResponse(
        model="gpt-test",
        answers={"relevant": PredicateAnswer(type="predicate", probability=0.75)},
    )
    normalized = from_openai_decision_response(response)
    assert normalized.nouls["relevant"] == 0.75
    assert normalized.refusals == frozenset()


def test_from_openai_surfaces_refusals() -> None:
    response = DecisionResponse(
        model="gpt-test",
        answers={"relevant": RefusalAnswer(type="refusal")},
    )
    normalized = from_openai_decision_response(response)
    assert "relevant" in normalized.refusals


def test_from_classifier_response_maps_noul() -> None:
    response = ClassifierResponse(
        model="jev-test",
        answers={"urgent": CoreNoulAnswer(type="noul", noul=0.82)},
    )
    normalized = from_classifier_response(response)
    assert normalized.nouls["urgent"] == 0.82
    assert isinstance(normalized, NormalizedDecisionResponse)


def test_from_openai_maps_choice_answers() -> None:
    response = DecisionResponse(
        model="gpt-test",
        answers={
            "verdict": ChoiceAnswer(
                type="choice",
                choice="supported",
                probabilities={"supported": 0.9},
                confidence=0.91,
            )
        },
    )
    normalized = from_openai_decision_response(response)
    assert normalized.choices["verdict"].choice == "supported"
    assert normalized.choices["verdict"].confidence == 0.91
