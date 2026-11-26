"""Normalize provider decision responses into :class:`~src.platform.decisions.response.NormalizedDecisionResponse`."""

from langchain_core.decisions import ClassifierResponse
from langchain_openai.decisions import DecisionResponse, RefusalAnswer

from ..response import NormalizedChoiceAnswer, NormalizedDecisionResponse


def from_openai_decision_response(response: DecisionResponse) -> NormalizedDecisionResponse:
    """Map an OpenAI Decisions response to the portable normalized structure."""
    nouls = {name: float(answer.probability) for name, answer in response.predicates.items()}
    choices = {
        name: NormalizedChoiceAnswer(choice=str(answer.choice), confidence=float(answer.confidence))
        for name, answer in response.choices.items()
    }
    refusals = frozenset(name for name, answer in response.answers.items() if isinstance(answer, RefusalAnswer))
    return NormalizedDecisionResponse(
        model=response.model,
        nouls=nouls,
        choices=choices,
        refusals=refusals,
    )


def from_classifier_response(response: ClassifierResponse) -> NormalizedDecisionResponse:
    """Map a :class:`langchain_core.decisions.ClassifierResponse` (TypeSafe/Jev) to the portable shape."""
    nouls = {name: float(answer.noul) for name, answer in response.nouls.items()}
    choices = {
        name: NormalizedChoiceAnswer(choice=str(answer.choice), confidence=float(answer.confidence))
        for name, answer in response.choices.items()
    }
    return NormalizedDecisionResponse(
        model=response.model,
        nouls=nouls,
        choices=choices,
        refusals=frozenset(),
    )
