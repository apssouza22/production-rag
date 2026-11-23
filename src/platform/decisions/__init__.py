"""Multi-provider decision-model integration."""

from src.platform.decisions.client import DecisionsClient
from src.platform.decisions.evaluator import DecisionEvaluator
from src.platform.decisions.exceptions import DecisionsProviderError
from src.platform.decisions.factory import make_decision_backend, make_openai_decisions, resolve_decisions_model
from src.platform.decisions.response import NormalizedDecisionResponse

__all__ = [
    "DecisionsClient",
    "DecisionsProviderError",
    "NormalizedDecisionResponse",
    "DecisionEvaluator",
    "make_decision_backend",
    "make_openai_decisions",
    "resolve_decisions_model",
]
