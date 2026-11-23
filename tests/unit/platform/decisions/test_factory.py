import pytest
from langchain_openai.decisions import OpenAIDecisions
from src.config import Settings
from src.platform.decisions import DecisionsProviderError, make_decision_backend
from src.platform.decisions.providers.openai import OpenAIDecisionBackend


def test_make_decision_backend_defaults_to_openai() -> None:
    settings = Settings(openai_api_key="k", agent_model="gpt-6-luna", decisions_provider="openai")
    backend = make_decision_backend(settings)
    assert isinstance(backend, OpenAIDecisionBackend)
    assert isinstance(backend._runnable, OpenAIDecisions)


def test_typesafe_provider_requires_package() -> None:
    settings = Settings(decisions_provider="typesafe", typesafe_api_key="k")
    with pytest.raises(DecisionsProviderError, match="langchain-typesafe"):
        make_decision_backend(settings)
