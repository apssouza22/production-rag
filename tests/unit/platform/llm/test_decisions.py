import importlib.metadata

from langchain_openai.decisions import OpenAIDecisions
from src.config import Settings
from src.platform.decisions import make_openai_decisions, resolve_decisions_model


def test_langchain_openai_version_supports_decisions() -> None:
    version = importlib.metadata.version("langchain-openai")
    major, minor, *_ = (int(part) for part in version.split(".")[:2])
    assert (major, minor) >= (1, 7)


def test_resolve_decisions_model() -> None:
    settings = Settings(agent_model="gpt-5.6-luna", decisions_model="")
    assert resolve_decisions_model(settings) == "gpt-5.6-luna"

    settings_override = Settings(agent_model="gpt-5.6-luna", decisions_model="gpt-6-luna")
    assert resolve_decisions_model(settings_override) == "gpt-6-luna"

    assert resolve_decisions_model(settings, model="openai/gpt-4o-mini") == "gpt-4o-mini"


def test_make_openai_decisions_applies_settings() -> None:
    settings = Settings(
        openai_api_key="test-key",
        openai_base_url="https://gateway.example.com/v1/",
        llm_request_timeout=120,
        agent_model="gpt-6-luna",
    )
    runnable = make_openai_decisions(settings)
    assert isinstance(runnable, OpenAIDecisions)
    assert runnable.model == "gpt-6-luna"
    assert runnable.base_url == "https://gateway.example.com/v1"
    assert runnable.timeout == 120.0
    assert runnable.api_key is not None
    assert runnable.api_key.get_secret_value() == "test-key"


def test_make_openai_decisions_normalizes_model_prefix() -> None:
    settings = Settings(openai_api_key="k", agent_model="gpt-6-luna")
    runnable = make_openai_decisions(settings, model="openai/gpt-6-luna")
    assert runnable.model == "gpt-6-luna"


def test_make_openai_decisions_uses_default_base_url_when_unset() -> None:
    settings = Settings(openai_api_key="k", openai_base_url=None, agent_model="gpt-6-luna")
    runnable = make_openai_decisions(settings)
    assert runnable.base_url == "https://api.openai.com/v1"
