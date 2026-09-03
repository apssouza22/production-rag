"""Tests for evals configuration helpers."""

from rag_evals.config import Settings, build_openai_client_kwargs


def test_build_openai_client_kwargs_uses_direct_openai_by_default():
    settings = Settings(
        evaluation_api_key="test-openai-key",
        evaluation_base_url="https://api.openai.com/v1",
        bifrost_enabled=False,
        llm_provider="openai",
    )

    kwargs = build_openai_client_kwargs(settings)

    assert kwargs["api_key"] == "test-openai-key"
    assert kwargs["base_url"] == "https://api.openai.com/v1"
    assert "default_headers" not in kwargs


def test_build_openai_client_kwargs_routes_through_bifrost_with_virtual_key_header():
    settings = Settings(
        bifrost_enabled=True,
        bifrost_host="http://localhost:8090",
        bifrost_api_key="sk-bf-agent-1-dev",
    )

    kwargs = build_openai_client_kwargs(settings)

    assert kwargs["api_key"] == "sk-bf-agent-1-dev"
    assert kwargs["base_url"] == "http://localhost:8090/v1"
    assert kwargs["default_headers"] == {"x-bf-vk": "sk-bf-agent-1-dev"}


def test_build_openai_client_kwargs_uses_bifrost_when_llm_provider_is_bifrost():
    settings = Settings(
        llm_provider="bifrost",
        bifrost_host="http://bifrost:8080",
        bifrost_api_key="sk-bf-agent-2-dev",
    )

    kwargs = build_openai_client_kwargs(settings)

    assert kwargs["base_url"] == "http://bifrost:8080/v1"
    assert kwargs["default_headers"]["x-bf-vk"] == "sk-bf-agent-2-dev"
