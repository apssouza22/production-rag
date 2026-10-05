"""Tests for evals configuration helpers."""

from evals.custom_evals.config import Settings, build_openai_client_kwargs


def test_build_openai_client_kwargs_uses_direct_openai_by_default():
    settings = Settings(
        evaluation_api_key="test-openai-key",
        evaluation_base_url="https://api.openai.com/v1",
    )

    kwargs = build_openai_client_kwargs(settings)

    assert kwargs["api_key"] == "test-openai-key"
    assert kwargs["base_url"] == "https://api.openai.com/v1"
    assert "default_headers" not in kwargs
