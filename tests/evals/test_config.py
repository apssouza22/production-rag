"""Tests for evals configuration helpers."""

from unittest.mock import Mock, patch

from evals.core.llm import make_eval_chat_model


def test_make_eval_chat_model_configures_langchain_openai_model() -> None:
    with patch("evals.core.llm.ChatOpenAI", return_value=Mock()) as chat_openai:
        make_eval_chat_model(
            model="judge-model",
            api_key="test-openai-key",
            base_url="https://api.openai.com/v1",
        )

    chat_openai.assert_called_once_with(
        model="judge-model",
        api_key="test-openai-key",
        base_url="https://api.openai.com/v1",
    )
