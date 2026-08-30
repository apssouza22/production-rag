from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Settings
from src.domain.ollama.client import OllamaClient


def test_build_fallbacks_excludes_primary_model():
    client = OllamaClient(
        Settings(
            ollama_fallback_models="llama3.2:3b,llama3.2:1b",
        )
    )

    assert client._build_fallbacks("llama3.2:1b") == ["llama3.2:3b"]


def test_build_fallbacks_returns_empty_when_disabled():
    client = OllamaClient(Settings(ollama_fallback_models=""))

    assert client._build_fallbacks("llama3.2:1b") == []


def test_build_model_chain_includes_primary_and_fallbacks():
    client = OllamaClient(
        Settings(
            ollama_fallback_models="llama3.2:3b,llama3.2:1b",
        )
    )

    assert client._build_model_chain("llama3.2:1b") == ["llama3.2:1b", "llama3.2:3b"]


def test_get_langchain_model_uses_with_fallbacks_when_configured():
    client = OllamaClient(
        Settings(
            ollama_fallback_models="llama3.2:3b",
        )
    )
    primary = MagicMock()
    fallback = MagicMock()
    primary.with_fallbacks.return_value = MagicMock(name="runnable_with_fallbacks")

    with patch.object(client, "_create_chat_model", side_effect=[primary, fallback]) as create_chat_model:
        result = client.get_langchain_model(model="llama3.2:1b", temperature=0.2)

    assert result is primary.with_fallbacks.return_value
    create_chat_model.assert_any_call(model="llama3.2:1b", temperature=0.2)
    create_chat_model.assert_any_call(model="llama3.2:3b", temperature=0.2)
    primary.with_fallbacks.assert_called_once_with([fallback])


def test_get_langchain_model_returns_primary_when_no_fallbacks():
    client = OllamaClient(Settings(ollama_fallback_models=""))
    primary = MagicMock()

    with patch.object(client, "_create_chat_model", return_value=primary) as create_chat_model:
        result = client.get_langchain_model(model="llama3.2:1b")

    assert result is primary
    create_chat_model.assert_called_once_with(model="llama3.2:1b", temperature=0.7)


@pytest.mark.asyncio
async def test_generate_tries_configured_fallback_models():
    client = OllamaClient(
        Settings(
            ollama_fallback_models="llama3.2:3b,llama3.2:1b",
        )
    )
    primary_result = {"response": "fallback answer", "usage_metadata": {}}

    with patch.object(
        client,
        "_generate_with_model",
        side_effect=[RuntimeError("primary failed"), primary_result],
    ) as generate_with_model:
        result = await client.generate(model="llama3.2:1b", prompt="hello")

    assert result == primary_result
    assert generate_with_model.call_count == 2
    assert generate_with_model.call_args_list[0].args[1] == "llama3.2:1b"
    assert generate_with_model.call_args_list[1].args[1] == "llama3.2:3b"
