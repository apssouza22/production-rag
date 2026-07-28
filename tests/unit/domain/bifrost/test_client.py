from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.config import Settings
from src.domain.bifrost.client import BifrostClient


def test_normalize_model_routes_openai_models():
    assert BifrostClient._normalize_model("gpt-4o-mini") == "openai/gpt-4o-mini"
    assert BifrostClient._normalize_model("openai/gpt-4o-mini") == "openai/gpt-4o-mini"


def test_normalize_model_routes_ollama_models():
    assert BifrostClient._normalize_model("llama3.2:1b") == "ollama/llama3.2:1b"
    assert BifrostClient._normalize_model("ollama/llama3.2:1b") == "ollama/llama3.2:1b"


def test_build_fallbacks_excludes_primary_model():
    client = BifrostClient(
        Settings(
            bifrost_fallback_models="openai/gpt-4o-mini,ollama/llama3.2:1b",
        )
    )

    assert client._build_fallbacks("llama3.2:1b") == ["openai/gpt-4o-mini"]
    assert client._build_fallbacks("gpt-4o-mini") == ["ollama/llama3.2:1b"]


def test_build_fallbacks_returns_empty_when_disabled():
    client = BifrostClient(Settings(bifrost_fallback_models=""))

    assert client._build_fallbacks("llama3.2:1b") == []


def test_build_model_chain_includes_primary_and_fallbacks():
    client = BifrostClient(
        Settings(
            bifrost_fallback_models="openai/gpt-4o-mini,ollama/llama3.2:1b",
        )
    )

    assert client._build_model_chain("llama3.2:1b") == [
        "llama3.2:1b",
        "openai/gpt-4o-mini",
    ]


def test_create_chat_model_does_not_pass_fallbacks_to_openai_sdk():
    client = BifrostClient(
        Settings(
            bifrost_fallback_models="openai/gpt-4o-mini,ollama/llama3.2:1b",
        )
    )

    llm = client._create_chat_model(model="llama3.2:1b")

    assert "fallbacks" not in llm.model_kwargs


@pytest.mark.asyncio
async def test_generate_tries_configured_fallback_models():
    client = BifrostClient(
        Settings(
            bifrost_fallback_models="openai/gpt-4o-mini,ollama/llama3.2:1b",
        )
    )
    primary_llm = MagicMock()
    primary_llm.ainvoke = AsyncMock(side_effect=RuntimeError("primary failed"))
    fallback_llm = MagicMock()
    fallback_llm.ainvoke = AsyncMock(return_value=AIMessage(content="fallback answer"))

    with patch.object(
        client,
        "_create_chat_model",
        side_effect=[primary_llm, fallback_llm],
    ) as create_chat_model:
        result = await client.generate(model="llama3.2:1b", prompt="hello")

    assert result == {"response": "fallback answer", "usage_metadata": {}}
    assert create_chat_model.call_count == 2
    create_chat_model.assert_any_call(
        model="llama3.2:1b",
        temperature=0.7,
        top_p=0.9,
        response_format=None,
    )
    create_chat_model.assert_any_call(
        model="openai/gpt-4o-mini",
        temperature=0.7,
        top_p=0.9,
        response_format=None,
    )
