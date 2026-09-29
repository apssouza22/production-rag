from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from src.config import Settings
from src.platform.bifrost.client import BifrostClient


def test_normalize_model_routes_openai_models():
    assert BifrostClient._normalize_model("gpt-4o-mini") == "openai/gpt-4o-mini"
    assert BifrostClient._normalize_model("openai/gpt-4o-mini") == "openai/gpt-4o-mini"
    assert BifrostClient._normalize_model("gpt-5.6-luna") == "openai/gpt-5.6-luna"


def test_normalize_model_preserves_explicit_provider_prefix():
    assert BifrostClient._normalize_model("anthropic/claude-3-5-sonnet") == "anthropic/claude-3-5-sonnet"


def test_supports_reasoning_effort_for_openai_reasoning_models():
    assert BifrostClient._supports_reasoning_effort("gpt-5.6-luna") is True
    assert BifrostClient._supports_reasoning_effort("openai/gpt-5.6-luna") is True
    assert BifrostClient._supports_reasoning_effort("openai/gpt-4o-mini") is False


def test_build_fallbacks_excludes_primary_model():
    client = BifrostClient(
        Settings(
            bifrost_fallback_models="openai/gpt-4o-mini,openai/gpt-4o",
        )
    )

    assert client._build_fallbacks("gpt-4o-mini") == ["openai/gpt-4o"]
    assert client._build_fallbacks("openai/gpt-4o-mini") == ["openai/gpt-4o"]


def test_build_fallbacks_returns_empty_when_disabled():
    client = BifrostClient(Settings(bifrost_fallback_models=""))

    assert client._build_fallbacks("gpt-4o-mini") == []


def test_build_model_chain_includes_primary_and_fallbacks():
    client = BifrostClient(
        Settings(
            bifrost_fallback_models="openai/gpt-4o-mini,openai/gpt-4o",
        )
    )

    assert client._build_model_chain("gpt-4o-mini") == [
        "gpt-4o-mini",
        "openai/gpt-4o",
    ]


def test_create_chat_model_passes_reasoning_effort_for_reasoning_models():
    client = BifrostClient(Settings(reasoning_effort="low"))

    llm = client._create_chat_model(model="gpt-5.6-luna")

    assert llm.reasoning_effort == "low"
    assert llm.top_p is None
    assert llm.temperature is None


def test_create_chat_model_allows_sampling_params_when_reasoning_disabled():
    client = BifrostClient(Settings(reasoning_effort="none"))

    llm = client._create_chat_model(model="gpt-5.6-luna", temperature=0.2, top_p=0.8)

    assert llm.reasoning_effort == "none"
    assert llm.temperature == 0.2
    assert llm.top_p == 0.8


def test_create_chat_model_omits_reasoning_effort_for_non_reasoning_models():
    client = BifrostClient(Settings(reasoning_effort="low"))

    llm = client._create_chat_model(model="gpt-4o-mini")

    assert llm.reasoning_effort is None


def test_create_chat_model_does_not_pass_fallbacks_to_openai_sdk():
    client = BifrostClient(
        Settings(
            bifrost_fallback_models="openai/gpt-5.6-luna,openai/gpt-4o-mini",
        )
    )

    llm = client._create_chat_model(model="gpt-4o-mini")

    assert "fallbacks" not in llm.model_kwargs


def test_get_langchain_model_uses_with_fallbacks_when_configured():
    client = BifrostClient(
        Settings(
            bifrost_fallback_models="openai/gpt-4o-mini,openai/gpt-4o",
        )
    )
    primary = MagicMock()
    fallback = MagicMock()
    primary.with_fallbacks.return_value = MagicMock(name="runnable_with_fallbacks")

    with patch.object(client, "_create_chat_model", side_effect=[primary, fallback]) as create_chat_model:
        result = client.get_langchain_model(model="gpt-4o-mini", temperature=0.2)

    assert result is primary.with_fallbacks.return_value
    create_chat_model.assert_any_call(model="gpt-4o-mini", temperature=0.2)
    create_chat_model.assert_any_call(model="openai/gpt-4o", temperature=0.2)
    primary.with_fallbacks.assert_called_once_with([fallback])


def test_get_langchain_model_returns_primary_when_no_fallbacks():
    client = BifrostClient(Settings(bifrost_fallback_models=""))
    primary = MagicMock()

    with patch.object(client, "_create_chat_model", return_value=primary) as create_chat_model:
        result = client.get_langchain_model(model="gpt-4o-mini")

    assert result is primary
    create_chat_model.assert_called_once_with(model="gpt-4o-mini", temperature=0.7)


@pytest.mark.asyncio
async def test_generate_tries_configured_fallback_models():
    client = BifrostClient(
        Settings(
            bifrost_fallback_models="openai/gpt-4o-mini,openai/gpt-4o",
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
        result = await client.generate(model="gpt-4o-mini", prompt="hello")

    assert result == {"response": "fallback answer", "usage_metadata": {}}
    assert create_chat_model.call_count == 2
    create_chat_model.assert_any_call(
        model="gpt-4o-mini",
        temperature=0.7,
        top_p=0.9,
        response_format=None,
    )
    create_chat_model.assert_any_call(
        model="openai/gpt-4o",
        temperature=0.7,
        top_p=0.9,
        response_format=None,
    )
