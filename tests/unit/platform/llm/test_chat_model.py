from unittest.mock import MagicMock, patch

from src.config import Settings
from src.platform.llm.chat_model import make_chat_model, normalize_model, supports_reasoning_effort


def test_normalize_model_strips_openai_prefix():
    assert normalize_model("gpt-4o-mini") == "gpt-4o-mini"
    assert normalize_model("openai/gpt-4o-mini") == "gpt-4o-mini"
    assert normalize_model("gpt-5.6-luna") == "gpt-5.6-luna"


def test_supports_reasoning_effort_for_openai_reasoning_models():
    assert supports_reasoning_effort("gpt-5.6-luna") is True
    assert supports_reasoning_effort("openai/gpt-5.6-luna") is True
    assert supports_reasoning_effort("openai/gpt-4o-mini") is False


def test_make_chat_model_omits_reasoning_effort_for_non_reasoning_models():
    settings = Settings(reasoning_effort="low")
    llm = make_chat_model(settings, "gpt-4o-mini")

    assert llm.reasoning_effort is None


def test_make_chat_model_passes_reasoning_effort_for_reasoning_models():
    settings = Settings(reasoning_effort="low")
    llm = make_chat_model(settings, "gpt-5.6-luna")

    assert llm.reasoning_effort == "low"
    assert llm.top_p is None
    assert llm.temperature is None


def test_make_chat_model_uses_with_fallbacks_when_configured():
    settings = Settings(llm_fallback_models="gpt-4o-mini,gpt-4o")
    primary = MagicMock()
    fallback = MagicMock()
    primary.with_fallbacks.return_value = MagicMock(name="runnable_with_fallbacks")

    with patch("src.platform.llm.chat_model._create_chat_openai", side_effect=[primary, fallback]) as create_chat:
        result = make_chat_model(settings, "gpt-4o-mini", temperature=0.2)

    assert result is primary.with_fallbacks.return_value
    assert create_chat.call_count == 2
    primary.with_fallbacks.assert_called_once_with([fallback])


def test_make_chat_model_returns_primary_when_no_fallbacks():
    settings = Settings(llm_fallback_models="")
    primary = MagicMock()

    with patch("src.platform.llm.chat_model._create_chat_openai", return_value=primary) as create_chat:
        result = make_chat_model(settings, "gpt-4o-mini")

    assert result is primary
    create_chat.assert_called_once()

