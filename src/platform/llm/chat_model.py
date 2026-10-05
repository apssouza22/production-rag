from typing import Any, Dict, List, Optional

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config import Settings
from src.platform.llm.exceptions import LLMConnectionError, LLMException, LLMTimeoutError
from src.platform.llm.fallback import build_fallback_models

_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def normalize_model(model: str) -> str:
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def supports_reasoning_effort(model: str) -> bool:
    return normalize_model(model).startswith(_REASONING_MODEL_PREFIXES)


def _build_fallbacks(settings: Settings, model: str) -> List[str]:
    return [
        normalize_model(fallback)
        for fallback in build_fallback_models(model, settings.llm_fallback_models, normalize_model)
    ]


def _create_chat_openai(
    settings: Settings,
    model: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    response_format: Optional[Dict[str, Any]] = None,
    api_key: str | None = None,
) -> ChatOpenAI:
    model_kwargs: Dict[str, Any] = {}
    if response_format is not None:
        model_kwargs["response_format"] = response_format

    normalized_model = normalize_model(model)
    base_url = (settings.openai_base_url or _DEFAULT_BASE_URL).rstrip("/")
    chat_kwargs: Dict[str, Any] = {
        "model": normalized_model,
        "base_url": base_url,
        "api_key": api_key or settings.openai_api_key,
        "timeout": float(settings.llm_request_timeout),
        "model_kwargs": model_kwargs,
    }
    if supports_reasoning_effort(model):
        chat_kwargs["reasoning_effort"] = settings.reasoning_effort
        if settings.reasoning_effort == "none":
            chat_kwargs["temperature"] = temperature
            chat_kwargs["top_p"] = top_p
    else:
        chat_kwargs["temperature"] = temperature
        chat_kwargs["top_p"] = top_p

    return ChatOpenAI(**chat_kwargs)


def make_chat_model(
    settings: Settings,
    model: str,
    temperature: float = 0.7,
    *,
    api_key: str | None = None,
) -> BaseChatModel:
    """Create a LangChain chat model for OpenAI with optional fallback models."""
    primary = _create_chat_openai(settings, model, temperature=temperature, api_key=api_key)
    fallbacks = _build_fallbacks(settings, model)
    if not fallbacks:
        return primary

    fallback_models = [
        _create_chat_openai(settings, fallback, temperature=temperature, api_key=api_key)
        for fallback in fallbacks
    ]
    return primary.with_fallbacks(fallback_models)


async def check_openai_health(settings: Settings) -> Dict[str, str]:
    api_key = settings.openai_api_key
    if not api_key:
        return {"status": "unhealthy", "message": "OPENAI_API_KEY is not configured"}

    base_url = (settings.openai_base_url or _DEFAULT_BASE_URL).rstrip("/")
    timeout = float(settings.llm_request_timeout)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            response = await client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )

            if response.status_code == 200:
                return {"status": "healthy", "message": "OpenAI API is reachable"}

            raise LLMException(f"OpenAI returned status {response.status_code}")

    except httpx.ConnectError as e:
        raise LLMConnectionError(f"Cannot connect to OpenAI API: {e}") from e
    except httpx.TimeoutException as e:
        raise LLMTimeoutError(f"OpenAI API timeout: {e}") from e
    except LLMException:
        raise
    except Exception as e:
        raise LLMException(f"OpenAI health check failed: {str(e)}") from e
