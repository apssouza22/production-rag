import logging
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.config import Settings
from src.domain.bifrost.exceptions import BifrostConnectionError, BifrostException, BifrostTimeoutError
from src.domain.llm.rag import RAGGenerationMixin
from src.domain.ollama.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)


class BifrostClient(RAGGenerationMixin):
    """Client for LLM requests routed through the Bifrost gateway."""

    def __init__(self, settings: Settings, api_key: str | None = None):
        self.bifrost_host = settings.bifrost_host.rstrip("/")
        self.api_key = api_key or settings.bifrost_api_key
        self.fallback_models = settings.bifrost_fallback_models
        self.timeout = float(settings.ollama_timeout)
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()

    @property
    def langchain_base_url(self) -> str:
        return f"{self.bifrost_host}/langchain"

    @staticmethod
    def _normalize_model(model: str) -> str:
        if "/" in model:
            return model

        if model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
            return f"openai/{model}"

        return f"ollama/{model}"

    def _build_fallbacks(self, model: str) -> List[str]:
        """Build the Bifrost fallback chain, excluding the primary model."""
        if not self.fallback_models.strip():
            return []

        normalized_primary = self._normalize_model(model)
        fallbacks: List[str] = []
        for fallback in self.fallback_models.split(","):
            candidate = fallback.strip()
            if not candidate:
                continue

            normalized = self._normalize_model(candidate)
            if normalized != normalized_primary and normalized not in fallbacks:
                fallbacks.append(normalized)

        return fallbacks

    def _build_model_chain(self, model: str) -> List[str]:
        """Return the primary model followed by configured fallbacks."""
        return [model, *self._build_fallbacks(model)]

    @staticmethod
    def _message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(part) for part in content)
        return str(content)

    def _create_chat_model(
        self,
        model: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ChatOpenAI:
        model_kwargs: Dict[str, Any] = {}
        if response_format is not None:
            model_kwargs["response_format"] = response_format

        return ChatOpenAI(
            model=self._normalize_model(model),
            base_url=self.langchain_base_url,
            api_key=self.api_key,
            temperature=temperature,
            top_p=top_p,
            timeout=self.timeout,
            model_kwargs=model_kwargs,
        )

    def get_langchain_model(self, model: str, temperature: float = 0.7) -> ChatOpenAI:
        """Return a LangChain ChatOpenAI instance configured for Bifrost."""
        return self._create_chat_model(model=model, temperature=temperature)

    async def health_check(self) -> Dict[str, Any]:
        """Check whether the Bifrost gateway is healthy."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
                response = await client.get(f"{self.bifrost_host}/health")

                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "message": "Bifrost gateway is running",
                    }

                raise BifrostException(f"Bifrost returned status {response.status_code}")

        except httpx.ConnectError as e:
            raise BifrostConnectionError(f"Cannot connect to Bifrost gateway: {e}") from e
        except httpx.TimeoutException as e:
            raise BifrostTimeoutError(f"Bifrost gateway timeout: {e}") from e
        except BifrostException:
            raise
        except Exception as e:
            raise BifrostException(f"Bifrost health check failed: {str(e)}") from e

    async def list_models(self) -> List[Dict[str, Any]]:
        """List models exposed by Bifrost."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
                response = await client.get(f"{self.bifrost_host}/v1/models")

                if response.status_code == 200:
                    data = response.json()
                    return [{"name": model["id"], "model": model["id"]} for model in data.get("data", [])]

                raise BifrostException(f"Failed to list models: {response.status_code}")

        except httpx.ConnectError as e:
            raise BifrostConnectionError(f"Cannot connect to Bifrost gateway: {e}") from e
        except httpx.TimeoutException as e:
            raise BifrostTimeoutError(f"Bifrost gateway timeout: {e}") from e
        except BifrostException:
            raise
        except Exception as e:
            raise BifrostException(f"Error listing models: {e}") from e

    async def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate text through Bifrost's LangChain-compatible endpoint."""
        if stream:
            raise BifrostException("Use generate_stream() for streaming responses")

        response_format = {"type": "json_object"} if kwargs.get("format") else None
        extra_log_params = {
            key: value for key, value in kwargs.items() if key not in {"temperature", "top_p", "format"}
        }
        model_chain = self._build_model_chain(model)
        last_error: Exception | None = None

        for index, candidate in enumerate(model_chain):
            try:
                llm = self._create_chat_model(
                    model=candidate,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                    response_format=response_format,
                )

                logger.info(
                    "Sending request to Bifrost: model=%s, response_format=%s, extra_params=%s",
                    candidate,
                    response_format is not None,
                    extra_log_params,
                )
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                answer_text = self._message_content(response.content)

                usage_metadata: Dict[str, Any] = {}
                token_usage = getattr(response, "response_metadata", {}).get("token_usage", {})
                if token_usage:
                    usage_metadata["prompt_tokens"] = token_usage.get("prompt_tokens", 0)
                    usage_metadata["completion_tokens"] = token_usage.get("completion_tokens", 0)
                    usage_metadata["total_tokens"] = token_usage.get("total_tokens", 0)

                return {
                    "response": answer_text,
                    "usage_metadata": usage_metadata,
                }

            except httpx.ConnectError as e:
                raise BifrostConnectionError(f"Cannot connect to Bifrost gateway: {e}") from e
            except httpx.TimeoutException as e:
                raise BifrostTimeoutError(f"Bifrost gateway timeout: {e}") from e
            except BifrostException:
                raise
            except Exception as e:
                last_error = e
                if index < len(model_chain) - 1:
                    logger.warning(
                        "Bifrost model %s failed (%s); trying fallback model %s",
                        candidate,
                        e,
                        model_chain[index + 1],
                    )
                    continue
                break

        raise BifrostException(f"Error generating with Bifrost: {last_error}") from last_error

    async def generate_stream(self, model: str, prompt: str, **kwargs):
        """Generate text with streaming response through Bifrost."""
        model_chain = self._build_model_chain(model)
        last_error: Exception | None = None

        for index, candidate in enumerate(model_chain):
            try:
                llm = self._create_chat_model(
                    model=candidate,
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.9),
                )

                logger.info("Starting streaming generation through Bifrost: model=%s", candidate)

                async for chunk in llm.astream([HumanMessage(content=prompt)]):
                    content = self._message_content(chunk.content)
                    if content:
                        yield {"response": content, "done": False}

                yield {"response": "", "done": True}
                return

            except httpx.ConnectError as e:
                raise BifrostConnectionError(f"Cannot connect to Bifrost gateway: {e}") from e
            except httpx.TimeoutException as e:
                raise BifrostTimeoutError(f"Bifrost gateway timeout: {e}") from e
            except BifrostException:
                raise
            except Exception as e:
                last_error = e
                if index < len(model_chain) - 1:
                    logger.warning(
                        "Bifrost streaming model %s failed (%s); trying fallback model %s",
                        candidate,
                        e,
                        model_chain[index + 1],
                    )
                    continue
                break

        raise BifrostException(f"Error in streaming generation: {last_error}") from last_error
