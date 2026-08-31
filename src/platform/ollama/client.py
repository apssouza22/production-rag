import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from langchain_ollama import ChatOllama

from src.config import Settings
from src.platform.llm.fallback import build_fallback_models, build_model_chain
from src.platform.llm.protocol import LlmProviderClient
from src.platform.ollama.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError

logger = logging.getLogger(__name__)


class OllamaClient(LlmProviderClient):
    """Client for interacting with Ollama local LLM service."""

    def __init__(self, settings: Settings):
        """Initialize Ollama client with settings."""
        self.base_url = settings.ollama_host
        self.fallback_models = settings.ollama_fallback_models
        self.timeout = httpx.Timeout(float(settings.ollama_timeout))

    def _build_fallbacks(self, model: str) -> List[str]:
        """Build fallback model names, excluding the primary model."""
        return build_fallback_models(model, self.fallback_models)

    def _build_model_chain(self, model: str) -> List[str]:
        """Return the primary model followed by configured fallbacks."""
        return build_model_chain(model, self.fallback_models)

    def _create_chat_model(self, model: str, temperature: float = 0.7) -> ChatOllama:
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=self.base_url,
        )

    def get_langchain_model(self, model: str, temperature: float = 0.7) -> ChatOllama:
        """Return a LangChain ChatOllama instance with optional configured fallbacks."""
        primary = self._create_chat_model(model=model, temperature=temperature)
        fallbacks = self._build_fallbacks(model)
        if not fallbacks:
            return primary

        fallback_models = [
            self._create_chat_model(model=fallback, temperature=temperature) for fallback in fallbacks
        ]
        return primary.with_fallbacks(fallback_models)

    async def health_check(self) -> Dict[str, Any]:
        """Check if Ollama service is healthy and responding."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/version")

                if response.status_code == 200:
                    version_data = response.json()
                    return {
                        "status": "healthy",
                        "message": "Ollama service is running",
                        "version": version_data.get("version", "unknown"),
                    }

                raise OllamaException(f"Ollama returned status {response.status_code}")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}") from e
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}") from e
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Ollama health check failed: {str(e)}") from e

    async def list_models(self) -> List[Dict[str, Any]]:
        """Get list of available models."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")

                if response.status_code == 200:
                    data = response.json()
                    return data.get("models", [])

                raise OllamaException(f"Failed to list models: {response.status_code}")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}") from e
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}") from e
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error listing models: {e}") from e

    async def _generate_with_model(
        self,
        client: httpx.AsyncClient,
        model: str,
        prompt: str,
        stream: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        data = {"model": model, "prompt": prompt, "stream": stream, **kwargs}

        logger.info(
            "Sending request to Ollama: model=%s, stream=%s, extra_params=%s",
            model,
            stream,
            kwargs,
        )
        response = await client.post(f"{self.base_url}/api/generate", json=data)

        if response.status_code != 200:
            raise OllamaException(f"Generation failed: {response.status_code}")

        result = response.json()

        usage_metadata: Dict[str, Any] = {}
        if "prompt_eval_count" in result:
            usage_metadata["prompt_tokens"] = result.get("prompt_eval_count", 0)
        if "eval_count" in result:
            usage_metadata["completion_tokens"] = result.get("eval_count", 0)

        if usage_metadata:
            usage_metadata["total_tokens"] = (
                usage_metadata.get("prompt_tokens", 0) + usage_metadata.get("completion_tokens", 0)
            )

        if "total_duration" in result:
            usage_metadata["latency_ms"] = round(result["total_duration"] / 1_000_000, 2)

        if "prompt_eval_duration" in result:
            usage_metadata["prompt_eval_duration_ms"] = round(result["prompt_eval_duration"] / 1_000_000, 2)
        if "eval_duration" in result:
            usage_metadata["eval_duration_ms"] = round(result["eval_duration"] / 1_000_000, 2)

        result["usage_metadata"] = usage_metadata
        logger.debug("Usage metadata: %s", usage_metadata)

        return result

    async def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate text using the primary model, then configured fallbacks on failure."""
        if stream:
            raise OllamaException("Use generate_stream() for streaming responses")

        model_chain = self._build_model_chain(model)
        last_error: Exception | None = None

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for index, candidate in enumerate(model_chain):
                    try:
                        return await self._generate_with_model(
                            client,
                            candidate,
                            prompt,
                            stream=False,
                            **kwargs,
                        )
                    except httpx.ConnectError as e:
                        raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}") from e
                    except httpx.TimeoutException as e:
                        raise OllamaTimeoutError(f"Ollama service timeout: {e}") from e
                    except OllamaException:
                        raise
                    except Exception as e:
                        last_error = e
                        if index < len(model_chain) - 1:
                            logger.warning(
                                "Ollama model %s failed (%s); trying fallback model %s",
                                candidate,
                                e,
                                model_chain[index + 1],
                            )
                            continue
                        break
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}") from e
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}") from e
        except OllamaException:
            raise

        raise OllamaException(f"Error generating with Ollama: {last_error}") from last_error

    async def generate_stream(self, model: str, prompt: str, **kwargs):
        """Generate text with streaming response, falling back to configured models on failure."""
        model_chain = self._build_model_chain(model)
        last_error: Exception | None = None

        for index, candidate in enumerate(model_chain):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    data = {"model": candidate, "prompt": prompt, "stream": True, **kwargs}

                    logger.info("Starting streaming generation: model=%s", candidate)

                    async with client.stream("POST", f"{self.base_url}/api/generate", json=data) as response:
                        if response.status_code != 200:
                            raise OllamaException(f"Streaming generation failed: {response.status_code}")

                        async for line in response.aiter_lines():
                            if line.strip():
                                try:
                                    chunk = json.loads(line)
                                    yield chunk
                                except json.JSONDecodeError:
                                    logger.warning("Failed to parse streaming chunk: %s", line)
                                    continue
                return
            except httpx.ConnectError as e:
                raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}") from e
            except httpx.TimeoutException as e:
                raise OllamaTimeoutError(f"Ollama service timeout: {e}") from e
            except OllamaException:
                raise
            except Exception as e:
                last_error = e
                if index < len(model_chain) - 1:
                    logger.warning(
                        "Ollama streaming model %s failed (%s); trying fallback model %s",
                        candidate,
                        e,
                        model_chain[index + 1],
                    )
                    continue
                break

        raise OllamaException(f"Error in streaming generation: {last_error}") from last_error
