import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from langchain_ollama import ChatOllama

from src.config import Settings
from src.domain.llm.protocol import LlmProviderClient
from src.domain.ollama.exceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError

logger = logging.getLogger(__name__)


class OllamaClient(LlmProviderClient):
    """Client for interacting with Ollama local LLM service."""

    def __init__(self, settings: Settings):
        """Initialize Ollama client with settings."""
        self.base_url = settings.ollama_host
        self.timeout = httpx.Timeout(float(settings.ollama_timeout))

    def get_langchain_model(self, model: str, temperature: float = 0.7) -> ChatOllama:
        """Return a LangChain ChatOllama instance for agent graph nodes."""
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=self.base_url,
        )

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

    async def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate text using specified model."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                data = {"model": model, "prompt": prompt, "stream": stream, **kwargs}

                logger.info("Sending request to Ollama: model=%s, stream=%s, extra_params=%s", model, stream, kwargs)
                response = await client.post(f"{self.base_url}/api/generate", json=data)

                if response.status_code == 200:
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

                raise OllamaException(f"Generation failed: {response.status_code}")

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}") from e
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}") from e
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error generating with Ollama: {e}") from e

    async def generate_stream(self, model: str, prompt: str, **kwargs):
        """Generate text with streaming response."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                data = {"model": model, "prompt": prompt, "stream": True, **kwargs}

                logger.info("Starting streaming generation: model=%s", model)

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

        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}") from e
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}") from e
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error in streaming generation: {e}") from e
