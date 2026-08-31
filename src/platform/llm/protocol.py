from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional


class LlmProviderClient(ABC):
    """Low-level interface for direct Ollama and Bifrost-backed LLM backends."""

    @abstractmethod
    def get_langchain_model(self, model: str, temperature: float = 0.7):
        """Return a LangChain chat model for agent graph nodes."""

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check whether the configured LLM backend is healthy."""

    @abstractmethod
    async def list_models(self) -> List[Dict[str, Any]]:
        """List models available from the configured backend."""

    @abstractmethod
    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Generate text from a prompt."""

    @abstractmethod
    async def generate_stream(self, model: str, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Stream text generation chunks from a prompt."""
