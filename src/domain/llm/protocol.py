from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Common interface for direct Ollama and Bifrost-backed LLM clients."""

    def get_langchain_model(self, model: str, temperature: float = 0.7):
        """Return a LangChain chat model for agent graph nodes."""

    async def health_check(self) -> Dict[str, Any]:
        """Check whether the configured LLM backend is healthy."""

    async def list_models(self) -> List[Dict[str, Any]]:
        """List models available from the configured backend."""

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Generate text from a prompt."""

    async def generate_stream(self, model: str, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """Stream text generation chunks from a prompt."""

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "llama3.2",
        use_structured_output: bool = False,
    ) -> Dict[str, Any]:
        """Generate a RAG answer using retrieved chunks."""

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "llama3.2",
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream a RAG answer using retrieved chunks."""
