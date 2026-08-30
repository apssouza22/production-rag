import logging
from typing import Any, AsyncIterator, Dict, List

from src.domain.llm.exceptions import LLMException
from src.domain.llm.protocol import RagService, LlmProviderClient
from src.domain.ollama.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)


class RagServiceSimple(RagService):
    """LLM client that delegates generation to a provider and adds RAG helpers."""

    def __init__(self, provider: LlmProviderClient):
        self._provider = provider
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()

    @property
    def provider(self) -> LlmProviderClient:
        return self._provider

    def get_langchain_model(self, model: str, temperature: float = 0.7):
        return self._provider.get_langchain_model(model, temperature)

    async def health_check(self) -> Dict[str, Any]:
        return await self._provider.health_check()

    async def list_models(self) -> List[Dict[str, Any]]:
        return await self._provider.list_models()

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any] | None:
        return await self._provider.generate(model, prompt, stream=stream, **kwargs)

    async def generate_stream(self, model: str, prompt: str, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        async for chunk in self._provider.generate_stream(model, prompt, **kwargs):
            yield chunk

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "llama3.2",
        use_structured_output: bool = False,
    ) -> Dict[str, Any]:
        """Generate a RAG answer using retrieved chunks."""
        try:
            if use_structured_output:
                prompt_data = self.prompt_builder.create_structured_prompt(query, chunks)
                response = await self.generate(
                    model=model,
                    prompt=prompt_data["prompt"],
                    temperature=0.7,
                    top_p=0.9,
                    format=prompt_data["format"],
                )
            else:
                prompt = self.prompt_builder.create_rag_prompt(query, chunks)
                response = await self.generate(
                    model=model,
                    prompt=prompt,
                    temperature=0.7,
                    top_p=0.9,
                )

            if response and "response" in response:
                answer_text = response["response"]
                logger.debug("Raw LLM response: %s", answer_text[:500])

                if use_structured_output:
                    parsed_response = self.response_parser.parse_structured_response(answer_text)
                    logger.debug("Parsed response: %s", parsed_response)
                    return parsed_response

                sources = []
                seen_urls = set()
                for chunk in chunks:
                    arxiv_id = chunk.get("arxiv_id")
                    if arxiv_id:
                        arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                        pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                        if pdf_url not in seen_urls:
                            sources.append(pdf_url)
                            seen_urls.add(pdf_url)

                citations = list(set(chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")))

                return {
                    "answer": answer_text,
                    "sources": sources,
                    "confidence": "medium",
                    "citations": citations[:5],
                }

            raise LLMException("No response generated from LLM backend")

        except LLMException:
            raise
        except Exception as e:
            logger.error("Error generating RAG answer: %s", e)
            raise LLMException(f"Failed to generate RAG answer: {e}") from e

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "llama3.2",
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate a streaming RAG answer using retrieved chunks."""
        try:
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            async for chunk in self.generate_stream(
                model=model,
                prompt=prompt,
                temperature=0.7,
                top_p=0.9,
            ):
                yield chunk

        except LLMException:
            raise
        except Exception as e:
            logger.error("Error generating streaming RAG answer: %s", e)
            raise LLMException(f"Failed to generate streaming RAG answer: {e}") from e
