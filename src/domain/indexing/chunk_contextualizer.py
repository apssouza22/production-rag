import asyncio
import logging
from typing import List, Optional

from src.domain.llm.protocol import LLMClient

from .models import TextChunk
from .prompts import CHUNK_CONTEXT_PROMPT

logger = logging.getLogger(__name__)


class ChunkContextualizer:
    """Generate LLM context for each chunk to improve retrieval (Contextual Retrieval).

    For every chunk, an LLM produces a short situating context based on the full
    document. The context is prepended to the chunk text for embedding and BM25
    indexing, while the original chunk text is preserved for answer generation.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str,
        max_document_chars: int = 50_000,
        max_concurrent_requests: int = 3,
        temperature: float = 0.0,
    ):
        self.llm_client = llm_client
        self.model = model
        self.max_document_chars = max_document_chars
        self.max_concurrent_requests = max_concurrent_requests
        self.temperature = temperature

        logger.info(
            "Chunk contextualizer initialized: model=%s, max_document_chars=%d, max_concurrent=%d",
            model,
            max_document_chars,
            max_concurrent_requests,
        )

    async def contextualize_chunks(
        self,
        chunks: List[TextChunk],
        document: str,
        title: str = "",
        abstract: str = "",
    ) -> List[TextChunk]:
        """Generate context for each chunk and attach it to chunk metadata.

        :param chunks: Chunks produced by TextChunker
        :param document: Full document text (typically raw_text from PDF)
        :param title: Paper title for document header
        :param abstract: Paper abstract for document header
        :returns: Same chunks with chunk_context populated
        """
        if not chunks:
            return chunks

        prepared_document = self._prepare_document(document, title, abstract)
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def process_chunk(chunk: TextChunk) -> TextChunk:
            async with semaphore:
                try:
                    context = await self._generate_context(prepared_document, chunk.text)
                    chunk.metadata.chunk_context = context
                except Exception as e:
                    logger.warning(
                        "Context generation failed for %s chunk %d: %s",
                        chunk.arxiv_id,
                        chunk.metadata.chunk_index,
                        e,
                    )
                    chunk.metadata.chunk_context = None
                return chunk

        results = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])
        contextualized_count = sum(1 for chunk in results if chunk.metadata.chunk_context)
        logger.info(
            "Contextualized %d/%d chunks for paper %s",
            contextualized_count,
            len(chunks),
            chunks[0].arxiv_id if chunks else "unknown",
        )
        return list(results)

    def _prepare_document(self, document: str, title: str, abstract: str) -> str:
        """Build document context for the LLM, truncating if necessary."""
        header_parts = []
        if title:
            header_parts.append(f"Title: {title}")
        if abstract:
            header_parts.append(f"Abstract: {abstract}")
        header = "\n\n".join(header_parts)
        suffix = "\n\n[Document truncated for context generation]"
        full_document = f"{header}\n\n{document}" if header else document

        if len(full_document) <= self.max_document_chars:
            return full_document

        if header:
            overhead = len(header) + len(suffix) + 2  # newlines between header and body
            available = max(self.max_document_chars - overhead, 0)
            truncated_body = document[:available]
            return f"{header}\n\n{truncated_body}{suffix}"

        if len(document) > self.max_document_chars:
            return document[: self.max_document_chars] + suffix
        return document

    async def _generate_context(self, document: str, chunk: str) -> str:
        """Call the LLM to generate situating context for a single chunk."""
        prompt = CHUNK_CONTEXT_PROMPT.format(document=document, chunk=chunk)
        result = await self.llm_client.generate(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
        )
        context = self._extract_response(result)
        if not context:
            raise ValueError("LLM returned empty context")
        return context

    @staticmethod
    def _extract_response(result: Optional[dict]) -> str:
        if not result:
            return ""
        return str(result.get("response", "")).strip()
