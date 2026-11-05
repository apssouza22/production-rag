import asyncio
import logging
from dataclasses import dataclass
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage

from .models import TextChunk
from .prompts import CHUNK_CONTEXT_PROMPT

logger = logging.getLogger(__name__)

_TRUNCATION_SUFFIX = "\n\n[Document truncated for context generation]"


@dataclass(frozen=True)
class _DocumentContextBuilder:
    """Assemble title/abstract header + body for the chunk-context LLM prompt."""

    max_document_chars: int
    title: str = ""
    abstract: str = ""

    def build(self, document: str) -> str:
        header = self._format_header()
        full_document = f"{header}\n\n{document}" if header else document

        if len(full_document) <= self.max_document_chars:
            return full_document

        return self._truncate(document, header)

    def _format_header(self) -> str:
        parts: list[str] = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.abstract:
            parts.append(f"Abstract: {self.abstract}")
        return "\n\n".join(parts)

    def _truncate(self, document: str, header: str) -> str:
        if header:
            overhead = len(header) + len(_TRUNCATION_SUFFIX) + 2  # newlines between header and body
            available = max(self.max_document_chars - overhead, 0)
            truncated_body = document[:available]
            return f"{header}\n\n{truncated_body}{_TRUNCATION_SUFFIX}"

        if len(document) > self.max_document_chars:
            return document[: self.max_document_chars] + _TRUNCATION_SUFFIX
        return document


class ChunkContextualizer:
    """Generate LLM context for each chunk to improve retrieval (Contextual Retrieval).

    For every chunk, an LLM produces a short situating context based on the full
    document. The context is prepended to the chunk text for embedding and BM25
    indexing, while the original chunk text is preserved for answer generation.
    """

    def __init__(
        self,
        chat_model: BaseChatModel,
        max_document_chars: int = 50_000,
        max_concurrent_requests: int = 3,
    ):
        self.chat_model = chat_model
        self.max_document_chars = max_document_chars
        self.max_concurrent_requests = max_concurrent_requests
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        logger.info(
            "Chunk contextualizer initialized: max_document_chars=%d, max_concurrent=%d",
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
        results = await asyncio.gather(
            *[self._contextualize_chunk(prepared_document, chunk) for chunk in chunks]
        )
        self._log_contextualization_summary(results)
        return list(results)

    async def _contextualize_chunk(self, prepared_document: str, chunk: TextChunk) -> TextChunk:
        async with self._semaphore:
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

    def _prepare_document(self, document: str, title: str, abstract: str) -> str:
        """Build document context for the LLM, truncating if necessary."""
        return _DocumentContextBuilder(
            max_document_chars=self.max_document_chars,
            title=title,
            abstract=abstract,
        ).build(document)

    async def _generate_context(self, document: str, chunk: str) -> str:
        """Call the LLM to generate situating context for a single chunk."""
        prompt = CHUNK_CONTEXT_PROMPT.format(document=document, chunk=chunk)
        response = await self.chat_model.ainvoke([HumanMessage(content=prompt)])
        context = self._extract_message_text(response).strip()
        if not context:
            raise ValueError("LLM returned empty context")
        return context

    @staticmethod
    def _extract_message_text(message: BaseMessage) -> str:
        content = message.content
        return content if isinstance(content, str) else str(content)

    @staticmethod
    def _log_contextualization_summary(chunks: List[TextChunk]) -> None:
        contextualized_count = sum(1 for chunk in chunks if chunk.metadata.chunk_context)
        arxiv_id = chunks[0].arxiv_id if chunks else "unknown"
        logger.info(
            "Contextualized %d/%d chunks for paper %s",
            contextualized_count,
            len(chunks),
            arxiv_id,
        )
