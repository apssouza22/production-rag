import json
import logging
from typing import Any, List, Optional

from langchain_core.messages import HumanMessage, ToolMessage

from src.agents.fusionsearch.models import SourceItem
from src.domain.rerank.schemas import SearchDocument

logger = logging.getLogger(__name__)


def get_latest_query(messages: List) -> str:
    """Get the latest user query from messages.

    :param messages: List of messages
    :returns: Latest query text
    :raises ValueError: If no user query found
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content

    raise ValueError("No user query found in messages")


def parse_tool_message_content(content: Any) -> dict[str, Any]:
    """Parse structured retrieval output from a tool message."""
    if isinstance(content, dict):
        return content

    if not isinstance(content, str) or not content.strip():
        return {}

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    return {"formatted_context": content}


def get_latest_retrieval_payload(messages: List) -> dict[str, Any]:
    """Get the latest structured retrieval payload from tool messages."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return parse_tool_message_content(msg.content if hasattr(msg, "content") else "")

    return {}


def get_latest_context(messages: List) -> str:
    """Get the latest retrieval context from tool messages.

    :param messages: List of messages
    :returns: Latest context text or empty string
    """
    payload = get_latest_retrieval_payload(messages)
    formatted_context = payload.get("formatted_context")
    if isinstance(formatted_context, str):
        return formatted_context

    if isinstance(payload.get("context"), str):
        return payload["context"]

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            content = msg.content if hasattr(msg, "content") else ""
            if isinstance(content, str):
                return content

    return ""


def format_document_for_context(document: SearchDocument, *, index: int) -> str:
    """Format a single retrieved chunk for LLM context with citation metadata."""
    authors = document.authors if isinstance(document.authors, str) else ", ".join(document.authors)
    return (
        f"[Document {index}] Title: {document.title}\n"
        f"arXiv ID: {document.arxiv_id}\n"
        f"Authors: {authors}\n"
        f"Section: {document.section_name or 'N/A'}\n"
        f"Content:\n{document.chunk_text}"
    )


def format_documents_for_context(documents: list[SearchDocument]) -> str:
    """Format retrieved documents into a single context block."""
    if not documents:
        return ""

    return "\n\n---\n\n".join(
        format_document_for_context(document, index=index)
        for index, document in enumerate(documents, start=1)
    )


def build_source_items(
    documents: list[SearchDocument],
    *,
    rerank_applied: bool,
) -> list[SourceItem]:
    """Build structured source items from retrieved documents."""
    from src.agents.fusionsearch.confidence import normalize_document_scores

    if not documents:
        return []

    scores = normalize_document_scores(documents, rerank_applied=rerank_applied)
    sources: list[SourceItem] = []

    for document, relevance_score in zip(documents, scores, strict=True):
        authors = document.authors
        if isinstance(authors, str):
            author_list = [author.strip() for author in authors.split(",") if author.strip()]
        else:
            author_list = list(authors)

        sources.append(
            SourceItem(
                arxiv_id=document.arxiv_id,
                title=document.title,
                authors=author_list,
                url=f"https://arxiv.org/pdf/{document.arxiv_id}.pdf",
                relevance_score=relevance_score,
            )
        )

    return sources


def sources_to_urls(sources: list[SourceItem]) -> list[str]:
    """Convert structured sources to PDF URL strings for API responses."""
    return [source.url for source in sources if source.url]
