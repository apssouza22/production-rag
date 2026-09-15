import logging
from typing import Any

from langchain_core.tools import tool

from src.agents.fusionsearch.confidence import compute_retrieval_confidence, filter_confident_documents
from src.agents.fusionsearch.config import GraphConfig
from src.agents.fusionsearch.utils import build_source_items, format_documents_for_context
from src.domain.rerank.schemas import SearchDocument
from src.domain.rerank.service import RerankSearchService

logger = logging.getLogger(__name__)


def create_retriever_tool(rerank_search_service: RerankSearchService, graph_config: GraphConfig):
    """Create a retriever tool that wraps the rerank search service.

    :param rerank_search_service: OpenSearch + Jina rerank search service
    :param graph_config: Graph configuration with per-request top_k
    :returns: LangChain tool for retrieving papers
    """

    @tool
    async def retrieve_papers(query: str) -> dict[str, Any]:
        """Search and return relevant arXiv research papers.

        Use this tool when the user asks about:
        - Machine learning concepts or techniques
        - Deep learning architectures
        - Natural language processing
        - Computer vision methods
        - AI research topics
        - Specific algorithms or models

        :param query: The search query describing what papers to find
        :returns: Structured retrieval payload with formatted context and confidence metadata
        """
        top_k = graph_config.top_k

        logger.info(f"Retrieving papers for query: {query[:100]}...")

        search_result = await rerank_search_service.search(query=query, top_k=top_k)
        rerank_applied = search_result.rerank_applied

        confident_documents = filter_confident_documents(
            search_result.after_rerank,
            min_document_score=graph_config.min_document_score,
            rerank_applied=rerank_applied,
        )
        retrieval_confidence = compute_retrieval_confidence(confident_documents, rerank_applied=rerank_applied)
        sources = build_source_items(confident_documents, rerank_applied=rerank_applied)
        formatted_context = format_documents_for_context(confident_documents)

        logger.info(
            "Retrieval complete: before_rerank=%d, after_rerank=%d, confident=%d, confidence=%.3f, rerank_applied=%s",
            len(search_result.before_rerank),
            len(search_result.after_rerank),
            len(confident_documents),
            retrieval_confidence,
            search_result.rerank_applied,
        )
        logger.info(
            "✓ Retrieved %d papers (%d above confidence threshold, score=%.3f)",
            len(search_result.after_rerank),
            len(confident_documents),
            retrieval_confidence,
        )

        return {
            "formatted_context": formatted_context,
            "retrieval_confidence": retrieval_confidence,
            "document_count": len(confident_documents),
            "search_mode": search_result.search_mode,
            "rerank_applied": rerank_applied,
            "sources": [source.to_dict() for source in sources],
            "documents": [_serialize_document(document) for document in confident_documents],
        }

    return retrieve_papers


def _serialize_document(document: SearchDocument) -> dict[str, Any]:
    return {
        "arxiv_id": document.arxiv_id,
        "title": document.title,
        "authors": document.authors,
        "section_name": document.section_name,
        "score": document.score,
        "rerank_score": document.rerank_score,
        "chunk_text": document.chunk_text,
    }
