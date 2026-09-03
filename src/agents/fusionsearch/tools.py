import logging

from langchain_core.documents import Document
from langchain_core.tools import tool

from src.agents.fusionsearch.retrieval_settings import RetrievalSettings
from src.domain.rerank.schemas import SearchDocument
from src.domain.rerank.service import RerankSearchService

logger = logging.getLogger(__name__)


def _to_langchain_document(document: SearchDocument, *, search_mode: str, top_k: int, reranked: bool) -> Document:
    return Document(
        page_content=document.chunk_text,
        metadata={
            "arxiv_id": document.arxiv_id,
            "title": document.title,
            "authors": document.authors,
            "score": document.score,
            "source": f"https://arxiv.org/pdf/{document.arxiv_id}.pdf",
            "section": document.section_name,
            "search_mode": search_mode,
            "top_k": top_k,
            "reranked": reranked,
            "rerank_score": document.rerank_score,
            "original_rank": document.original_rank,
        },
    )


def create_retriever_tool(
    rerank_search_service: RerankSearchService,
    retrieval_settings: RetrievalSettings,
):
    """Create a retriever tool that wraps the rerank search service.

    :param rerank_search_service: OpenSearch + Jina rerank search service
    :param retrieval_settings: Mutable retrieval settings (updated per request)
    :returns: LangChain tool for retrieving papers
    """

    @tool
    async def retrieve_papers(query: str) -> list[Document]:
        """Search and return relevant arXiv research papers.

        Use this tool when the user asks about:
        - Machine learning concepts or techniques
        - Deep learning architectures
        - Natural language processing
        - Computer vision methods
        - AI research topics
        - Specific algorithms or models

        :param query: The search query describing what papers to find
        :returns: List of relevant paper excerpts with metadata
        """
        top_k = retrieval_settings.top_k

        logger.info(f"Retrieving papers for query: {query[:100]}...")

        search_result = await rerank_search_service.search(
            query=query,
            top_k=top_k,
            use_hybrid=retrieval_settings.use_hybrid,
            rerank_enabled=retrieval_settings.rerank_enabled,
            rerank_candidate_multiplier=retrieval_settings.rerank_candidate_multiplier,
            rerank_model=retrieval_settings.rerank_model,
        )

        documents = [
            _to_langchain_document(
                document,
                search_mode=search_result.search_mode,
                top_k=top_k,
                reranked=search_result.rerank_applied and document.rerank_score is not None,
            )
            for document in search_result.after_rerank
        ]

        logger.debug(
            "Retrieval complete: before_rerank=%d, after_rerank=%d, rerank_applied=%s",
            len(search_result.before_rerank),
            len(search_result.after_rerank),
            search_result.rerank_applied,
        )
        logger.info(f"✓ Retrieved {len(documents)} papers successfully")

        return documents

    return retrieve_papers
