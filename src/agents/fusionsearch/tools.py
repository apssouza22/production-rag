import logging

from langchain_core.documents import Document
from langchain_core.tools import tool

from src.agents.fusionsearch.retrieval_settings import RetrievalSettings
from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.domain.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)


async def _rerank_hits(
    reranker_client: JinaRerankerClient,
    query: str,
    hits: list[dict],
    top_n: int,
    model: str,
) -> list[dict]:
    """Rerank OpenSearch hits using Jina reranker API."""
    documents = [hit["chunk_text"] for hit in hits]
    rerank_results = await reranker_client.rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model=model,
    )

    reranked_hits = []
    for result in rerank_results:
        hit = dict(hits[result.index])
        hit["score"] = result.relevance_score
        hit["rerank_score"] = result.relevance_score
        hit["original_rank"] = result.index
        reranked_hits.append(hit)

    return reranked_hits


def create_retriever_tool(
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    retrieval_settings: RetrievalSettings,
    reranker_client: JinaRerankerClient | None = None,
):
    """Create a retriever tool that wraps OpenSearch service.

    :param opensearch_client: Existing OpenSearch service
    :param embeddings_client: Existing Jina embeddings service
    :param retrieval_settings: Mutable retrieval settings (updated per request)
    :param reranker_client: Optional Jina reranker client
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
        use_hybrid = retrieval_settings.use_hybrid
        rerank_enabled = (
            retrieval_settings.rerank_enabled
            and reranker_client is not None
            and reranker_client.is_configured
        )

        logger.info(f"Retrieving papers for query: {query[:100]}...")
        logger.debug(
            "Search mode: %s, top_k: %s, rerank: %s",
            "hybrid" if use_hybrid else "bm25",
            top_k,
            rerank_enabled,
        )

        search_size = top_k
        if rerank_enabled:
            search_size = top_k * retrieval_settings.rerank_candidate_multiplier

        # Generate query embedding
        logger.debug("Generating query embedding")
        query_embedding = await embeddings_client.embed_query(query)
        logger.debug(f"Generated embedding with {len(query_embedding)} dimensions")

        # Search using OpenSearch
        logger.debug("Searching OpenSearch")
        search_results = opensearch_client.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=search_size,
            use_hybrid=use_hybrid,
        )

        hits = search_results.get("hits", [])
        logger.info(f"Found {len(hits)} documents from OpenSearch")

        if rerank_enabled and len(hits) > 1:
            try:
                logger.info("Reranking %d candidates with Jina reranker", len(hits))
                hits = await _rerank_hits(
                    reranker_client=reranker_client,
                    query=query,
                    hits=hits,
                    top_n=top_k,
                    model=retrieval_settings.rerank_model,
                )
                logger.info("Reranking complete, returning top %d documents", len(hits))
            except Exception as e:
                logger.warning(
                    "Reranking failed, using original search order: query_preview=%r, "
                    "candidate_count=%d, top_k=%d, model=%s, error_type=%s, error=%s",
                    query[:100],
                    len(hits),
                    top_k,
                    retrieval_settings.rerank_model,
                    type(e).__name__,
                    e,
                )
                hits = hits[:top_k]
        else:
            hits = hits[:top_k]

        # Convert SearchHit to LangChain Document
        documents = []
        for hit in hits:
            doc = Document(
                page_content=hit["chunk_text"],
                metadata={
                    "arxiv_id": hit["arxiv_id"],
                    "title": hit.get("title", ""),
                    "authors": hit.get("authors", ""),
                    "score": hit.get("score", 0.0),
                    "source": f"https://arxiv.org/pdf/{hit['arxiv_id']}.pdf",
                    "section": hit.get("section_name", ""),
                    "search_mode": "hybrid" if use_hybrid else "bm25",
                    "top_k": top_k,
                    "reranked": rerank_enabled and "rerank_score" in hit,
                    "rerank_score": hit.get("rerank_score"),
                    "original_rank": hit.get("original_rank"),
                },
            )
            documents.append(doc)

        logger.debug(f"Converted {len(documents)} hits to LangChain Documents")
        logger.info(f"✓ Retrieved {len(documents)} papers successfully")

        return documents

    return retrieve_papers
