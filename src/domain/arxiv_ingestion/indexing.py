import logging

from src.domain.opensearch.factory import make_opensearch_client_fresh

logger = logging.getLogger(__name__)


def index_papers_hybrid(persist_results: dict | None = None) -> dict:
    """Index papers via the ingestion gateway (backward-compatible entry point)."""
    from src.domain.arxiv_ingestion.service import get_ingestion_service

    return get_ingestion_service().index_papers_hybrid(persist_results)


def verify_hybrid_index() -> dict:
    """Verify hybrid index health and get statistics."""
    try:
        opensearch_client = make_opensearch_client_fresh()
        stats = opensearch_client.client.indices.stats(index=opensearch_client.index_name)
        count = opensearch_client.client.count(index=opensearch_client.index_name)
        paper_count_query = {"aggs": {"unique_papers": {"cardinality": {"field": "arxiv_id"}}}, "size": 0}
        paper_count_response = opensearch_client.client.search(index=opensearch_client.index_name, body=paper_count_query)
        unique_papers = paper_count_response["aggregations"]["unique_papers"]["value"]

        result = {
            "index_name": opensearch_client.index_name,
            "total_chunks": count["count"],
            "unique_papers": unique_papers,
            "avg_chunks_per_paper": (count["count"] / unique_papers if unique_papers > 0 else 0),
            "index_size_mb": stats["indices"][opensearch_client.index_name]["total"]["store"]["size_in_bytes"] / (1024 * 1024),
        }

        logger.info(
            f"Hybrid index stats: {result['total_chunks']} chunks, "
            f"{result['unique_papers']} papers, "
            f"{result['avg_chunks_per_paper']:.1f} chunks/paper"
        )

        return result

    except Exception as e:
        logger.error(f"Failed to verify hybrid index: {e}")
        raise
