import json
import logging
from datetime import datetime

from .common import get_cached_services

logger = logging.getLogger(__name__)


def generate_daily_report(
    download_stats: dict,
    parse_stats: dict,
    hybrid_stats: dict,
    execution_date: datetime | None = None,
) -> dict:
    """Build a daily report from ingestion results and current service statistics."""
    logger.info("Generating daily ingestion report")

    report = {
        "execution_date": (execution_date or datetime.now()).isoformat(),
        "download_statistics": {
            "papers_fetched": download_stats.get("papers_fetched", 0),
            "pdfs_downloaded": download_stats.get("pdfs_downloaded", 0),
            "target_date": download_stats.get("date", parse_stats.get("date", "unknown")),
        },
        "parse_statistics": {
            "pdfs_parsed": parse_stats.get("pdfs_parsed", 0),
            "papers_stored": parse_stats.get("papers_stored", 0),
        },
        "indexing_statistics": {
            "papers_processed": hybrid_stats.get("papers_processed", 0),
            "chunks_created": hybrid_stats.get("total_chunks_created", 0),
            "chunks_indexed": hybrid_stats.get("total_chunks_indexed", 0),
            "embeddings_generated": hybrid_stats.get("total_embeddings_generated", 0),
        },
        "pipeline_status": "success" if download_stats and parse_stats and hybrid_stats else "partial",
    }

    try:
        _arxiv_client, _pdf_parser, database, _metadata_fetcher, opensearch_client = get_cached_services()

        with database.get_session() as session:
            from sqlalchemy import func

            from src.domain.paper import Paper

            total_papers = session.query(func.count(Paper.id)).scalar()
            report["database_statistics"] = {"total_papers": total_papers}

        if opensearch_client.health_check():
            try:
                stats_response = opensearch_client.client.indices.stats(index=opensearch_client.index_name)

                count_response = opensearch_client.client.count(index=opensearch_client.index_name)

                index_stats = stats_response["indices"][opensearch_client.index_name]["total"]

                report["opensearch_statistics"] = {
                    "index_name": opensearch_client.index_name,
                    "document_count": count_response["count"],
                    "index_size_mb": round(index_stats["store"]["size_in_bytes"] / (1024 * 1024), 2),
                }
            except Exception as stats_error:
                logger.error(f"Failed to get OpenSearch statistics: {stats_error}")
                report["opensearch_statistics"] = {"index_name": opensearch_client.index_name, "error": str(stats_error)}
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        report["error"] = str(e)

    logger.info("Daily Ingestion Report:")
    logger.info(json.dumps(report, indent=2))

    return report
