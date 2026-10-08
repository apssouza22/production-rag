import asyncio
import logging
from datetime import datetime, timedelta

from src.domain.arxiv.paper import ArxivPaper

from .common import get_cached_services

logger = logging.getLogger(__name__)


async def run_paper_ingestion_pipeline(
    target_date: str,
    process_pdfs: bool = True,
) -> dict:
    """Async wrapper for the paper ingestion pipeline.

    :param target_date: Date to fetch papers for (YYYYMMDD format)
    :param process_pdfs: Whether to download and process PDFs
    :returns: Dictionary with ingestion statistics
    """
    arxiv_client, _, database, metadata_fetcher, _ = get_cached_services()

    max_results = arxiv_client.max_results
    logger.info(f"Using default max_results from config: {max_results}")

    with database.get_session() as session:
        return await metadata_fetcher.fetch_and_process_papers(
            max_results=max_results,
            from_date=target_date,
            to_date=target_date,
            process_pdfs=process_pdfs,
            store_to_db=True,
            db_session=session,
        )


def download_papers(target_date: str) -> dict:
    """Fetch paper metadata and download PDFs for a date (YYYYMMDD)."""
    logger.info("Downloading papers for date: %s", target_date)

    arxiv_client, _, _, metadata_fetcher, _ = get_cached_services()
    max_results = arxiv_client.max_results

    results = asyncio.run(
        metadata_fetcher.fetch_and_download_papers(
            max_results=max_results,
            from_date=target_date,
            to_date=target_date,
        )
    )

    results["date"] = target_date
    logger.info(
        f"Daily download complete: {results['papers_fetched']} papers, {results['pdfs_downloaded']} PDFs for {target_date}"
    )

    return results


def parse_pdfs(download_results: dict) -> dict:
    """Parse downloaded PDFs and store papers in PostgreSQL."""
    logger.info("Starting PDF parsing task")

    papers_data = download_results.get("papers", [])
    if not papers_data:
        logger.info("No papers to parse")
        results = {
            "pdfs_parsed": 0,
            "papers_stored": 0,
            "errors": [],
            "date": download_results.get("date"),
        }
        return results

    papers = [ArxivPaper(**paper_data) for paper_data in papers_data]
    downloaded_paths = download_results.get("downloaded_paths", {})

    _, _, database, metadata_fetcher, _ = get_cached_services()
    with database.get_session() as session:
        results = asyncio.run(
            metadata_fetcher.parse_and_store_papers(
                papers=papers,
                downloaded_paths=downloaded_paths,
                db_session=session,
            )
        )

    results["date"] = download_results.get("date")
    logger.info(f"PDF parsing complete: {results['pdfs_parsed']} parsed, {results['papers_stored']} stored for {results['date']}")

    return results
