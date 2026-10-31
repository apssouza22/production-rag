import asyncio
import logging
from datetime import datetime
from pathlib import Path

from src.domain.arxiv.paper import ArxivPaper
from src.domain.pdf_parser.artifacts import load_parsed_paper_artifact
from src.domain.pdf_parser.models import ArxivMetadata

from .common import get_cached_services

logger = logging.getLogger(__name__)


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


def parse_pdfs_to_markdown(download_results: dict) -> dict:
    """Parse downloaded PDFs into shared Markdown and JSON artifacts."""
    _, pdf_parser, _, metadata_fetcher, _ = get_cached_services()
    output_dir = Path(metadata_fetcher.pdf_cache_dir).parent / "arxiv_md"
    papers = [ArxivMetadata(**paper_data) for paper_data in download_results.get("papers", [])]
    return asyncio.run(
        pdf_parser.parse_pdfs_to_markdown(
            papers=papers,
            downloaded_paths=download_results.get("downloaded_paths", {}),
            output_dir=output_dir,
            max_concurrent_parsing=metadata_fetcher.max_concurrent_parsing,
            date=download_results.get("date"),
        )
    )


def persist_papers(download_results: dict, markdown_results: dict) -> dict:
    """Persist paper metadata and parsed sidecars to PostgreSQL."""
    logger.info("Starting paper persistence task")
    start_time = datetime.now()

    papers_data = download_results.get("papers", [])
    if not papers_data:
        logger.info("No papers to persist")
        return {
            "pdfs_parsed": markdown_results.get("pdfs_parsed", 0),
            "papers_stored": 0,
            "errors": list(markdown_results.get("errors", [])),
            "date": download_results.get("date"),
            "processing_time": 0,
        }

    papers = [ArxivPaper(**paper_data) for paper_data in papers_data]
    parsed_papers = {}
    errors = list(markdown_results.get("errors", []))

    for arxiv_id, artifact in markdown_results.get("artifacts", {}).items():
        try:
            parsed_paper = load_parsed_paper_artifact(artifact["sidecar_path"])
            if parsed_paper.arxiv_metadata.arxiv_id != arxiv_id:
                raise ValueError(
                    f"Sidecar arXiv ID {parsed_paper.arxiv_metadata.arxiv_id} does not match artifact key {arxiv_id}"
                )
            parsed_papers[arxiv_id] = parsed_paper
        except Exception as exc:
            error = f"Failed to load parsed artifact for {arxiv_id}: {exc}"
            logger.exception(error)
            errors.append(error)

    _, _, database, metadata_fetcher, _ = get_cached_services()
    with database.get_session() as session:
        papers_stored = metadata_fetcher.store_papers(papers, parsed_papers, session)

    results = {
        "pdfs_parsed": markdown_results.get("pdfs_parsed", 0),
        "papers_stored": papers_stored,
        "errors": errors,
        "date": download_results.get("date"),
        "processing_time": (datetime.now() - start_time).total_seconds(),
    }
    logger.info(
        "Paper persistence complete: %d stored for %s",
        results["papers_stored"],
        results["date"],
    )
    return results
