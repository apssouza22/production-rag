import asyncio
import logging
from datetime import datetime
from pathlib import Path

from src.domain.arxiv.paper import ArxivPaper

from .artifacts import load_parsed_paper_artifact, write_parsed_paper_artifacts
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


def parse_pdfs_to_markdown(download_results: dict) -> dict:
    """Parse downloaded PDFs into shared Markdown and JSON artifacts."""
    logger.info("Starting PDF-to-Markdown task")
    start_time = datetime.now()

    papers_data = download_results.get("papers", [])
    if not papers_data:
        logger.info("No papers to parse")
        return {
            "pdfs_parsed": 0,
            "artifacts": {},
            "errors": [],
            "date": download_results.get("date"),
            "processing_time": 0,
        }

    papers = [ArxivPaper(**paper_data) for paper_data in papers_data]
    downloaded_paths = download_results.get("downloaded_paths", {})

    _, _, _, metadata_fetcher, _ = get_cached_services()
    output_dir = Path(metadata_fetcher.pdf_cache_dir).parent / "arxiv_md"
    parse_results = asyncio.run(
        metadata_fetcher.parse_papers(
            papers=papers,
            downloaded_paths=downloaded_paths,
            artifacts_dir=output_dir,
        )
    )

    artifacts = {}
    errors = list(parse_results["errors"])
    for arxiv_id, parsed_paper in parse_results["parsed_papers"].items():
        try:
            artifacts[arxiv_id] = write_parsed_paper_artifacts(output_dir, parsed_paper)
        except Exception as exc:
            error = f"Failed to write parsed artifacts for {arxiv_id}: {exc}"
            logger.exception(error)
            errors.append(error)

    results = {
        "pdfs_parsed": len(artifacts),
        "artifacts": artifacts,
        "errors": errors,
        "date": download_results.get("date"),
        "processing_time": (datetime.now() - start_time).total_seconds(),
    }
    logger.info(
        "PDF-to-Markdown complete: %d parsed for %s",
        results["pdfs_parsed"],
        results["date"],
    )
    return results


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
