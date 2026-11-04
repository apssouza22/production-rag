"""Ingest the experiment PDFs, skipping the run when the partition is already indexed."""

import logging
from datetime import datetime
from typing import Any

from src.domain.arxiv_ingestion import setup_environment
from src.domain.arxiv_ingestion.service import get_ingestion_service
from src.domain.opensearch.hybrid_rrf_search import PaperRrfSearch

from evals.experiments.config import ExperimentSettings

logger = logging.getLogger(__name__)


def count_indexed_chunks(paper_rrf_search: PaperRrfSearch, ingestion_name: str) -> int:
    """Return how many chunks are indexed for an ingestion partition."""
    if not paper_rrf_search.client.indices.exists(index=paper_rrf_search.index_name):
        return 0

    response = paper_rrf_search.client.count(
        index=paper_rrf_search.index_name,
        body={"query": {"term": {"ingestion_name": ingestion_name}}},
    )
    return int(response["count"])


def ensure_ingested(
    paper_rrf_search: PaperRrfSearch,
    settings: ExperimentSettings,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Run the ingestion pipeline over the local PDFs unless the partition already has chunks."""
    ingestion_name = settings.experiment_ingestion_name
    chunk_count = count_indexed_chunks(paper_rrf_search, ingestion_name)

    if chunk_count and not force:
        logger.info(
            "Skipping ingestion: %s chunks already indexed for ingestion_name=%s",
            chunk_count,
            ingestion_name,
        )
        return {"skipped": True, "chunk_count": chunk_count, "ingestion_name": ingestion_name}

    pdfs_dir = settings.experiment_pdfs_dir.resolve()
    if not pdfs_dir.is_dir():
        raise FileNotFoundError(f"Experiment PDF directory does not exist: {pdfs_dir}")

    logger.info("Ingesting PDFs from %s into ingestion_name=%s", pdfs_dir, ingestion_name)
    setup_environment()
    ingestion = get_ingestion_service()

    # Local PDFs replace the download step; the pipeline then runs as usual from parsing onward.
    target_date = datetime.now().strftime("%Y%m%d")
    download_results = ingestion.load_papers_from_pdf_directory(pdfs_dir, target_date)
    if not download_results.papers:
        raise RuntimeError(f"No arXiv metadata could be resolved for the PDFs in {pdfs_dir}")

    markdown_results = ingestion.parse_pdfs_to_markdown(download_results)
    persist_results = ingestion.persist_papers(download_results, markdown_results)

    # Index by arXiv ID: papers already present in PostgreSQL keep their original
    # created_at, so the recency-based default would index unrelated papers.
    arxiv_ids = [paper.arxiv_id for paper in download_results.papers]
    hybrid_stats = ingestion.index_papers_hybrid(
        persist_results,
        ingestion_name=ingestion_name,
        arxiv_ids=arxiv_ids,
    )

    chunk_count = count_indexed_chunks(paper_rrf_search, ingestion_name)
    if not chunk_count:
        raise RuntimeError(f"Ingestion produced no chunks for ingestion_name={ingestion_name}")

    logger.info("Ingestion complete: %s chunks indexed for ingestion_name=%s", chunk_count, ingestion_name)
    return {
        "skipped": False,
        "chunk_count": chunk_count,
        "ingestion_name": ingestion_name,
        "pdfs_dir": str(pdfs_dir),
        "arxiv_ids": arxiv_ids,
        "papers_stored": persist_results.papers_stored,
        "pdfs_parsed": markdown_results.get("pdfs_parsed", 0),
        "chunks_indexed": hybrid_stats.get("total_chunks_indexed", 0),
    }
