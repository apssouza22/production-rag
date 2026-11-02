import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config import Settings
from src.domain.arxiv.client import ArxivClient
from src.domain.db.interfaces.base import BaseDatabase
from src.domain.pdf_parser.artifacts import load_parsed_paper_artifact
from src.domain.pdf_parser.models import ArxivMetadata, ParsedPaper
from src.domain.pdf_parser.parser import PDFParserService

from .paper_fetcher import PaperFetcher, make_paper_fetcher
from .paper_indexer import PaperIndexer, make_paper_indexer
from .paper_persist import PaperPersist, make_paper_persist
from .schemas import DownloadResults, PersistResults

logger = logging.getLogger(__name__)


class ArxivIngestionService:
    """Gateway for arXiv ingestion pipeline steps (download, parse artifacts, persist, index)."""

    def __init__(
        self,
        arxiv_client: ArxivClient,
        pdf_parser: PDFParserService,
        database: BaseDatabase,
        metadata_fetcher: PaperFetcher,
        paper_persist: PaperPersist,
        paper_indexer: PaperIndexer,
    ) -> None:
        self.arxiv_client = arxiv_client
        self.pdf_parser = pdf_parser
        self.database = database
        self.paper_fetcher = metadata_fetcher
        self.paper_persist = paper_persist
        self.paper_indexer = paper_indexer

    def download_papers(self, target_date: str) -> DownloadResults:
        """Fetch paper metadata and download PDFs for a date (YYYYMMDD)."""
        logger.info("Downloading papers for date: %s", target_date)

        max_results = self.arxiv_client.max_results
        results = asyncio.run(
            self.paper_fetcher.fetch_and_download_papers(
                max_results=max_results,
                from_date=target_date,
                to_date=target_date,
            )
        )
        results = results.model_copy(update={"date": target_date})
        logger.info(
            "Daily download complete: %s papers, %s PDFs for %s",
            results.papers_fetched,
            results.pdfs_downloaded,
            target_date,
        )
        return results

    def load_papers_from_pdf_directory(self, pdf_dir: Path, target_date: str) -> DownloadResults:
        """Use PDFs from ``pdf_dir`` and fetch arXiv metadata instead of downloading."""
        logger.info("Loading papers from PDF directory %s for date label %s", pdf_dir, target_date)
        results = asyncio.run(
            self.paper_fetcher.load_papers_from_pdf_directory(
                pdf_dir,
                target_date=target_date,
            )
        )
        logger.info(
            "Local PDF step complete: %s papers, %s PDF paths for %s",
            results.papers_fetched,
            results.pdfs_downloaded,
            target_date,
        )
        return results

    def parse_pdfs_to_markdown(self, download_results: DownloadResults) -> dict[str, Any]:
        """Parse downloaded PDFs into shared Markdown and JSON artifacts."""
        output_dir = Path(self.paper_fetcher.pdf_cache_dir).parent / "arxiv_md"
        papers = [ArxivMetadata(**paper.model_dump()) for paper in download_results.papers]
        return asyncio.run(
            self.pdf_parser.parse_pdfs_to_markdown(
                papers=papers,
                pdf_paths_by_id=download_results.downloaded_paths,
                output_dir=output_dir,
                max_concurrent_parsing=self.paper_fetcher.max_concurrent_parsing,
                date=download_results.date,
            )
        )

    def persist_papers(self, download_results: DownloadResults, markdown_results: dict[str, Any]) -> PersistResults:
        """Persist paper metadata and parsed sidecars to PostgreSQL."""
        logger.info("Starting paper persistence task")
        start_time = datetime.now()

        if not download_results.papers:
            logger.info("No papers to persist")
            return PersistResults(
                pdfs_parsed=markdown_results.get("pdfs_parsed", 0),
                papers_stored=0,
                errors=list(markdown_results.get("errors", [])),
                date=download_results.date,
                processing_time=0,
            )

        papers = download_results.papers
        parsed_papers, errors = self._get_parsed_papers( markdown_results)

        with self.database.get_session() as session:
            papers_stored = self.paper_persist.store_papers(papers, parsed_papers, session)

        results = PersistResults(
            pdfs_parsed=markdown_results.get("pdfs_parsed", 0),
            papers_stored=papers_stored,
            errors=errors,
            date=download_results.date,
            processing_time=(datetime.now() - start_time).total_seconds(),
        )
        logger.info(
            "Paper persistence complete: %d stored for %s",
            results.papers_stored,
            results.date,
        )
        return results

    @staticmethod
    def _get_parsed_papers(markdown_results: dict[str, Any]) -> tuple[dict[str, ParsedPaper], list[str]]:
        errors = list(markdown_results.get("errors", []))
        parsed_papers: dict[str, ParsedPaper] = {}
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
        return parsed_papers, errors

    def index_papers_hybrid(
        self,
        persist_results: PersistResults | None = None,
        *,
        ingestion_name: str = "dev",
        arxiv_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Index papers with chunking and vector embeddings for hybrid search."""
        return self.paper_indexer.index_papers_hybrid(
            persist_results,
            ingestion_name=ingestion_name,
            arxiv_ids=arxiv_ids,
        )


def make_arxiv_ingestion_service(
    arxiv_client: ArxivClient,
    pdf_parser: PDFParserService,
    database: BaseDatabase,
    settings: Optional[Settings] = None,
) -> ArxivIngestionService:
    """Create ArxivIngestionService with configured dependencies."""
    metadata_fetcher = make_paper_fetcher(arxiv_client, pdf_parser, settings=settings)
    return ArxivIngestionService(
        arxiv_client=arxiv_client,
        pdf_parser=pdf_parser,
        database=database,
        metadata_fetcher=metadata_fetcher,
        paper_persist=make_paper_persist(),
        paper_indexer=make_paper_indexer(database),
    )


def get_ingestion_service() -> ArxivIngestionService:
    """Return the cached ingestion gateway (used by Airflow task adapters)."""
    from src.domain.arxiv_ingestion.common import get_cached_services

    *_, ingestion_service, _ = get_cached_services()
    return ingestion_service
