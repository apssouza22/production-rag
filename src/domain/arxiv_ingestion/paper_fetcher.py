import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config import Settings
from src.domain.arxiv.client import ArxivClient
from src.domain.arxiv.paper import ArxivPaper
from src.domain.pdf_parser.parser import PDFParserService

from .exceptions import PipelineException

logger = logging.getLogger(__name__)


class PaperFetcher:
    """Service for fetching arXiv papers and downloading PDFs."""

    def __init__(
        self,
        arxiv_client: ArxivClient,
        pdf_parser: PDFParserService,
        pdf_cache_dir: Optional[Path] = None,
        max_concurrent_downloads: int = 5,
        max_concurrent_parsing: int = 3,
        settings: Optional[Settings] = None,
    ):
        """Initialize metadata fetcher with services and settings.

        :param arxiv_client: Client for arXiv API operations
        :param pdf_parser: Service for parsing PDF documents
        :param pdf_cache_dir: Directory for caching downloaded PDFs
        :param max_concurrent_downloads: Maximum concurrent PDF downloads
        :param max_concurrent_parsing: Maximum concurrent PDF parsing operations
        :param settings: Application settings instance
        :type arxiv_client: ArxivClient
        :type pdf_parser: PDFParserService
        :type pdf_cache_dir: Optional[Path]
        :type max_concurrent_downloads: int
        :type max_concurrent_parsing: int
        :type settings: Optional[Settings]
        """
        from src.config import get_settings

        self.arxiv_client = arxiv_client
        self.pdf_parser = pdf_parser
        self.pdf_cache_dir = pdf_cache_dir or self.arxiv_client.pdf_cache_dir
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_concurrent_parsing = max_concurrent_parsing
        self.settings = settings or get_settings()

    async def fetch_and_download_papers(
        self,
        max_results: Optional[int] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> dict[str, Any]:
        """Fetch paper metadata from arXiv and download PDFs without parsing.

        :param max_results: Maximum papers to fetch
        :param from_date: Filter papers from this date (YYYYMMDD)
        :param to_date: Filter papers to this date (YYYYMMDD)
        :returns: Dictionary with fetch/download statistics and serialized papers for XCom
        """
        results: dict[str, Any] = {
            "papers_fetched": 0,
            "pdfs_downloaded": 0,
            "errors": [],
            "papers": [],
            "downloaded_paths": {},
            "processing_time": 0,
        }

        start_time = datetime.now()

        try:
            papers = await self.arxiv_client.fetch_papers(
                max_results=max_results,
                from_date=from_date,
                to_date=to_date,
                sort_by="submittedDate",
                sort_order="descending",
            )
            results["papers_fetched"] = len(papers)

            if not papers:
                logger.warning("No papers found")
                return results

            download_results = await self._download_pdfs_batch(papers)
            results["pdfs_downloaded"] = download_results["downloaded"]
            results["errors"].extend(download_results["errors"])
            results["downloaded_paths"] = {arxiv_id: str(path) for arxiv_id, path in download_results["paths"].items()}
            results["papers"] = [paper.model_dump() for paper in papers]

            results["processing_time"] = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Download step completed in {results['processing_time']:.1f}s: "
                f"{results['papers_fetched']} papers, {results['pdfs_downloaded']} PDFs"
            )
            return results

        except Exception as e:
            logger.error(f"Download step error: {e}")
            results["errors"].append(f"Download step error: {str(e)}")
            raise PipelineException(f"Download step failed: {e}") from e

    async def _download_pdfs_batch(self, papers: list[ArxivPaper]) -> dict[str, Any]:
        """Download PDFs for a batch of papers without parsing."""
        results: dict[str, Any] = {
            "downloaded": 0,
            "paths": {},
            "errors": [],
            "download_failures": [],
        }

        logger.info(f"Starting download batch for {len(papers)} PDFs")
        download_semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        download_tasks = [self._download_pdf_only(paper, download_semaphore) for paper in papers]
        download_results = await asyncio.gather(*download_tasks, return_exceptions=True)

        for paper, result in zip(papers, download_results):
            if isinstance(result, Exception):
                error_msg = f"Download error for {paper.arxiv_id}: {result}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                results["download_failures"].append(paper.arxiv_id)
            elif result:
                results["downloaded"] += 1
                results["paths"][paper.arxiv_id] = result
            else:
                results["download_failures"].append(paper.arxiv_id)

        if results["download_failures"]:
            results["errors"].extend([f"Download failed: {arxiv_id}" for arxiv_id in results["download_failures"]])

        logger.info(f"Download batch complete: {results['downloaded']}/{len(papers)} downloaded")
        return results

    async def _download_pdf_only(self, paper: ArxivPaper, download_semaphore: asyncio.Semaphore) -> Optional[Path]:
        """Download a single PDF with concurrency control."""
        async with download_semaphore:
            logger.debug(f"Starting download: {paper.arxiv_id}")
            pdf_path = await self.arxiv_client.download_pdf(paper, False)

            if pdf_path:
                logger.debug(f"Download complete: {paper.arxiv_id}")
                return pdf_path

            logger.error(f"Download failed: {paper.arxiv_id}")
            return None


def make_paper_fetcher(
    arxiv_client: ArxivClient,
    pdf_parser: PDFParserService,
    pdf_cache_dir: Optional[Path] = None,
    settings: Optional[Settings] = None,
) -> PaperFetcher:
    """Create MetadataFetcher instance with configuration settings.

    :param arxiv_client: Client for arXiv API operations
    :param pdf_parser: Service for parsing PDF documents
    :param pdf_cache_dir: Directory for caching downloaded PDFs
    :param settings: Application settings instance (uses default if None)
    :type arxiv_client: ArxivClient
    :type pdf_parser: PDFParserService
    :type pdf_cache_dir: Optional[Path]
    :type settings: Optional[Settings]
    :returns: Configured MetadataFetcher instance
    :rtype: PaperFetcher
    """
    from src.config import get_settings

    if settings is None:
        settings = get_settings()

    return PaperFetcher(
        arxiv_client=arxiv_client,
        pdf_parser=pdf_parser,
        pdf_cache_dir=pdf_cache_dir,
        max_concurrent_downloads=settings.arxiv.max_concurrent_downloads,
        max_concurrent_parsing=settings.arxiv.max_concurrent_parsing,
        settings=settings,
    )
