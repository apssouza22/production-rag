import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.domain.pdf_parser.exceptions import PDFParsingException, PDFValidationError

from .artifacts import ParsedPaperArtifact, artifact_stem, write_parsed_paper_artifacts
from .docling import DoclingParser
from .models import ArxivMetadata, ParsedPaper, PdfContent

logger = logging.getLogger(__name__)


class PDFParserService:
    """Main PDF parsing service using Docling only."""

    def __init__(
        self,
        max_pages: int,
        max_file_size_mb: int,
        do_ocr: bool = False,
        do_table_structure: bool = False,
        do_figure_extraction: bool = False,
    ):
        """Initialize PDF parser service with configurable limits."""
        self.docling_parser = DoclingParser(
            max_pages=max_pages,
            max_file_size_mb=max_file_size_mb,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            do_figure_extraction=do_figure_extraction,
        )

    async def parse_pdf(self, pdf_path: Path, artifacts_dir: Optional[Path] = None) -> Optional[PdfContent]:
        """Parse PDF using Docling parser only.

        :param pdf_path: Path to PDF file
        :returns: PdfContent object or None if parsing failed
        """
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise PDFValidationError(f"PDF file not found: {pdf_path}")

        try:
            if artifacts_dir is None:
                result = await asyncio.to_thread(self.docling_parser.parse_pdf, pdf_path)
            else:
                result = await asyncio.to_thread(self.docling_parser.parse_pdf, pdf_path, artifacts_dir)
            if result:
                logger.info(f"Parsed {pdf_path.name}")
                return result

            logger.warning(f"Docling parsing returned no result for {pdf_path.name} (skipped or empty)")
            return None

        except (PDFValidationError, PDFParsingException):
            raise
        except Exception as e:
            logger.error(f"Docling parsing error for {pdf_path.name}: {e}")
            raise PDFParsingException(f"Docling parsing error for {pdf_path.name}: {e}")

    async def parse_paper(
        self,
        pdf_path: Path,
        metadata: ArxivMetadata,
        artifacts_dir: Optional[Path] = None,
    ) -> Optional[ParsedPaper]:
        """Parse one PDF and combine its content with source metadata."""
        pdf_content = await self.parse_pdf(pdf_path, artifacts_dir=artifacts_dir)
        if pdf_content is None:
            return None

        return ParsedPaper(arxiv_metadata=metadata, pdf_content=pdf_content)

    async def parse_papers(
        self,
        papers: list[ArxivMetadata],
        downloaded_paths: dict[str, str],
        artifacts_dir: Optional[Path] = None,
        max_concurrent_parsing: int = 1,
    ) -> dict[str, Any]:
        """Parse already-downloaded PDFs with bounded concurrency."""
        if not papers:
            logger.warning("No papers to parse")
            return {"parsed": 0, "parsed_papers": {}, "errors": [], "parse_failures": []}
        if max_concurrent_parsing < 1:
            raise ValueError("max_concurrent_parsing must be at least 1")

        logger.info("Starting parse batch for %d PDFs", len(papers))
        parse_semaphore = asyncio.Semaphore(max_concurrent_parsing)

        async def parse_one(paper: ArxivMetadata) -> Optional[ParsedPaper]:
            downloaded_path = downloaded_paths.get(paper.arxiv_id)
            if not downloaded_path:
                logger.warning("No PDF path available for parsing: %s", paper.arxiv_id)
                return None

            pdf_path = Path(downloaded_path)
            if not pdf_path.exists():
                logger.warning("No PDF available for parsing: %s", paper.arxiv_id)
                return None

            async with parse_semaphore:
                paper_artifacts_dir = (
                    artifacts_dir / f"{artifact_stem(paper.arxiv_id)}_assets" if artifacts_dir is not None else None
                )
                return await self.parse_paper(pdf_path, paper, artifacts_dir=paper_artifacts_dir)

        parse_results = await asyncio.gather(*(parse_one(paper) for paper in papers), return_exceptions=True)
        parsed_papers: dict[str, ParsedPaper] = {}
        errors: list[str] = []
        parse_failures: list[str] = []

        for paper, result in zip(papers, parse_results):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, Exception):
                error = f"Parse error for {paper.arxiv_id}: {result}"
                logger.error(error)
                errors.append(error)
                parse_failures.append(paper.arxiv_id)
            elif result is not None:
                parsed_papers[paper.arxiv_id] = result
            else:
                parse_failures.append(paper.arxiv_id)

        errors.extend(f"PDF parse failed: {arxiv_id}" for arxiv_id in parse_failures)
        logger.info("Parse batch complete: %d/%d parsed", len(parsed_papers), len(papers))
        return {
            "parsed": len(parsed_papers),
            "parsed_papers": parsed_papers,
            "errors": errors,
            "parse_failures": parse_failures,
        }

    async def parse_pdfs_to_markdown(
        self,
        papers: list[ArxivMetadata],
        downloaded_paths: dict[str, str],
        output_dir: Path,
        max_concurrent_parsing: int = 1,
        date: Optional[str] = None,
    ) -> dict[str, Any]:
        """Parse downloaded PDFs and write shared Markdown and JSON artifacts."""
        logger.info("Starting PDF-to-Markdown task")
        start_time = datetime.now()

        if not papers:
            logger.info("No papers to parse")
            return {
                "pdfs_parsed": 0,
                "artifacts": {},
                "errors": [],
                "date": date,
                "processing_time": 0,
            }

        parse_results = await self.parse_papers(
            papers=papers,
            downloaded_paths=downloaded_paths,
            artifacts_dir=output_dir,
            max_concurrent_parsing=max_concurrent_parsing,
        )

        artifacts: dict[str, ParsedPaperArtifact] = {}
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
            "date": date,
            "processing_time": (datetime.now() - start_time).total_seconds(),
        }
        logger.info("PDF-to-Markdown complete: %d parsed for %s", results["pdfs_parsed"], date)
        return results
