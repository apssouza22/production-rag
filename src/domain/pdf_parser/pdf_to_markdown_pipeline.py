"""Orchestration steps for PDF-to-Markdown batch export."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Self

from src.domain.pdf_parser.exceptions import PDFParsingException, PDFValidationError

from .artifacts import ParsedPaperArtifact, artifact_stem, write_parsed_paper_artifacts
from .content_extractor import ContentExtractor
from .docling import DoclingParser
from .models import ArxivMetadata, ParsedPaper, PdfContent

logger = logging.getLogger(__name__)

ConvertPdfStep = Callable[[Path], Awaitable[Any | None]]
ExtractContentStep = Callable[[Any, Path, Optional[Path]], PdfContent]


class PdfToMarkdownPipeline:
    """Run parse-then-write steps for turning downloaded PDFs into Markdown artifacts."""

    def __init__(
        self,
        docling_parser: DoclingParser,
        content_extractor: ContentExtractor,
        output_dir: Path,
        max_concurrent_parsing: int = 1,
        *,
        extract_parsing_artifacts: bool = False,
    ) -> None:
        if max_concurrent_parsing < 1:
            raise ValueError("max_concurrent_parsing must be at least 1")
        self._docling_parser = docling_parser
        self._content_extractor = content_extractor
        self._extract_parsing_artifacts = extract_parsing_artifacts
        self._output_dir = output_dir
        self._parse_semaphore = asyncio.Semaphore(max_concurrent_parsing)
        self._papers: list[ArxivMetadata] | None = None
        self._pdf_paths_by_id: dict[str, str] | None = None
        self._start_time: datetime | None = None

    def set_pdfs(self, papers: list[ArxivMetadata], pdf_paths_by_id: dict[str, str]) -> Self:
        """Configure the batch of papers and local PDF paths for the next run."""
        self._papers = papers
        self._pdf_paths_by_id = pdf_paths_by_id
        return self

    def begin(self) -> datetime:
        """Log pipeline start and capture the run start time."""
        logger.info("Starting PDF-to-Markdown task")
        self._start_time = datetime.now()
        return self._start_time

    def empty_result(self, date: Optional[str]) -> dict[str, Any]:
        """Build the result payload when there is nothing to parse."""
        logger.info("No papers to parse")
        return {
            "pdfs_parsed": 0,
            "artifacts": {},
            "errors": [],
            "date": date,
            "processing_time": 0,
        }

    async def convert_pdf(self, pdf_path: Path) -> Any | None:
        """Run Docling conversion for one PDF file."""
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise PDFValidationError(f"PDF file not found: {pdf_path}")

        try:
            conversion = await asyncio.to_thread(self._docling_parser.parse_pdf, pdf_path)
            if conversion is None:
                logger.warning(f"Docling parsing returned no result for {pdf_path.name} (skipped or empty)")
            return conversion
        except (PDFValidationError, PDFParsingException):
            raise
        except Exception as e:
            logger.error(f"Docling parsing error for {pdf_path.name}: {e}")
            raise PDFParsingException(f"Docling parsing error for {pdf_path.name}: {e}")

    def extract_content(
        self,
        conversion: Any,
        pdf_path: Path,
        artifacts_dir: Optional[Path] = None,
    ) -> PdfContent:
        """Build structured ``PdfContent`` from a Docling conversion result."""
        effective_artifacts_dir = None
        if self._extract_parsing_artifacts:
            effective_artifacts_dir = artifacts_dir
            if effective_artifacts_dir is None:
                effective_artifacts_dir = pdf_path.parent / f"{pdf_path.stem}_assets"

        result = self._content_extractor.build_content_session(
            conversion,
            artifacts_dir=effective_artifacts_dir,
        )
        logger.info(f"Parsed {pdf_path.name}")
        return result

    async def parse_pdf(self, pdf_path: Path, artifacts_dir: Optional[Path] = None) -> Optional[PdfContent]:
        """Parse one PDF with Docling and build structured content."""
        conversion = await self.convert_pdf(pdf_path)
        if conversion is None:
            return None
        return self.extract_content(conversion, pdf_path, artifacts_dir)

    async def run_pdf_parse(
        self,
        convert_pdf: ConvertPdfStep,
        extract_content: ExtractContentStep,
    ) -> dict[str, Any]:
        """Parse configured PDFs into in-memory ``ParsedPaper`` objects."""
        if self._papers is None or self._pdf_paths_by_id is None:
            raise RuntimeError("Call set_pdfs() before run_pdf_parse()")

        return await self.parse_pdfs(
            self._papers,
            self._pdf_paths_by_id,
            convert_pdf=convert_pdf,
            extract_content=extract_content,
        )

    async def parse_one(
        self,
        paper: ArxivMetadata,
        pdf_paths_by_id: dict[str, str],
        *,
        convert_pdf: ConvertPdfStep,
        extract_content: ExtractContentStep,
    ) -> Optional[ParsedPaper]:
        """Parse a single downloaded PDF and attach arXiv metadata."""
        pdf_path_dir = pdf_paths_by_id.get(paper.arxiv_id)
        if not pdf_path_dir:
            logger.warning("No PDF path available for parsing: %s", paper.arxiv_id)
            return None

        pdf_path = Path(pdf_path_dir)
        if not pdf_path.exists():
            logger.warning("No PDF available for parsing: %s", paper.arxiv_id)
            return None

        async with self._parse_semaphore:
            paper_artifacts_dir = self._output_dir / f"{artifact_stem(paper.arxiv_id)}_assets"
            conversion = await convert_pdf(pdf_path)
            if conversion is None:
                return None
            pdf_content = extract_content(conversion, pdf_path, paper_artifacts_dir)
            return ParsedPaper(arxiv_metadata=paper, pdf_content=pdf_content)

    async def parse_pdfs(
        self,
        papers: list[ArxivMetadata],
        pdf_paths_by_id: dict[str, str],
        *,
        convert_pdf: ConvertPdfStep,
        extract_content: ExtractContentStep,
    ) -> dict[str, Any]:
        """Parse PDFs into in-memory ``ParsedPaper`` objects."""
        logger.info("Starting parse batch for %d PDFs", len(papers))

        parse_results = await asyncio.gather(
            *(
                self.parse_one(
                    paper,
                    pdf_paths_by_id,
                    convert_pdf=convert_pdf,
                    extract_content=extract_content,
                )
                for paper in papers
            ),
            return_exceptions=True,
        )
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

    def write_markdown_artifacts(
        self,
        parse_results: dict[str, Any],
    ) -> tuple[dict[str, ParsedPaperArtifact], list[str]]:
        """Persist Markdown and JSON sidecars for successfully parsed papers."""
        artifacts: dict[str, ParsedPaperArtifact] = {}
        errors = list(parse_results["errors"])
        for arxiv_id, parsed_paper in parse_results["parsed_papers"].items():
            try:
                artifacts[arxiv_id] = write_parsed_paper_artifacts(self._output_dir, parsed_paper)
            except Exception as exc:
                error = f"Failed to write parsed artifacts for {arxiv_id}: {exc}"
                logger.exception(error)
                errors.append(error)
        return artifacts, errors

    def finalize(
        self,
        artifacts: dict[str, ParsedPaperArtifact],
        errors: list[str],
        date: Optional[str],
        start_time: datetime,
    ) -> dict[str, Any]:
        """Assemble the pipeline result and log completion."""
        results = {
            "pdfs_parsed": len(artifacts),
            "artifacts": artifacts,
            "errors": errors,
            "date": date,
            "processing_time": (datetime.now() - start_time).total_seconds(),
        }
        logger.info("PDF-to-Markdown complete: %d parsed for %s", results["pdfs_parsed"], date)
        return results
