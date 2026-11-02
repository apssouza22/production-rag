"""Orchestration steps for PDF-to-Markdown batch export."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Self

from src.domain.pdf_parser.exceptions import PDFParsingException, PDFValidationError

from .artifacts import ParsedPaperArtifact, artifact_stem
from .content_extractor import ContentExtractor
from .docling import DoclingParser
from .markdown_writer import ArtifactWriteResult, MarkdownWriter
from .models import ArxivMetadata, ParsedPaper, PdfContent

logger = logging.getLogger(__name__)

ConvertPdfStep = Callable[[Path], Awaitable[Any | None]]
ExtractContentStep = Callable[[Any, Path, Optional[Path]], PdfContent]


@dataclass
class PdfParseResults:
    """Outcome of parsing a batch of PDFs into in-memory ``ParsedPaper`` objects."""

    parsed_papers: dict[str, ParsedPaper] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    parse_failures: list[str] = field(default_factory=list)

    @property
    def parsed_count(self) -> int:
        return len(self.parsed_papers)


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
        self._markdown_writer = MarkdownWriter(output_dir)
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

    async def run(
        self,
        papers: list[ArxivMetadata],
        pdf_paths_by_id: dict[str, str],
        *,
        date: Optional[str] = None,
        convert_pdf: ConvertPdfStep | None = None,
        extract_content: ExtractContentStep | None = None,
    ) -> dict[str, Any]:
        """Parse configured PDFs and write Markdown artifacts end-to-end."""
        if not papers:
            return self.empty_result(date)

        self.set_pdfs(papers, pdf_paths_by_id)
        start_time = self.begin()
        pdf_parse_results = await self.run_pdf_parse(
            convert_pdf or self.convert_pdf,
            extract_content or self.extract_content,
        )
        write_result = self.write_markdown_artifacts(pdf_parse_results)
        return self.finalize(write_result.artifacts, write_result.errors, date, start_time)

    async def convert_pdf(self, pdf_path: Path) -> Any | None:
        """Run Docling conversion for one PDF file."""
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise PDFValidationError(f"PDF file not found: {pdf_path}")

        try:
            # To thread the CPU-bound Docling call out of the event loop.
            conversion = await asyncio.to_thread(self._docling_parser.parse_pdf, pdf_path)
            if conversion is None:
                logger.warning(f"Docling parsing returned no result for {pdf_path.name} (skipped or empty)")
            return conversion
        except (PDFValidationError, PDFParsingException):
            raise
        except Exception as e:
            logger.error(f"Docling parsing error for {pdf_path.name}: {e}")
            raise PDFParsingException(f"Docling parsing error for {pdf_path.name}: {e}") from e

    def extract_content(
        self,
        conversion: Any,
        pdf_path: Path,
        artifacts_dir: Optional[Path] = None,
    ) -> PdfContent:
        """Build structured ``PdfContent`` from a Docling conversion result."""
        result = self._content_extractor.build_content_session(
            conversion,
            artifacts_dir=self._content_artifacts_dir(pdf_path, artifacts_dir),
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
    ) -> PdfParseResults:
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
        pdf_path = self._local_pdf_path(paper, pdf_paths_by_id)
        if pdf_path is None:
            return None

        async with self._parse_semaphore:
            conversion = await convert_pdf(pdf_path)
            if conversion is None:
                return None
            assets_dir = self._paper_assets_dir(paper.arxiv_id)
            pdf_content = extract_content(conversion, pdf_path, assets_dir)
            return ParsedPaper(arxiv_metadata=paper, pdf_content=pdf_content)

    async def parse_pdfs(
        self,
        papers: list[ArxivMetadata],
        pdf_paths_by_id: dict[str, str],
        *,
        convert_pdf: ConvertPdfStep,
        extract_content: ExtractContentStep,
    ) -> PdfParseResults:
        """Parse PDFs into in-memory ``ParsedPaper`` objects."""
        logger.info("Starting parse batch for %d PDFs", len(papers))

        outcomes = await asyncio.gather(
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
        batch = self._collect_parse_batch(papers, outcomes)
        logger.info("Parse batch complete: %d/%d parsed", batch.parsed_count, len(papers))
        return batch

    def write_markdown_artifacts(self, pdf_parse_result: PdfParseResults) -> ArtifactWriteResult:
        """Persist Markdown and JSON sidecars for successfully parsed papers."""
        return self._markdown_writer.write_all(
            pdf_parse_result.parsed_papers,
            errors=pdf_parse_result.errors,
        )

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

    def _paper_assets_dir(self, arxiv_id: str) -> Path:
        return self._output_dir / f"{artifact_stem(arxiv_id)}_assets"

    def _content_artifacts_dir(self, pdf_path: Path, artifacts_dir: Optional[Path]) -> Optional[Path]:
        if not self._extract_parsing_artifacts:
            return None
        return artifacts_dir or pdf_path.parent / f"{pdf_path.stem}_assets"

    def _local_pdf_path(self, paper: ArxivMetadata, pdf_paths_by_id: dict[str, str]) -> Path | None:
        pdf_path_value = pdf_paths_by_id.get(paper.arxiv_id)
        if not pdf_path_value:
            logger.warning("No PDF path available for parsing: %s", paper.arxiv_id)
            return None

        pdf_path = Path(pdf_path_value)
        if not pdf_path.exists():
            logger.warning("No PDF available for parsing: %s", paper.arxiv_id)
            return None
        return pdf_path

    def _collect_parse_batch(
        self,
        papers: list[ArxivMetadata],
        outcomes: list[ParsedPaper | BaseException | None],
    ) -> PdfParseResults:
        parsed_papers: dict[str, ParsedPaper] = {}
        errors: list[str] = []
        parse_failures: list[str] = []

        for paper, outcome in zip(papers, outcomes):
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            if isinstance(outcome, BaseException):
                error = f"Parse error for {paper.arxiv_id}: {outcome}"
                logger.error(error)
                errors.append(error)
                parse_failures.append(paper.arxiv_id)
            elif outcome is not None:
                parsed_papers[paper.arxiv_id] = outcome
            else:
                parse_failures.append(paper.arxiv_id)

        errors.extend(f"PDF parse failed: {arxiv_id}" for arxiv_id in parse_failures)
        return PdfParseResults(
            parsed_papers=parsed_papers,
            errors=errors,
            parse_failures=parse_failures,
        )
