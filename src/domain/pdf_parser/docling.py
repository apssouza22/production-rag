import logging
from pathlib import Path
from typing import Any, Optional

import pypdfium2 as pdfium
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel
from src.domain.pdf_parser.exceptions import PDFParsingException, PDFValidationError

from .markdown_export import export_markdown
from .models import PaperSection, ParserType, PdfContent

logger = logging.getLogger(__name__)

_HEADER_LABELS = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}
_CONTENT_LABELS = {
    DocItemLabel.TEXT,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CAPTION,
    DocItemLabel.CODE,
    DocItemLabel.FORMULA,
}


def extract_sections_from_document(doc: Any) -> list[PaperSection]:
    """Extract hierarchical sections with breadcrumbs from a Docling document.

    Walks the document tree via ``iterate_items`` and maintains a header stack so
    nested sections retain their full path (e.g. ``Methods > Training``).
    """
    sections: list[PaperSection] = []
    header_stack: list[tuple[int, str]] = []
    current: Optional[dict[str, Any]] = {"title": "Content", "content": "", "level": 1, "breadcrumb": None}

    def flush_current() -> None:
        nonlocal current
        if current and current["content"].strip():
            sections.append(
                PaperSection(
                    title=current["title"],
                    content=current["content"].strip(),
                    level=current["level"],
                    breadcrumb=current.get("breadcrumb"),
                )
            )

    for item, depth in doc.iterate_items():
        label = getattr(item, "label", None)
        text = getattr(item, "text", "").strip()
        if not text:
            continue

        if label in _HEADER_LABELS:
            flush_current()

            if label == DocItemLabel.TITLE:
                # Keep the document title as the breadcrumb root across same-level section headers.
                level = 0
            else:
                level = getattr(item, "level", None) or max(depth, 1)

            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, text))

            breadcrumb = " > ".join(title for _, title in header_stack)
            current = {
                "title": text,
                "content": "",
                "level": level,
                "breadcrumb": breadcrumb,
            }
        elif current is not None and label in _CONTENT_LABELS:
            current["content"] += text + "\n"

    flush_current()
    return sections


class DoclingParser:
    """Docling PDF parser for scientific document processing."""

    def __init__(
        self,
        max_pages: int,
        max_file_size_mb: int,
        do_ocr: bool = False,
        do_table_structure: bool = True,
        accelerator_device: AcceleratorDevice | str | None = None,
    ):
        """Initialize DocumentConverter with optimized pipeline options.

        :param max_pages: Maximum number of pages to process
        :param max_file_size_mb: Maximum file size in MB
        :param do_ocr: Enable OCR for scanned PDFs (default: False, very slow)
        :param do_table_structure: Extract table structures (default: True)
        :param accelerator_device: Device for Docling models ("cpu", "cuda", "mps", "auto");
            None keeps Docling's default, which also honors the ``DOCLING_DEVICE`` environment variable.
            ``PDFParserService`` and the factory do not pass it; use ``DOCLING_DEVICE`` there
        """
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions(
            do_table_structure=do_table_structure,
            do_ocr=do_ocr,  # Usually disabled for speed
            generate_picture_images=True,  # Needed to export figure PNGs in the Markdown export
        )
        if accelerator_device is not None:
            pipeline_options.accelerator_options = AcceleratorOptions(device=accelerator_device)

        self._converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
        self._warmed_up = False
        self.max_pages = max_pages
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _warm_up_models(self):
        """Pre-warm the models with a small dummy document to avoid cold start."""
        if not self._warmed_up:
            # This happens only once per DoclingParser instance
            self._warmed_up = True

    def _validate_pdf(self, pdf_path: Path) -> bool:
        """Comprehensive PDF validation including size and page limits.

        :param pdf_path: Path to PDF file
        :returns: True if PDF appears valid and within limits, False otherwise
        """
        try:
            # Check file exists and is not empty
            if pdf_path.stat().st_size == 0:
                logger.error(f"PDF file is empty: {pdf_path}")
                raise PDFValidationError(f"PDF file is empty: {pdf_path}")

            # Check file size limit
            file_size = pdf_path.stat().st_size
            if file_size > self.max_file_size_bytes:
                logger.warning(
                    f"PDF file size ({file_size / 1024 / 1024:.1f}MB) exceeds limit ({self.max_file_size_bytes / 1024 / 1024:.1f}MB), skipping processing"
                )
                raise PDFValidationError(
                    f"PDF file too large: {file_size / 1024 / 1024:.1f}MB > {self.max_file_size_bytes / 1024 / 1024:.1f}MB"
                )

            # Check if file starts with PDF header
            with open(pdf_path, "rb") as f:
                header = f.read(8)
                if not header.startswith(b"%PDF-"):
                    logger.error(f"File does not have PDF header: {pdf_path}")
                    raise PDFValidationError(f"File does not have PDF header: {pdf_path}")

            # Check page count limit
            pdf_doc = pdfium.PdfDocument(str(pdf_path))
            actual_pages = len(pdf_doc)
            pdf_doc.close()

            if actual_pages > self.max_pages:
                logger.warning(
                    f"PDF has {actual_pages} pages, exceeding limit of {self.max_pages} pages. Skipping processing to avoid performance issues."
                )
                raise PDFValidationError(f"PDF has too many pages: {actual_pages} > {self.max_pages}")

            return True

        except PDFValidationError:
            raise
        except Exception as e:
            logger.error(f"Error validating PDF {pdf_path}: {e}")
            raise PDFValidationError(f"Error validating PDF {pdf_path}: {e}")

    def parse_pdf(self, pdf_path: Path, markdown_output_dir: Path | None = None) -> Optional[PdfContent]:
        """Parse PDF using Docling parser.
        Limited to 20 pages to avoid memory issues with large papers.

        :param pdf_path: Path to PDF file
        :param markdown_output_dir: When set, also export ``{pdf_stem}.md`` and ``{pdf_stem}_assets/`` there
        :returns: PdfContent object or None if parsing failed
        """
        try:
            # Validate PDF first (includes size and page limits)
            self._validate_pdf(pdf_path)

            # Warm up models on first use
            self._warm_up_models()

            # Convert PDF using the modern API
            # Limit processing to avoid memory issues with large papers
            result = self._converter.convert(str(pdf_path), max_num_pages=self.max_pages, max_file_size=self.max_file_size_bytes)

            # Extract structured content
            doc = result.document
            sections = extract_sections_from_document(doc)

            markdown_path: Path | None = None
            artifacts_dir: Path | None = None
            if markdown_output_dir is not None:
                markdown_path, artifacts_dir = export_markdown(doc, pdf_path.stem, markdown_output_dir)

            # Focus on what arXiv API doesn't provide: structured full text content only
            return PdfContent(
                sections=sections,
                figures=[],  # Removed: basic metadata not useful
                tables=[],  # Removed: basic metadata not useful
                raw_text=doc.export_to_text(),
                references=[],
                parser_used=ParserType.DOCLING,
                metadata={"source": "docling", "note": "Content extracted from PDF, metadata comes from arXiv API"},
                markdown_path=markdown_path,
                artifacts_dir=artifacts_dir,
            )

        except PDFValidationError as e:
            # Handle size/page limit validation errors gracefully by returning None
            error_msg = str(e).lower()
            if "too large" in error_msg or "too many pages" in error_msg:
                logger.info(f"Skipping PDF processing due to size/page limits: {e}")
                return None
            else:
                # Re-raise other validation errors (corrupted files, etc.)
                raise
        except PDFParsingException:
            # Raised by the Markdown export with its own message; do not re-wrap it below.
            # Must come after PDFValidationError, which is a subclass.
            raise
        except Exception as e:
            logger.error(f"Failed to parse PDF with Docling: {e}")
            logger.error(f"PDF path: {pdf_path}")
            logger.error(f"PDF size: {pdf_path.stat().st_size} bytes")
            logger.error(f"Error type: {type(e).__name__}")

            # Add specific handling for common issues
            error_msg = str(e).lower()

            # Note: Page and size limit checks are now handled in _validate_pdf method

            if "not valid" in error_msg:
                logger.error("PDF appears to be corrupted or not a valid PDF file")
                raise PDFParsingException(f"PDF appears to be corrupted or invalid: {pdf_path}")
            elif "timeout" in error_msg:
                logger.error("PDF processing timed out - file may be too complex")
                raise PDFParsingException(f"PDF processing timed out: {pdf_path}")
            elif "memory" in error_msg or "ram" in error_msg:
                logger.error("Out of memory - PDF may be too large or complex")
                raise PDFParsingException(f"Out of memory processing PDF: {pdf_path}")
            elif "max_num_pages" in error_msg or "page" in error_msg:
                logger.error(f"PDF processing issue likely related to page limits (current limit: {self.max_pages} pages)")
                raise PDFParsingException(
                    f"PDF processing failed, possibly due to page limit ({self.max_pages} pages). Error: {e}"
                )
            else:
                raise PDFParsingException(f"Failed to parse PDF with Docling: {e}")
