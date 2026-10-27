import asyncio
import logging
from pathlib import Path
from typing import Optional

from src.domain.pdf_parser.exceptions import PDFParsingException, PDFValidationError

from .docling import DoclingParser
from .models import PdfContent

logger = logging.getLogger(__name__)


class PDFParserService:
    """Main PDF parsing service using Docling only."""

    def __init__(self, max_pages: int, max_file_size_mb: int, do_ocr: bool = False, do_table_structure: bool = True):
        """Initialize PDF parser service with configurable limits."""
        self.docling_parser = DoclingParser(
            max_pages=max_pages, max_file_size_mb=max_file_size_mb, do_ocr=do_ocr, do_table_structure=do_table_structure
        )

    async def parse_pdf(self, pdf_path: Path, markdown_output_dir: Path | None = None) -> Optional[PdfContent]:
        """Parse PDF using Docling parser only.

        :param pdf_path: Path to PDF file
        :param markdown_output_dir: When set, also export ``{pdf_stem}.md`` and ``{pdf_stem}_assets/`` there
        :returns: PdfContent object or None if parsing failed
        """
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise PDFValidationError(f"PDF file not found: {pdf_path}")

        try:
            result = await asyncio.to_thread(self.docling_parser.parse_pdf, pdf_path, markdown_output_dir)
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
