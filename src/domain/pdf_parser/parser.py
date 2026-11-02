import logging
from pathlib import Path
from typing import Any, Optional

from .content_extractor import ContentExtractor
from .docling import DoclingParser
from .models import ArxivMetadata
from .pdf_to_markdown_pipeline import PdfToMarkdownPipeline

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
        heading_hierarchy_enabled: bool = True,
    ):
        """Initialize PDF parser service with configurable limits."""
        self.do_table_structure = do_table_structure
        self.do_figure_extraction = do_figure_extraction
        self.docling_parser = DoclingParser(
            max_pages=max_pages,
            max_file_size_mb=max_file_size_mb,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            do_figure_extraction=do_figure_extraction,
            heading_hierarchy_enabled=heading_hierarchy_enabled,
        )
        self.content_extractor = ContentExtractor(
            do_table_structure=do_table_structure,
            do_figure_extraction=do_figure_extraction,
        )

    async def parse_pdfs_to_markdown(
        self,
        papers: list[ArxivMetadata],
        pdf_paths_by_id: dict[str, str],
        output_dir: Path,
        max_concurrent_parsing: int = 1,
        date: Optional[str] = None,
    ) -> dict[str, Any]:
        """Parse downloaded PDFs and write shared Markdown and JSON artifacts."""
        pipeline = PdfToMarkdownPipeline(
            docling_parser=self.docling_parser,
            content_extractor=self.content_extractor,
            output_dir=output_dir,
            max_concurrent_parsing=max_concurrent_parsing,
            extract_parsing_artifacts=self.do_figure_extraction or self.do_table_structure,
        )
        return await pipeline.run(papers, pdf_paths_by_id, date=date)
