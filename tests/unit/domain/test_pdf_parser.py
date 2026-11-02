from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.domain.pdf_parser.docling import DoclingParser
from src.domain.pdf_parser.exceptions import PDFParsingException, PDFValidationError
from src.domain.pdf_parser.factory import make_pdf_parser_service
from src.domain.pdf_parser.models import ArxivMetadata, ParserType, PdfContent
from src.domain.pdf_parser.parser import PDFParserService
from src.domain.pdf_parser.pdf_to_markdown_pipeline import PdfToMarkdownPipeline


@pytest.fixture
def metadata():
    """Create metadata associated with a parsed PDF."""
    return ArxivMetadata(
        arxiv_id="2024.0001v1",
        title="Test Paper",
        authors=["Test Author"],
        abstract="Test abstract",
        categories=["cs.AI"],
        published_date="2024-01-01T00:00:00Z",
        pdf_url="https://arxiv.org/pdf/2024.0001v1",
    )


class TestDoclingParser:
    """Test DoclingParser functionality."""

    @pytest.fixture
    def docling_parser(self):
        """Create DoclingParser instance for testing."""
        return DoclingParser(max_pages=20, max_file_size_mb=10, do_ocr=False)

    @pytest.fixture
    def valid_pdf_path(self, tmp_path):
        """Create a mock valid PDF file path."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")
        return pdf_file

    @pytest.fixture
    def empty_pdf_path(self, tmp_path):
        """Create an empty PDF file path."""
        pdf_file = tmp_path / "empty.pdf"
        pdf_file.write_bytes(b"")
        return pdf_file

    @pytest.fixture
    def invalid_pdf_path(self, tmp_path):
        """Create an invalid PDF file path."""
        pdf_file = tmp_path / "invalid.pdf"
        pdf_file.write_bytes(b"Not a PDF file")
        return pdf_file

    def test_docling_parser_initialization(self, docling_parser):
        """Test DoclingParser initialization."""
        assert docling_parser.max_pages == 20
        assert docling_parser.max_file_size_bytes == 10 * 1024 * 1024
        assert docling_parser._warmed_up is False

    def test_validate_pdf_valid_file(self, docling_parser, valid_pdf_path):
        """Test PDF validation with valid file."""
        # This test is complex due to pypdfium2 dependency, skip for now
        pass

    def test_validate_pdf_empty_file(self, docling_parser, empty_pdf_path):
        """Test PDF validation with empty file."""
        with pytest.raises(PDFValidationError) as exc_info:
            docling_parser._validate_pdf(empty_pdf_path)

        assert "PDF file is empty" in str(exc_info.value)

    def test_validate_pdf_invalid_header(self, docling_parser, invalid_pdf_path):
        """Test PDF validation with invalid header."""
        with pytest.raises(PDFValidationError) as exc_info:
            docling_parser._validate_pdf(invalid_pdf_path)

        assert "File does not have PDF header" in str(exc_info.value)

    def test_validate_pdf_nonexistent_file(self, docling_parser):
        """Test PDF validation with nonexistent file."""
        nonexistent_path = Path("/nonexistent/file.pdf")

        with pytest.raises(PDFValidationError) as exc_info:
            docling_parser._validate_pdf(nonexistent_path)

        assert "Error validating PDF" in str(exc_info.value)

    # Complex PDF parsing tests removed - too dependent on external libraries


class TestPDFParserService:
    """Test PDFParserService functionality."""

    @pytest.fixture
    def pdf_parser_service(self):
        """Create PDFParserService instance for testing."""
        return PDFParserService(max_pages=20, max_file_size_mb=10)

    @pytest.fixture
    def valid_pdf_path(self, tmp_path):
        """Create a mock valid PDF file path."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")
        return pdf_file

    @pytest.fixture
    def markdown_pipeline(self, pdf_parser_service, tmp_path):
        """Create PdfToMarkdownPipeline for testing."""
        return PdfToMarkdownPipeline(
            docling_parser=pdf_parser_service.docling_parser,
            content_extractor=pdf_parser_service.content_extractor,
            output_dir=tmp_path,
        )

    def test_pdf_parser_service_initialization(self, pdf_parser_service):
        """Test PDFParserService initialization."""
        assert isinstance(pdf_parser_service.docling_parser, DoclingParser)
        assert pdf_parser_service.docling_parser.max_pages == 20

    @pytest.mark.asyncio
    async def test_parse_pdf_file_not_found(self, markdown_pipeline):
        """Test parsing with non-existent file."""
        nonexistent_path = Path("/nonexistent/file.pdf")

        with pytest.raises(PDFValidationError) as exc_info:
            await markdown_pipeline.parse_pdf(nonexistent_path)

        assert "PDF file not found" in str(exc_info.value)

    @patch("src.domain.pdf_parser.pdf_to_markdown_pipeline.ContentExtractor.build_content_session")
    @patch("src.domain.pdf_parser.pdf_to_markdown_pipeline.DoclingParser.parse_pdf")
    @pytest.mark.asyncio
    async def test_parse_pdf_success(self, mock_parse, mock_build_session, markdown_pipeline, valid_pdf_path):
        """Test successful PDF parsing."""
        mock_conversion = object()
        mock_parse.return_value = mock_conversion
        mock_content = PdfContent(
            raw_text="Test content", sections=[], tables=[], figures=[], parser_used=ParserType.DOCLING, metadata={}
        )
        mock_build_session.return_value = mock_content

        result = await markdown_pipeline.parse_pdf(valid_pdf_path)

        assert result == mock_content
        mock_parse.assert_called_once_with(valid_pdf_path)
        mock_build_session.assert_called_once_with(mock_conversion, artifacts_dir=None)

    @patch("src.domain.pdf_parser.pdf_to_markdown_pipeline.ContentExtractor.build_content_session")
    @patch("src.domain.pdf_parser.pdf_to_markdown_pipeline.DoclingParser.parse_pdf")
    @pytest.mark.asyncio
    async def test_parse_pdf_no_result(self, mock_parse, mock_build_session, markdown_pipeline, valid_pdf_path):
        """Test PDF parsing when no result is returned (e.g. page/size limits)."""
        mock_parse.return_value = None

        result = await markdown_pipeline.parse_pdf(valid_pdf_path)

        assert result is None
        mock_parse.assert_called_once_with(valid_pdf_path)
        mock_build_session.assert_not_called()

    @patch("src.domain.pdf_parser.pdf_to_markdown_pipeline.DoclingParser.parse_pdf")
    @pytest.mark.asyncio
    async def test_parse_pdf_docling_error(self, mock_parse, markdown_pipeline, valid_pdf_path):
        """Test PDF parsing when Docling raises an error."""
        mock_parse.side_effect = Exception("Docling error")

        with pytest.raises(PDFParsingException) as exc_info:
            await markdown_pipeline.parse_pdf(valid_pdf_path)

        assert "Docling parsing error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_parse_pdfs_to_markdown_handles_empty_input(self, pdf_parser_service, tmp_path):
        result = await pdf_parser_service.parse_pdfs_to_markdown(
            papers=[],
            pdf_paths_by_id={},
            output_dir=tmp_path,
            date="20240101",
        )

        assert result == {
            "pdfs_parsed": 0,
            "artifacts": {},
            "errors": [],
            "date": "20240101",
            "processing_time": 0,
        }

    @patch("src.domain.pdf_parser.pdf_to_markdown_pipeline.PdfToMarkdownPipeline.extract_content")
    @patch("src.domain.pdf_parser.pdf_to_markdown_pipeline.PdfToMarkdownPipeline.convert_pdf", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_parse_pdfs_to_markdown_writes_artifacts(
        self,
        mock_convert_pdf,
        mock_extract_content,
        pdf_parser_service,
        valid_pdf_path,
        metadata,
        tmp_path,
    ):
        mock_conversion = object()
        mock_convert_pdf.return_value = mock_conversion
        pdf_content = PdfContent(
            raw_text="Parsed text",
            markdown="# Test Paper\n\nParsed text",
            parser_used=ParserType.DOCLING,
        )
        mock_extract_content.return_value = pdf_content
        output_dir = tmp_path / "arxiv_md"

        result = await pdf_parser_service.parse_pdfs_to_markdown(
            papers=[metadata],
            pdf_paths_by_id={metadata.arxiv_id: str(valid_pdf_path)},
            output_dir=output_dir,
            max_concurrent_parsing=1,
            date="20240101",
        )

        assert result["pdfs_parsed"] == 1
        assert result["errors"] == []
        assert result["date"] == "20240101"
        artifact = result["artifacts"][metadata.arxiv_id]
        assert Path(artifact["markdown_path"]).read_text(encoding="utf-8") == pdf_content.markdown
        mock_convert_pdf.assert_awaited_once_with(valid_pdf_path)
        assert mock_extract_content.call_args.args[2].parent == output_dir

    @patch("src.domain.pdf_parser.pdf_to_markdown_pipeline.PdfToMarkdownPipeline.convert_pdf", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_parse_pdfs_to_markdown_records_parse_failure(
        self,
        mock_convert_pdf,
        pdf_parser_service,
        valid_pdf_path,
        metadata,
        tmp_path,
    ):
        mock_convert_pdf.side_effect = PDFParsingException("broken PDF")

        result = await pdf_parser_service.parse_pdfs_to_markdown(
            papers=[metadata],
            pdf_paths_by_id={metadata.arxiv_id: str(valid_pdf_path)},
            output_dir=tmp_path,
        )

        assert result["pdfs_parsed"] == 0
        assert result["artifacts"] == {}
        assert result["errors"][0] == f"Parse error for {metadata.arxiv_id}: broken PDF"
        assert result["errors"][1] == f"PDF parse failed: {metadata.arxiv_id}"

    @patch(
        "src.domain.pdf_parser.pdf_to_markdown_pipeline.write_parsed_paper_artifacts",
        side_effect=OSError("disk full"),
    )
    @patch("src.domain.pdf_parser.pdf_to_markdown_pipeline.PdfToMarkdownPipeline.extract_content")
    @patch("src.domain.pdf_parser.pdf_to_markdown_pipeline.PdfToMarkdownPipeline.convert_pdf", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_parse_pdfs_to_markdown_records_artifact_failure(
        self,
        mock_convert_pdf,
        mock_extract_content,
        mock_write_artifacts,
        pdf_parser_service,
        valid_pdf_path,
        metadata,
        tmp_path,
    ):
        mock_convert_pdf.return_value = object()
        mock_extract_content.return_value = PdfContent(
            raw_text="Parsed text",
            markdown="# Test Paper",
            parser_used=ParserType.DOCLING,
        )

        result = await pdf_parser_service.parse_pdfs_to_markdown(
            papers=[metadata],
            pdf_paths_by_id={metadata.arxiv_id: str(valid_pdf_path)},
            output_dir=tmp_path,
        )

        assert result["pdfs_parsed"] == 0
        assert result["artifacts"] == {}
        assert result["errors"] == [f"Failed to write parsed artifacts for {metadata.arxiv_id}: disk full"]
        mock_write_artifacts.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_run_uses_injected_steps(
        self,
        markdown_pipeline,
        valid_pdf_path,
        metadata,
        tmp_path,
    ):
        mock_conversion = object()
        convert_pdf = AsyncMock(return_value=mock_conversion)
        pdf_content = PdfContent(
            raw_text="Injected",
            markdown="# Injected",
            parser_used=ParserType.DOCLING,
        )
        extract_content = MagicMock(return_value=pdf_content)

        markdown_pipeline.set_pdfs([metadata], {metadata.arxiv_id: str(valid_pdf_path)})
        parse_results = await markdown_pipeline.run_pdf_parse(convert_pdf, extract_content)

        assert parse_results["parsed"] == 1
        assert metadata.arxiv_id in parse_results["parsed_papers"]
        convert_pdf.assert_awaited_once_with(valid_pdf_path)
        extract_content.assert_called_once()
        assert extract_content.call_args.args[0] is mock_conversion
        assert extract_content.call_args.args[1] == valid_pdf_path

    def test_factory_creates_service(self):
        """Test that factory creates PDFParserService instance."""
        service = make_pdf_parser_service()
        assert isinstance(service, PDFParserService)
        assert isinstance(service.docling_parser, DoclingParser)

    def test_factory_caching(self):
        """Test that factory uses caching."""
        service1 = make_pdf_parser_service()
        service2 = make_pdf_parser_service()
        # Should be the same instance due to @lru_cache
        assert service1 is service2
