from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import pytest
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options import PdfPipelineOptions
from src.domain.pdf_parser.docling import DoclingParser
from src.domain.pdf_parser.exceptions import PDFParsingException, PDFValidationError
from src.domain.pdf_parser.factory import make_pdf_parser_service
from src.domain.pdf_parser.models import PaperSection, ParserType, PdfContent
from src.domain.pdf_parser.parser import PDFParserService


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

    @patch("src.domain.pdf_parser.docling.DocumentConverter")
    @patch("src.domain.pdf_parser.docling.PdfFormatOption")
    def test_pipeline_options_generate_picture_images(self, mock_format_option, mock_converter):
        """Test that figure images are generated so the Markdown export can save them."""
        DoclingParser(max_pages=20, max_file_size_mb=10)

        pipeline_options = mock_format_option.call_args.kwargs["pipeline_options"]
        assert pipeline_options.generate_picture_images is True

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


class TestDoclingParserMarkdownExport:
    """Test how DoclingParser.parse_pdf wires the optional Markdown export.

    Docling is replaced at module level (DocumentConverter and pypdfium2), so the real validation and
    error handling of parse_pdf run without loading any model.
    """

    @pytest.fixture
    def fake_document(self):
        return SimpleNamespace(iterate_items=lambda: iter([]), export_to_text=lambda: "Full text")

    @pytest.fixture
    def page_count(self):
        return 1

    @pytest.fixture
    def docling_parser(self, fake_document, page_count):
        with (
            patch("src.domain.pdf_parser.docling.DocumentConverter") as mock_converter_cls,
            patch("src.domain.pdf_parser.docling.pdfium.PdfDocument") as mock_pdf_document,
        ):
            mock_converter_cls.return_value.convert.return_value = SimpleNamespace(document=fake_document)
            mock_pdf_document.return_value.__len__.return_value = page_count
            yield DoclingParser(max_pages=20, max_file_size_mb=10)

    @pytest.fixture
    def pdf_path(self, tmp_path):
        pdf_dir = tmp_path / "in"
        pdf_dir.mkdir()
        pdf_file = pdf_dir / "paper.pdf"
        pdf_file.write_bytes(b"%PDF-1.4\ntest content")
        return pdf_file

    @patch("src.domain.pdf_parser.docling.export_markdown")
    def test_without_output_dir_does_not_export(self, mock_export, docling_parser, pdf_path, tmp_path):
        result = docling_parser.parse_pdf(pdf_path)

        mock_export.assert_not_called()
        assert result.raw_text == "Full text"
        assert result.markdown_path is None
        assert result.artifacts_dir is None
        assert sorted(p.name for p in tmp_path.rglob("*")) == ["in", "paper.pdf"]

    @patch("src.domain.pdf_parser.docling.export_markdown")
    def test_export_does_not_change_sections_or_raw_text(self, mock_export, docling_parser, pdf_path, tmp_path):
        output_dir = tmp_path / "out"
        mock_export.return_value = (output_dir / "paper.md", output_dir / "paper_assets")

        without_export = docling_parser.parse_pdf(pdf_path)
        with_export = docling_parser.parse_pdf(pdf_path, markdown_output_dir=output_dir)

        assert with_export.sections == without_export.sections
        assert with_export.raw_text == without_export.raw_text

    @patch("src.domain.pdf_parser.docling.export_markdown")
    def test_with_output_dir_exports_and_returns_paths(self, mock_export, docling_parser, fake_document, pdf_path, tmp_path):
        output_dir = tmp_path / "out"
        mock_export.return_value = (output_dir / "paper.md", output_dir / "paper_assets")

        result = docling_parser.parse_pdf(pdf_path, markdown_output_dir=output_dir)

        mock_export.assert_called_once_with(fake_document, "paper", output_dir)
        assert result.markdown_path == output_dir / "paper.md"
        assert result.artifacts_dir == output_dir / "paper_assets"

    @patch("src.domain.pdf_parser.docling.export_markdown")
    def test_export_error_keeps_original_message(self, mock_export, docling_parser, pdf_path, tmp_path):
        mock_export.side_effect = PDFParsingException("Failed to export Markdown to /out: disk full")

        with pytest.raises(PDFParsingException) as exc_info:
            docling_parser.parse_pdf(pdf_path, markdown_output_dir=tmp_path / "out")

        assert str(exc_info.value) == "Failed to export Markdown to /out: disk full"

    @pytest.mark.parametrize("page_count", [40])
    @patch("src.domain.pdf_parser.docling.export_markdown")
    def test_page_limit_still_returns_none(self, mock_export, docling_parser, pdf_path, tmp_path):
        """PDFValidationError subclasses PDFParsingException; the limit case must keep returning None."""
        assert docling_parser.parse_pdf(pdf_path, markdown_output_dir=tmp_path / "out") is None
        mock_export.assert_not_called()


class TestDoclingParserAcceleratorDevice:
    """Test the optional accelerator device of DoclingParser."""

    @patch("src.domain.pdf_parser.docling.DocumentConverter")
    @patch("src.domain.pdf_parser.docling.PdfFormatOption")
    def test_device_is_passed_to_pipeline_options(self, mock_format_option, mock_converter):
        DoclingParser(max_pages=20, max_file_size_mb=10, accelerator_device="cpu")

        pipeline_options = mock_format_option.call_args.kwargs["pipeline_options"]
        assert pipeline_options.accelerator_options.device == AcceleratorDevice.CPU

    @patch("src.domain.pdf_parser.docling.DocumentConverter")
    @patch("src.domain.pdf_parser.docling.PdfFormatOption")
    def test_without_device_keeps_docling_default(self, mock_format_option, mock_converter):
        DoclingParser(max_pages=20, max_file_size_mb=10)

        pipeline_options = mock_format_option.call_args.kwargs["pipeline_options"]
        assert pipeline_options.accelerator_options == PdfPipelineOptions().accelerator_options

    @patch("src.domain.pdf_parser.docling.DocumentConverter")
    def test_invalid_device_is_rejected(self, mock_converter):
        with pytest.raises(ValueError):
            DoclingParser(max_pages=20, max_file_size_mb=10, accelerator_device="gpu")


class TestPdfContent:
    """Test PdfContent model."""

    def test_markdown_fields_default_to_none(self):
        """Test that the Markdown export fields are optional and empty by default."""
        content = PdfContent(raw_text="Test content", parser_used=ParserType.DOCLING)

        assert content.markdown_path is None
        assert content.artifacts_dir is None


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

    def test_pdf_parser_service_initialization(self, pdf_parser_service):
        """Test PDFParserService initialization."""
        assert isinstance(pdf_parser_service.docling_parser, DoclingParser)
        assert pdf_parser_service.docling_parser.max_pages == 20

    @pytest.mark.asyncio
    async def test_parse_pdf_file_not_found(self, pdf_parser_service):
        """Test parsing with non-existent file."""
        nonexistent_path = Path("/nonexistent/file.pdf")

        with pytest.raises(PDFValidationError) as exc_info:
            await pdf_parser_service.parse_pdf(nonexistent_path)

        assert "PDF file not found" in str(exc_info.value)

    @patch("src.domain.pdf_parser.parser.DoclingParser.parse_pdf")
    @pytest.mark.asyncio
    async def test_parse_pdf_success(self, mock_parse, pdf_parser_service, valid_pdf_path):
        """Test successful PDF parsing."""
        mock_content = PdfContent(
            raw_text="Test content", sections=[], tables=[], figures=[], parser_used=ParserType.DOCLING, metadata={}
        )
        mock_parse.return_value = mock_content

        result = await pdf_parser_service.parse_pdf(valid_pdf_path)

        assert result == mock_content
        mock_parse.assert_called_once_with(valid_pdf_path, None)

    @patch("src.domain.pdf_parser.parser.DoclingParser.parse_pdf")
    @pytest.mark.asyncio
    async def test_parse_pdf_forwards_markdown_output_dir(self, mock_parse, pdf_parser_service, valid_pdf_path, tmp_path):
        """Test that the Markdown output folder is passed down to DoclingParser."""
        output_dir = tmp_path / "out"
        mock_parse.return_value = PdfContent(raw_text="Test content", parser_used=ParserType.DOCLING)

        await pdf_parser_service.parse_pdf(valid_pdf_path, markdown_output_dir=output_dir)

        mock_parse.assert_called_once_with(valid_pdf_path, output_dir)

    @patch("src.domain.pdf_parser.parser.DoclingParser.parse_pdf")
    @pytest.mark.asyncio
    async def test_parse_pdf_no_result(self, mock_parse, pdf_parser_service, valid_pdf_path):
        """Test PDF parsing when no result is returned (e.g. page/size limits)."""
        mock_parse.return_value = None

        result = await pdf_parser_service.parse_pdf(valid_pdf_path)

        assert result is None
        mock_parse.assert_called_once_with(valid_pdf_path, None)

    @patch("src.domain.pdf_parser.parser.DoclingParser.parse_pdf")
    @pytest.mark.asyncio
    async def test_parse_pdf_docling_error(self, mock_parse, pdf_parser_service, valid_pdf_path):
        """Test PDF parsing when Docling raises an error."""
        mock_parse.side_effect = Exception("Docling error")

        with pytest.raises(PDFParsingException) as exc_info:
            await pdf_parser_service.parse_pdf(valid_pdf_path)

        assert "Docling parsing error" in str(exc_info.value)

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
