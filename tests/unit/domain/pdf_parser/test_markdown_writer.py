from unittest.mock import patch

import pytest
from src.domain.pdf_parser.markdown_writer import MarkdownWriter
from src.domain.pdf_parser.models import ArxivMetadata, ParsedPaper, ParserType, PdfContent


@pytest.fixture
def parsed_paper() -> ParsedPaper:
    metadata = ArxivMetadata(
        arxiv_id="2024.0001v1",
        title="Test Paper",
        authors=["Test Author"],
        abstract="Test abstract",
        categories=["cs.AI"],
        published_date="2024-01-01T00:00:00Z",
        pdf_url="https://arxiv.org/pdf/2024.0001v1",
    )
    return ParsedPaper(
        arxiv_metadata=metadata,
        pdf_content=PdfContent(
            raw_text="Parsed text",
            markdown="# Test Paper",
            parser_used=ParserType.DOCLING,
        ),
    )


class TestMarkdownWriter:
    @patch("src.domain.pdf_parser.markdown_writer._atomic_write_text")
    def test_write_all_records_artifact_failure(self, mock_atomic_write, parsed_paper, tmp_path):
        mock_atomic_write.side_effect = OSError("disk full")
        writer = MarkdownWriter(tmp_path)

        result = writer.write_all({parsed_paper.arxiv_metadata.arxiv_id: parsed_paper})

        assert result.artifacts == {}
        assert result.errors == ["Failed to write parsed artifacts for 2024.0001v1: disk full"]
        mock_atomic_write.assert_called_once()

    @patch("src.domain.pdf_parser.markdown_writer._atomic_write_text")
    def test_write_all_preserves_initial_errors(self, mock_atomic_write, parsed_paper, tmp_path):
        writer = MarkdownWriter(tmp_path)

        result = writer.write_all(
            {parsed_paper.arxiv_metadata.arxiv_id: parsed_paper},
            errors=["Parse error for 2024.0002v1: broken"],
        )

        assert parsed_paper.arxiv_metadata.arxiv_id in result.artifacts
        assert result.errors == ["Parse error for 2024.0002v1: broken"]
