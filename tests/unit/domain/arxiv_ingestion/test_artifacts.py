import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from src.domain.arxiv_ingestion.artifacts import (
    artifact_stem,
    load_parsed_paper_artifact,
    write_parsed_paper_artifacts,
)
from src.domain.pdf_parser.models import ArxivMetadata, ParsedPaper, ParserType, PdfContent


@pytest.fixture
def parsed_paper() -> ParsedPaper:
    return ParsedPaper(
        arxiv_metadata=ArxivMetadata(
            arxiv_id="hep-th/9901001v1",
            title="Test Paper",
            authors=["Test Author"],
            abstract="Test abstract",
            categories=["hep-th"],
            published_date="1999-01-01T00:00:00Z",
            pdf_url="https://arxiv.org/pdf/hep-th/9901001v1",
        ),
        pdf_content=PdfContent(
            raw_text="Structured paper content",
            markdown="# Test Paper\n\nStructured paper content",
            parser_used=ParserType.DOCLING,
        ),
    )


def test_write_and_load_parsed_paper_artifacts(tmp_path, parsed_paper) -> None:
    artifact = write_parsed_paper_artifacts(tmp_path, parsed_paper)

    assert artifact_stem("hep-th/9901001v1") in artifact["markdown_path"]
    assert "/" not in artifact_stem("hep-th/9901001v1")
    assert parsed_paper.pdf_content is not None
    assert Path(artifact["markdown_path"]).read_text(encoding="utf-8") == parsed_paper.pdf_content.markdown

    sidecar = json.loads(Path(artifact["sidecar_path"]).read_text(encoding="utf-8"))
    assert "markdown" not in sidecar["pdf_content"]

    loaded = load_parsed_paper_artifact(artifact["sidecar_path"])
    assert loaded.arxiv_metadata.arxiv_id == parsed_paper.arxiv_metadata.arxiv_id
    assert loaded.pdf_content is not None
    assert loaded.pdf_content.raw_text == "Structured paper content"
    assert loaded.pdf_content.markdown is None


def test_load_parsed_paper_artifact_rejects_corrupt_json(tmp_path) -> None:
    sidecar_path = tmp_path / "corrupt.json"
    sidecar_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_parsed_paper_artifact(sidecar_path)
