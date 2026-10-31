from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.domain.arxiv.paper import ArxivPaper
from src.domain.arxiv_ingestion.artifacts import write_parsed_paper_artifacts
from src.domain.arxiv_ingestion.fetching import parse_pdfs_to_markdown, persist_papers
from src.domain.pdf_parser.models import ArxivMetadata, ParsedPaper, ParserType, PdfContent


def _paper() -> ArxivPaper:
    return ArxivPaper(
        arxiv_id="2024.0001v1",
        title="Test Paper",
        authors=["Test Author"],
        abstract="Test abstract",
        categories=["cs.AI"],
        published_date="2024-01-01T00:00:00Z",
        pdf_url="https://arxiv.org/pdf/2024.0001v1",
    )


def _parsed_paper(paper: ArxivPaper) -> ParsedPaper:
    return ParsedPaper(
        arxiv_metadata=ArxivMetadata(**paper.model_dump()),
        pdf_content=PdfContent(
            raw_text="Parsed text",
            markdown="# Test Paper\n\nParsed text",
            parser_used=ParserType.DOCLING,
        ),
    )


def test_parse_pdfs_to_markdown_writes_artifacts_without_database_session(tmp_path) -> None:
    paper = _paper()
    pdf_cache_dir = tmp_path / "arxiv_pdfs"
    pdf_path = pdf_cache_dir / f"{paper.arxiv_id}.pdf"
    metadata_fetcher = MagicMock()
    metadata_fetcher.pdf_cache_dir = pdf_cache_dir
    metadata_fetcher.parse_papers = AsyncMock(
        return_value={
            "parsed": 1,
            "parsed_papers": {paper.arxiv_id: _parsed_paper(paper)},
            "errors": [],
            "parse_failures": [],
        }
    )
    database = MagicMock()
    download_results = {
        "papers": [paper.model_dump()],
        "downloaded_paths": {paper.arxiv_id: str(pdf_path)},
        "date": "20240101",
    }

    with patch(
        "src.domain.arxiv_ingestion.fetching.get_cached_services",
        return_value=(MagicMock(), MagicMock(), database, metadata_fetcher, MagicMock()),
    ):
        result = parse_pdfs_to_markdown(download_results)

    assert result["pdfs_parsed"] == 1
    assert paper.arxiv_id in result["artifacts"]
    assert Path(result["artifacts"][paper.arxiv_id]["markdown_path"]).read_text(encoding="utf-8").startswith("# Test Paper")
    database.get_session.assert_not_called()
    metadata_fetcher.parse_papers.assert_awaited_once_with(
        papers=[paper],
        downloaded_paths={paper.arxiv_id: str(pdf_path)},
        artifacts_dir=tmp_path / "arxiv_md",
    )


def test_persist_papers_loads_sidecars_and_stores_papers(tmp_path) -> None:
    paper = _paper()
    artifact = write_parsed_paper_artifacts(tmp_path / "arxiv_md", _parsed_paper(paper))
    database = MagicMock()
    session = MagicMock()
    database.get_session.return_value.__enter__.return_value = session
    metadata_fetcher = MagicMock()
    metadata_fetcher.store_papers.return_value = 1

    with patch(
        "src.domain.arxiv_ingestion.fetching.get_cached_services",
        return_value=(MagicMock(), MagicMock(), database, metadata_fetcher, MagicMock()),
    ):
        result = persist_papers(
            {"papers": [paper.model_dump()], "date": "20240101"},
            {"pdfs_parsed": 1, "artifacts": {paper.arxiv_id: artifact}, "errors": []},
        )

    assert result["papers_stored"] == 1
    stored_papers = metadata_fetcher.store_papers.call_args.args[1]
    assert stored_papers[paper.arxiv_id].pdf_content.raw_text == "Parsed text"
    metadata_fetcher.store_papers.assert_called_once_with([paper], stored_papers, session)


def test_persist_papers_stores_metadata_when_sidecar_is_missing(tmp_path) -> None:
    paper = _paper()
    database = MagicMock()
    session = MagicMock()
    database.get_session.return_value.__enter__.return_value = session
    metadata_fetcher = MagicMock()
    metadata_fetcher.store_papers.return_value = 1

    with patch(
        "src.domain.arxiv_ingestion.fetching.get_cached_services",
        return_value=(MagicMock(), MagicMock(), database, metadata_fetcher, MagicMock()),
    ):
        result = persist_papers(
            {"papers": [paper.model_dump()], "date": "20240101"},
            {
                "pdfs_parsed": 1,
                "artifacts": {
                    paper.arxiv_id: {
                        "markdown_path": str(tmp_path / "missing.md"),
                        "sidecar_path": str(tmp_path / "missing.json"),
                    }
                },
                "errors": [],
            },
        )

    assert result["papers_stored"] == 1
    assert result["errors"][0].startswith(f"Failed to load parsed artifact for {paper.arxiv_id}")
    metadata_fetcher.store_papers.assert_called_once_with([paper], {}, session)
