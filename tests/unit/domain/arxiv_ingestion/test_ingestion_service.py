from unittest.mock import MagicMock, patch

from src.domain.arxiv.paper import ArxivPaper
from src.domain.arxiv_ingestion.paper_indexer import PaperIndexer
from src.domain.arxiv_ingestion.paper_persist import PaperPersist
from src.domain.arxiv_ingestion.schemas import DownloadResults
from src.domain.arxiv_ingestion.service import ArxivIngestionService
from src.domain.pdf_parser.artifacts import write_parsed_paper_artifacts
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


def _ingestion_service(metadata_fetcher: MagicMock, database: MagicMock) -> ArxivIngestionService:
    return ArxivIngestionService(
        arxiv_client=MagicMock(),
        pdf_parser=MagicMock(),
        database=database,
        metadata_fetcher=metadata_fetcher,
        paper_persist=PaperPersist(),
        paper_indexer=PaperIndexer(database),
    )


@patch("src.domain.arxiv_ingestion.paper_persist.PaperRepository")
def test_persist_papers_loads_sidecars_and_stores_papers(mock_repo_cls, tmp_path) -> None:
    paper = _paper()
    artifact = write_parsed_paper_artifacts(tmp_path / "arxiv_md", _parsed_paper(paper))
    database = MagicMock()
    session = MagicMock()
    database.get_session.return_value.__enter__.return_value = session
    mock_repo_cls.return_value.upsert.return_value = MagicMock()
    service = _ingestion_service(MagicMock(), database)

    result = service.persist_papers(
        DownloadResults(papers=[paper], date="20240101"),
        {"pdfs_parsed": 1, "artifacts": {paper.arxiv_id: artifact}, "errors": []},
    )

    assert result.papers_stored == 1
    paper_create = mock_repo_cls.return_value.upsert.call_args[0][0]
    assert paper_create.raw_text == "Parsed text"
    session.commit.assert_called_once()


@patch("src.domain.arxiv_ingestion.paper_persist.PaperRepository")
def test_persist_papers_stores_metadata_when_sidecar_is_missing(mock_repo_cls, tmp_path) -> None:
    paper = _paper()
    database = MagicMock()
    session = MagicMock()
    database.get_session.return_value.__enter__.return_value = session
    mock_repo_cls.return_value.upsert.return_value = MagicMock()
    service = _ingestion_service(MagicMock(), database)

    result = service.persist_papers(
        DownloadResults(papers=[paper], date="20240101"),
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

    assert result.papers_stored == 1
    assert result.errors[0].startswith(f"Failed to load parsed artifact for {paper.arxiv_id}")
    paper_create = mock_repo_cls.return_value.upsert.call_args[0][0]
    assert paper_create.pdf_processed is False
    session.commit.assert_called_once()
