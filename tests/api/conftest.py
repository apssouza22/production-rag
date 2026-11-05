from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from src.main import app


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    """Async backend for testing."""
    return "asyncio"


@pytest.fixture
async def client():
    """HTTP client for API testing with mocked services."""
    mock_db = MagicMock()
    mock_db.get_session.return_value.__enter__.return_value = MagicMock()
    mock_db.get_session.return_value.__exit__.return_value = None
    mock_db.teardown = MagicMock()

    # Mock database startup and session to prevent real connections
    with (
        patch("src.main.make_database", return_value=mock_db),
        patch("src.domain.db.interfaces.postgresql.PostgreSQLDatabase.startup") as mock_startup,
        patch("src.domain.db.interfaces.postgresql.PostgreSQLDatabase.get_session") as mock_get_session,
        patch("src.main.make_hybrid_rrf_search") as mock_paper_search,
        patch("src.main.make_kg_indexing_service", return_value=None),
        patch("src.main.make_dual_level_kg_retrieval_service", return_value=None),
        patch("src.main.make_text_to_sql_service", return_value=MagicMock()),
        patch("src.main.make_knowledge_router_service", return_value=MagicMock()),
        patch("src.domain.arxiv.factory.make_arxiv_client") as mock_arxiv,
        patch("src.domain.pdf_parser.factory.make_pdf_parser_service") as mock_pdf,
        patch("src.domain.paper.repository.PaperRepository.get_by_arxiv_id") as mock_get_by_id,
    ):
        # Mock startup to do nothing
        mock_startup.return_value = None

        # Mock get_session to return a mock session
        mock_session = MagicMock()
        mock_get_session.return_value.__enter__.return_value = mock_session
        mock_get_session.return_value.__exit__.return_value = None

        # Mock repository methods to return None (not found) by default
        mock_get_by_id.return_value = None

        # Set up other mock return values
        mock_paper = MagicMock()
        mock_paper.health_check.return_value = True
        mock_paper.get_index_stats.return_value = {
            "index_name": "arxiv-papers-chunks",
            "document_count": 0,
            "exists": True,
        }
        mock_paper.search_unified.return_value = {"total": 0, "hits": []}
        mock_paper_search.return_value = mock_paper
        mock_arxiv.return_value = AsyncMock()
        mock_pdf.return_value = AsyncMock()

        async with LifespanManager(app) as manager:
            async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
                yield client
