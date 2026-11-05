import logging
from functools import lru_cache
from typing import Any, Tuple

from src.domain.arxiv.factory import make_arxiv_client
from src.domain.db.factory import make_database
from src.domain.pdf_parser.factory import make_pdf_parser_service
from src.domain.rrf.factory import make_paper_rrf_search

from .service import make_arxiv_ingestion_service

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cached_services() -> Tuple[Any, Any, Any, Any, Any]:
    """Get cached service instances using lru_cache for automatic memoization.

    :returns: Tuple of (arxiv_client, pdf_parser, database, ingestion_service, paper_rrf_search)
    """
    logger.info("Initializing services (cached with lru_cache)")

    # Initialize core services
    arxiv_client = make_arxiv_client()
    pdf_parser = make_pdf_parser_service()
    database = make_database()
    paper_rrf_search = make_paper_rrf_search()

    ingestion_service = make_arxiv_ingestion_service(arxiv_client, pdf_parser, database)

    logger.info("All services initialized and cached with lru_cache")
    return arxiv_client, pdf_parser, database, ingestion_service, paper_rrf_search
