import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.api import agentic_ask, hybrid_search, knowledge_router, ping, text_to_sql
from src.config import get_settings
from src.domain.arxiv.factory import make_arxiv_client
from src.domain.cache.factory import make_cache_client
from src.domain.db.factory import make_database
from src.domain.jinaai.factory import make_embeddings_service, make_reranker_client
from src.domain.langfuse.factory import make_langfuse_tracer
from src.domain.llm.factory import make_llm_client
from src.domain.opensearch.factory import make_opensearch_client
from src.domain.pdf_parser.factory import make_pdf_parser_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan for the API.
    """
    logger.info("Starting RAG API...")

    settings = get_settings()
    app.state.settings = settings

    database = make_database()
    app.state.database = database
    logger.info("Database connected")

    # Initialize search service
    opensearch_client = make_opensearch_client()
    app.state.opensearch_client = opensearch_client

    # Verify OpenSearch connectivity and create index if needed
    if opensearch_client.health_check():
        logger.info("OpenSearch connected successfully")

        # Setup hybrid index (supports all search types)
        setup_results = opensearch_client.setup_indices(force=False)
        if setup_results.get("hybrid_index"):
            logger.info("Hybrid index created")
        else:
            logger.info("Hybrid index already exists")

        # Get simple statistics
        try:
            stats = opensearch_client.client.count(index=opensearch_client.index_name)
            logger.info(f"OpenSearch ready: {stats['count']} documents indexed")
        except Exception:
            logger.info("OpenSearch index ready (stats unavailable)")
    else:
        logger.warning("OpenSearch connection failed - search features will be limited")

    # Initialize other services (kept for future endpoints and notebook demos)
    app.state.arxiv_client = make_arxiv_client()
    app.state.pdf_parser = make_pdf_parser_service()
    app.state.embeddings_service = make_embeddings_service()
    app.state.reranker_service = make_reranker_client()
    app.state.llm_client = make_llm_client()
    app.state.langfuse_tracer = make_langfuse_tracer()
    try:
        app.state.cache_client = make_cache_client(settings, embeddings_client=app.state.embeddings_service)
    except Exception as e:
        logger.warning("Cache unavailable, hybrid search will run without caching: %s", e)
        app.state.cache_client = None
    logger.info(
        "Services initialized: arXiv API client, PDF parser, OpenSearch, Embeddings, "
        "%s LLM provider, Langfuse, Cache",
        settings.llm_provider,
    )

    logger.info("API ready")
    yield

    database.teardown()
    logger.info("API shutdown complete")


app = FastAPI(
    title="arXiv Paper Curator API",
    description="Personal arXiv CS.AI paper curator with RAG capabilities",
    version=os.getenv("APP_VERSION", "0.1.0"),
    lifespan=lifespan,
)

# Include routers
app.include_router(ping.router, prefix="/api/v1")  # Health check endpoint
app.include_router(hybrid_search.router, prefix="/api/v1")  # Search chunks with BM25/hybrid
app.include_router(agentic_ask.router)  # Agentic RAG with intelligent retrieval
app.include_router(text_to_sql.router)  # Text-to-SQL agent over PostgreSQL
app.include_router(knowledge_router.router)  # Knowledge router across retrieval agents


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
