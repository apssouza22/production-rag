import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.agents.fusionsearch.factory import make_agentic_rag_service
from src.agents.knowledgerouter import make_knowledge_router_service
from src.agents.texttosql import make_text_to_sql_service
from src.api import agentic_ask, hybrid_search, knowledge_router, ping, rerank_search, text_to_sql
from src.config import get_settings
from src.domain.arxiv.factory import make_arxiv_client
from src.domain.db.factory import make_database
from src.domain.jinaai.factory import make_embeddings_service, make_reranker_client
from src.domain.opensearch.factory import make_opensearch_client
from src.domain.pdf_parser.factory import make_pdf_parser_service
from src.domain.rerank.factory import make_rerank_search_service
from src.platform.langfuse.factory import make_langfuse_tracer

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

    app.state.opensearch_client = make_opensearch_client()
    app.state.arxiv_client = make_arxiv_client()
    app.state.pdf_parser = make_pdf_parser_service()
    app.state.embeddings_service = make_embeddings_service()
    app.state.reranker_service = make_reranker_client()
    app.state.rerank_search_service = make_rerank_search_service(
        opensearch_client=app.state.opensearch_client,
        embeddings_client=app.state.embeddings_service,
        reranker_client=app.state.reranker_service,
    )
    app.state.langfuse_tracer = make_langfuse_tracer()

    app.state.agentic_rag_service = make_agentic_rag_service(
        opensearch_client=app.state.opensearch_client,
        embeddings_client=app.state.embeddings_service,
        reranker_client=app.state.reranker_service,
        langfuse_tracer=app.state.langfuse_tracer,
    )
    app.state.text_to_sql_service = make_text_to_sql_service(
        langfuse_tracer=app.state.langfuse_tracer,
        model=settings.agent_model,
    )
    app.state.knowledge_router_service = make_knowledge_router_service(
        agentic_rag_service=app.state.agentic_rag_service,
        text_to_sql_service=app.state.text_to_sql_service,
        langfuse_tracer=app.state.langfuse_tracer,
        model=settings.agent_model,
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
app.include_router(rerank_search.router, prefix="/api/v1")  # Search with optional Jina reranking
app.include_router(agentic_ask.router)  # Agentic RAG with intelligent retrieval
app.include_router(text_to_sql.router)  # Text-to-SQL agent over PostgreSQL
app.include_router(knowledge_router.router)  # Knowledge router across retrieval agents


if __name__ == "__main__":
    # Import string is required for reload — uvicorn re-imports the module in a child process.
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
