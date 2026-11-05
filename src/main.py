import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.agents.fusionsearch.factory import make_agentic_rag_service
from src.agents.knowledgerouter import make_knowledge_router_service
from src.agents.texttosql import make_text_to_sql_service
from src.api import agentic_ask, hybrid_search, kg_retrieval, knowledge_router, ping, rerank_search, text_to_sql
from src.config import get_settings
from src.domain.arxiv.factory import make_arxiv_client
from src.domain.db.factory import make_database
from src.domain.jinaai.factory import make_embeddings_service, make_reranker_client
from src.domain.kg.factory import make_dual_level_kg_retrieval_service, make_kg_indexing_service
from src.domain.pdf_parser.factory import make_pdf_parser_service
from src.domain.rrf.factory import make_hybrid_rrf_search
from src.domain.search.factory import make_rerank_search_service
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

    app.state.hybrid_rrf_search = make_hybrid_rrf_search()
    app.state.arxiv_client = make_arxiv_client()
    app.state.pdf_parser = make_pdf_parser_service()
    app.state.embeddings_service = make_embeddings_service()
    app.state.reranker_service = make_reranker_client()
    app.state.langfuse_tracer = make_langfuse_tracer()

    app.state.agentic_rag_service = make_agentic_rag_service(
        paper_rrf_search=app.state.hybrid_rrf_search,
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

    kg_retrieval_service = None
    app.state.kg_indexing_service = None
    app.state.kg_retrieval_service = None
    if settings.kg.enabled:
        app.state.kg_indexing_service = make_kg_indexing_service(
            database=app.state.database,
            settings=settings,
        )
        kg_retrieval_service = make_dual_level_kg_retrieval_service(
            database=app.state.database,
            settings=settings,
        )
        app.state.kg_retrieval_service = kg_retrieval_service
        logger.info(
            "KG services enabled (ingestion partition: %s)", settings.kg.ingestion_name or settings.opensearch.ingestion_name
        )

    app.state.rerank_search_service = make_rerank_search_service(
        paper_rrf_search=app.state.hybrid_rrf_search,
        embeddings_client=app.state.embeddings_service,
        reranker_client=app.state.reranker_service,
        kg_retrieval_service=kg_retrieval_service,
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
app.include_router(kg_retrieval.router, prefix="/api/v1")  # Dual-level KG debug retrieval


if __name__ == "__main__":
    # Import string is required for reload — uvicorn re-imports the module in a child process.
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
