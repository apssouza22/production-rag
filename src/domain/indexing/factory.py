from typing import Optional

from src.config import Settings, get_settings
from src.domain.jinaai.factory import make_embeddings_client
from src.domain.llm.factory import make_llm_client
from src.domain.opensearch.factory import make_opensearch_client_fresh

from .chunk_contextualizer import ChunkContextualizer
from .hybrid_indexer import HybridIndexingService
from .text_chunker import TextChunker


def _resolve_contextualization_model(settings: Settings) -> str:
    if settings.chunking.contextualization_model:
        return settings.chunking.contextualization_model
    if settings.llm_provider == "bifrost":
        return settings.agent_model
    return settings.ollama_model


def make_hybrid_indexing_service(
    settings: Optional[Settings] = None, opensearch_host: Optional[str] = None
) -> HybridIndexingService:
    """Factory function to create hybrid indexing service.

    Creates a new service instance each time.

    :param settings: Optional settings instance
    :param opensearch_host: Optional OpenSearch host override
    :returns: HybridIndexingService instance
    """
    if settings is None:
        settings = get_settings()

    # Create dependencies using configuration
    chunker = TextChunker(
        chunk_size=settings.chunking.chunk_size,
        overlap_size=settings.chunking.overlap_size,
        min_chunk_size=settings.chunking.min_chunk_size,
    )
    embeddings_client = make_embeddings_client(settings)
    opensearch_client = make_opensearch_client_fresh(settings, host=opensearch_host)

    contextualizer = None
    if settings.chunking.contextualization_enabled:
        contextualizer = ChunkContextualizer(
            llm_client=make_llm_client(settings),
            model=_resolve_contextualization_model(settings),
            max_document_chars=settings.chunking.max_document_chars,
            max_concurrent_requests=settings.chunking.max_concurrent_context_requests,
        )

    # Create indexing service
    return HybridIndexingService(
        chunker=chunker,
        embeddings_client=embeddings_client,
        opensearch_client=opensearch_client,
        contextualizer=contextualizer,
    )
