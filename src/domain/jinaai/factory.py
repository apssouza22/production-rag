from typing import Optional

from src.config import Settings, get_settings

from .jina_client import JinaEmbeddingsClient
from .jina_reranker_client import JinaRerankerClient


def make_embeddings_service(settings: Optional[Settings] = None) -> JinaEmbeddingsClient:
    """Factory function to create embeddings service.

    Creates a new client instance each time to avoid closed client issues.

    :param settings: Optional settings instance
    :returns: JinaEmbeddingsClient instance
    """
    if settings is None:
        settings = get_settings()

    api_key = settings.jina_api_key
    return JinaEmbeddingsClient(api_key=api_key)


def make_embeddings_client(settings: Optional[Settings] = None) -> JinaEmbeddingsClient:
    """Factory function to create embeddings client.

    Creates a new client instance each time to avoid closed client issues.

    :param settings: Optional settings instance
    :returns: JinaEmbeddingsClient instance
    """
    if settings is None:
        settings = get_settings()

    api_key = settings.jina_api_key
    return JinaEmbeddingsClient(api_key=api_key)


def make_reranker_client(settings: Optional[Settings] = None) -> JinaRerankerClient:
    """Factory function to create Jina reranker client.

    :param settings: Optional settings instance
    :returns: JinaRerankerClient instance
    """
    if settings is None:
        settings = get_settings()

    return JinaRerankerClient(api_key=settings.jina_api_key)
