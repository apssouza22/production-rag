"""OpenSearch index configuration for hybrid search (BM25 + Vector)."""

from src.config import OpenSearchSettings


def hybrid_rrf_pipeline_body(settings: OpenSearchSettings) -> dict:
    """RRF search pipeline definition keyed by settings.opensearch.rrf_pipeline_name."""
    return {
        "description": "Post processor for hybrid RRF search",
        "phase_results_processors": [
            {
                "score-ranker-processor": {
                    "combination": {
                        "technique": "rrf",
                        "rank_constant": 60,
                    }
                }
            }
        ],
    }
