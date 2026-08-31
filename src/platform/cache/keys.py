import hashlib
import json

from src.domain.opensearch.schemas import HybridSearchRequest


def build_params_data(request: HybridSearchRequest) -> dict:
    """Build request parameters used for cache scoping (excluding query text)."""
    return {
        "size": request.size,
        "from_": request.from_,
        "use_hybrid": request.use_hybrid,
        "categories": sorted(request.categories) if request.categories else [],
        "latest_papers": request.latest_papers,
        "min_score": request.min_score,
    }


def build_params_hash(request: HybridSearchRequest) -> str:
    """Hash request parameters so semantically similar queries share scope."""
    key_string = json.dumps(build_params_data(request), sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]


def build_exact_cache_key(request: HybridSearchRequest) -> str:
    """Generate exact cache key based on full request parameters."""
    key_data = {"query": request.query, **build_params_data(request)}
    key_string = json.dumps(key_data, sort_keys=True)
    key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
    return f"hybrid_search_cache:{key_hash}"
