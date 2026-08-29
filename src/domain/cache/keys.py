import hashlib
import json

from src.agents.fusionsearch.schemas import AskRequest


def build_params_data(request: AskRequest) -> dict:
    """Build request parameters used for cache scoping (excluding query text)."""
    return {
        "model": request.model,
        "top_k": request.top_k,
        "use_hybrid": request.use_hybrid,
        "categories": sorted(request.categories) if request.categories else [],
    }


def build_params_hash(request: AskRequest) -> str:
    """Hash request parameters so semantically similar queries share scope."""
    key_string = json.dumps(build_params_data(request), sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]


def build_exact_cache_key(request: AskRequest) -> str:
    """Generate exact cache key based on full request parameters."""
    key_data = {"query": request.query, **build_params_data(request)}
    key_string = json.dumps(key_data, sort_keys=True)
    key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
    return f"exact_cache:{key_hash}"
