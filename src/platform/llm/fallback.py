from typing import Callable, List, Optional


def build_fallback_models(
    primary_model: str,
    fallback_models_config: str,
    normalize: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """Return configured fallback models, excluding the primary model and duplicates."""
    if not fallback_models_config.strip():
        return []

    normalize_fn = normalize or (lambda model: model.strip())
    normalized_primary = normalize_fn(primary_model)
    seen = {normalized_primary}
    fallbacks: List[str] = []

    for entry in fallback_models_config.split(","):
        candidate = entry.strip()
        if not candidate:
            continue

        normalized = normalize_fn(candidate)
        if normalized in seen:
            continue

        seen.add(normalized)
        fallbacks.append(candidate)

    return fallbacks


def build_model_chain(
    primary_model: str,
    fallback_models_config: str,
    normalize: Optional[Callable[[str], str]] = None,
) -> List[str]:
    """Return the primary model followed by configured fallbacks."""
    return [primary_model, *build_fallback_models(primary_model, fallback_models_config, normalize)]
