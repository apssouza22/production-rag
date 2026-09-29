import pytest
from src.platform.llm.fallback import build_fallback_models, build_model_chain


def test_build_fallback_models_excludes_primary_and_duplicates():
    fallbacks = build_fallback_models(
        "gpt-4o-mini",
        "gpt-4o-mini,gpt-4o,gpt-4o",
    )

    assert fallbacks == ["gpt-4o"]


def test_build_fallback_models_returns_empty_when_disabled():
    assert build_fallback_models("gpt-4o-mini", "") == []


def test_build_model_chain_includes_primary_and_fallbacks():
    chain = build_model_chain("gpt-4o-mini", "gpt-4o,gpt-4o-mini")

    assert chain == ["gpt-4o-mini", "gpt-4o"]


def test_build_fallback_models_with_normalizer():
    def normalize(model: str) -> str:
        if "/" in model:
            return model
        return f"openai/{model.strip()}"

    fallbacks = build_fallback_models(
        "gpt-4o-mini",
        "openai/gpt-4o,openai/gpt-4o-mini",
        normalize=normalize,
    )

    assert fallbacks == ["openai/gpt-4o"]
