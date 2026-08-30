import pytest

from src.domain.llm.fallback import build_fallback_models, build_model_chain


def test_build_fallback_models_excludes_primary_and_duplicates():
    fallbacks = build_fallback_models(
        "llama3.2:1b",
        "llama3.2:1b,llama3.2:3b,llama3.2:3b",
    )

    assert fallbacks == ["llama3.2:3b"]


def test_build_fallback_models_returns_empty_when_disabled():
    assert build_fallback_models("llama3.2:1b", "") == []


def test_build_model_chain_includes_primary_and_fallbacks():
    chain = build_model_chain("llama3.2:1b", "llama3.2:3b,llama3.2:1b")

    assert chain == ["llama3.2:1b", "llama3.2:3b"]


def test_build_fallback_models_with_normalizer():
    def normalize(model: str) -> str:
        if "/" in model:
            return model
        return f"ollama/{model.strip()}"

    fallbacks = build_fallback_models(
        "llama3.2:1b",
        "openai/gpt-4o-mini,ollama/llama3.2:1b",
        normalize=normalize,
    )

    assert fallbacks == ["openai/gpt-4o-mini"]
