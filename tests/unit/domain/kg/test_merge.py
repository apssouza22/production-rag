import pytest
from src.domain.kg.merge import merge_descriptions, merge_unique_strings, normalize_entity_id, parse_chunk_key


def test_normalize_entity_id_slug() -> None:
    assert normalize_entity_id("  Transformer Model ") == "transformer_model"
    assert normalize_entity_id("GPT-4") == "gpt_4"


def test_merge_descriptions_dedupes_sentences() -> None:
    existing = "First sentence."
    new = "First sentence. Second idea."
    merged = merge_descriptions(existing, new)
    assert "First sentence." in merged
    assert "Second idea." in merged


def test_merge_unique_strings() -> None:
    assert merge_unique_strings(["a"], ["a", "b"]) == ["a", "b"]


def test_parse_chunk_key() -> None:
    assert parse_chunk_key("2611.02200:3") == ("2611.02200", 3)
