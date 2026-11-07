from src.domain.kg.prompts import (
    ENTITY_RELATION_EXTRACTION_PROMPT,
    KEYWORD_EXTRACTION_PROMPT,
)


def test_entity_extraction_prompt_formats_without_error() -> None:
    text = ENTITY_RELATION_EXTRACTION_PROMPT.format(
        chunk_text="Transformer encoders use self-attention.",
        section_title="Methods",
        max_entities=20,
        max_relations=20,
    )
    assert "Methods" in text
    assert "self-attention" in text
    assert "-Goal-" in text
    assert "Example 1" in text


def test_keyword_extraction_prompt_formats_without_error() -> None:
    text = KEYWORD_EXTRACTION_PROMPT.format(query="What is RAG?")
    assert "What is RAG?" in text
    assert "high_level_keywords" in text
    assert "Example 4" in text
