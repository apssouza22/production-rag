"""Tests for citation validation."""

from src.agents.fusionsearch.citations import answer_indicates_insufficient_information, validate_citations
from src.agents.fusionsearch.models import SourceItem


def _source(title: str, arxiv_id: str) -> SourceItem:
    return SourceItem(
        arxiv_id=arxiv_id,
        title=title,
        authors=["Author"],
        url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        relevance_score=0.9,
    )


class TestCitationValidation:
    def test_valid_title_citation(self):
        sources = [_source("Attention Is All You Need", "1706.03762")]
        answer = "Transformers use self-attention [Source: Attention Is All You Need]."

        result = validate_citations(answer, sources)

        assert result.is_trustworthy is True
        assert result.invalid_citations == []

    def test_invalid_title_citation(self):
        sources = [_source("Attention Is All You Need", "1706.03762")]
        answer = "Transformers are described in [Source: Unknown Paper]."

        result = validate_citations(answer, sources)

        assert result.is_trustworthy is False
        assert "Unknown Paper" in result.invalid_citations

    def test_valid_arxiv_citation(self):
        sources = [_source("Attention Is All You Need", "1706.03762")]
        answer = "See paper 1706.03762 for details."

        result = validate_citations(answer, sources)

        assert result.is_trustworthy is True
        assert "1706.03762" in result.valid_citations

    def test_no_citations_not_trustworthy(self):
        sources = [_source("Attention Is All You Need", "1706.03762")]
        answer = "Transformers use self-attention mechanisms."

        result = validate_citations(answer, sources)

        assert result.is_trustworthy is False
        assert result.total_citations == 0

    def test_detects_insufficient_information_answer(self):
        answer = "I don't have sufficient information to answer this question."

        assert answer_indicates_insufficient_information(answer) is True
