"""Tests for retrieval confidence scoring."""

from src.agents.fusionsearch.confidence import (
    compute_retrieval_confidence,
    filter_confident_documents,
    normalize_document_scores,
)
from src.domain.rerank.schemas import SearchDocument


def _document(
    *,
    arxiv_id: str = "1706.03762",
    score: float = 0.9,
    rerank_score: float | None = None,
) -> SearchDocument:
    return SearchDocument(
        arxiv_id=arxiv_id,
        chunk_text="sample",
        title="Sample",
        score=score,
        rank=0,
        rerank_score=rerank_score,
    )


class TestRetrievalConfidence:
    def test_compute_confidence_with_rerank_scores(self):
        documents = [
            _document(rerank_score=0.9, score=0.9),
            SearchDocument(
                arxiv_id="1810.04805",
                chunk_text="sample",
                title="Sample 2",
                score=0.6,
                rank=1,
                rerank_score=0.6,
            ),
        ]

        confidence = compute_retrieval_confidence(documents, rerank_applied=True)

        assert 0.7 < confidence < 0.95

    def test_compute_confidence_empty_documents(self):
        assert compute_retrieval_confidence([], rerank_applied=False) == 0.0

    def test_normalize_opensearch_scores_relative(self):
        documents = [
            _document(score=10.0),
            SearchDocument(arxiv_id="2", chunk_text="sample", title="Sample 2", score=5.0, rank=1),
        ]

        scores = normalize_document_scores(documents, rerank_applied=False)

        assert scores == [1.0, 0.5]

    def test_filter_confident_documents(self):
        documents = [
            _document(rerank_score=0.9),
            SearchDocument(
                arxiv_id="2",
                chunk_text="sample",
                title="Sample 2",
                score=0.1,
                rank=1,
                rerank_score=0.1,
            ),
        ]

        filtered = filter_confident_documents(
            documents,
            min_document_score=0.35,
            rerank_applied=True,
        )

        assert len(filtered) == 1
        assert filtered[0].arxiv_id == "1706.03762"
