"""Retrieval confidence scoring for confidence-aware RAG."""

from __future__ import annotations

from src.domain.rerank.schemas import SearchDocument

RERANK_SCORE_WEIGHT = 0.7
MEAN_TOP_K_WEIGHT = 0.3
TOP_K_FOR_MEAN = 3


def document_relevance_score(document: SearchDocument, *, rerank_applied: bool) -> float:
    """Normalize a single document score to the 0-1 range."""
    if rerank_applied and document.rerank_score is not None:
        return max(0.0, min(1.0, document.rerank_score))

    if document.score <= 0:
        return 0.0

    # OpenSearch scores vary by mode (RRF vs BM25); normalize within the result set later.
    return document.score


def normalize_document_scores(documents: list[SearchDocument], *, rerank_applied: bool) -> list[float]:
    """Return per-document relevance scores normalized to 0-1."""
    if not documents:
        return []

    if rerank_applied and any(doc.rerank_score is not None for doc in documents):
        return [document_relevance_score(doc, rerank_applied=True) for doc in documents]

    raw_scores = [max(0.0, doc.score) for doc in documents]
    max_score = max(raw_scores)
    if max_score <= 0:
        return [0.0 for _ in documents]

    return [min(1.0, score / max_score) for score in raw_scores]


def compute_retrieval_confidence(documents: list[SearchDocument], *, rerank_applied: bool) -> float:
    """Compute aggregate retrieval confidence from ranked documents.

    Combines the top document score with the mean of the top-k scores, following the
    defense-in-depth pattern from confidence-aware RAG: strong top hits with weak tail
    still produce moderate confidence.
    """
    if not documents:
        return 0.0

    normalized_scores = normalize_document_scores(documents, rerank_applied=rerank_applied)
    top_score = normalized_scores[0]
    mean_top = sum(normalized_scores[:TOP_K_FOR_MEAN]) / min(len(normalized_scores), TOP_K_FOR_MEAN)

    return max(0.0, min(1.0, RERANK_SCORE_WEIGHT * top_score + MEAN_TOP_K_WEIGHT * mean_top))


def filter_confident_documents(
    documents: list[SearchDocument],
    *,
    min_document_score: float,
    rerank_applied: bool,
) -> list[SearchDocument]:
    """Keep only documents whose normalized relevance score meets the threshold."""
    if not documents:
        return []

    normalized_scores = normalize_document_scores(documents, rerank_applied=rerank_applied)
    return [
        document
        for document, score in zip(documents, normalized_scores, strict=True)
        if score >= min_document_score
    ]
