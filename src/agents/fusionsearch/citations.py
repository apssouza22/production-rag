"""Citation extraction and validation for generated answers."""

from __future__ import annotations

import re

from src.agents.fusionsearch.models import CitationValidationResult, SourceItem

SOURCE_CITATION_PATTERN = re.compile(r"\[Source:\s*(.+?)\]", re.IGNORECASE)
ARXIV_ID_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
INSUFFICIENT_INFO_MARKERS = (
    "don't have sufficient information",
    "do not have sufficient information",
    "don't have enough information",
    "do not have enough information",
)


def answer_indicates_insufficient_information(answer: str) -> bool:
    """Detect when the model appropriately declined to answer from context."""
    normalized = answer.lower()
    return any(marker in normalized for marker in INSUFFICIENT_INFO_MARKERS)


def extract_source_title_citations(answer: str) -> list[str]:
    """Extract paper titles cited as ``[Source: <title>]``."""
    return [match.strip() for match in SOURCE_CITATION_PATTERN.findall(answer)]


def extract_arxiv_citations(answer: str) -> list[str]:
    """Extract arXiv IDs cited in the answer text."""
    return list(dict.fromkeys(ARXIV_ID_PATTERN.findall(answer)))


def validate_citations(answer: str, sources: list[SourceItem]) -> CitationValidationResult:
    """Verify that citations in the answer map to retrieved sources."""
    title_citations = extract_source_title_citations(answer)
    arxiv_citations = extract_arxiv_citations(answer)
    cited_titles = {citation.lower() for citation in title_citations}
    source_titles = {source.title.lower().strip() for source in sources if source.title}
    source_arxiv_ids = {source.arxiv_id for source in sources if source.arxiv_id}

    valid_titles = [citation for citation in title_citations if citation.lower() in source_titles]
    invalid_titles = [citation for citation in title_citations if citation.lower() not in source_titles]

    valid_arxiv = [arxiv_id for arxiv_id in arxiv_citations if arxiv_id in source_arxiv_ids]
    invalid_arxiv = [arxiv_id for arxiv_id in arxiv_citations if arxiv_id not in source_arxiv_ids]

    total_citations = len(title_citations) + len(arxiv_citations)
    valid_citations = valid_titles + valid_arxiv
    invalid_citations = invalid_titles + invalid_arxiv

    return CitationValidationResult(
        total_citations=total_citations,
        valid_citations=valid_citations,
        invalid_citations=invalid_citations,
        is_trustworthy=len(invalid_citations) == 0 and total_citations > 0,
    )
