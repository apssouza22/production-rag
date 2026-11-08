from unittest.mock import MagicMock

from src.domain.rrf.parent_expansion import ParentSectionAssembler, ParentSectionExpander


def _expander_with_sections(side_effect) -> ParentSectionExpander:
    expander = ParentSectionExpander(MagicMock())
    expander.get_chunks_by_section = MagicMock(side_effect=side_effect)
    return expander


def test_assemble_parent_text_dedupes_shared_header() -> None:
    header = "Title\n\nAbstract: abs\n\nSection: Methods > Training\n\n"
    siblings = [
        {"chunk_index": 0, "chunk_text": f"{header}alpha"},
        {"chunk_index": 1, "chunk_text": f"{header}beta"},
    ]
    assembled = ParentSectionAssembler.assemble(siblings)
    assert assembled == f"{header}alphabeta"
    assert assembled.count("Abstract: abs") == 1


def test_expand_hits_skips_single_sibling_section() -> None:
    hits = [
        {
            "document_id": "2401.00001",
            "chunk_index": 0,
            "section_breadcrumb": "Methods > Training",
            "chunk_text": "only",
            "score": 1.0,
        }
    ]

    expander = _expander_with_sections(
        lambda _doc, _crumb: [{"chunk_index": 0, "chunk_text": "only"}],
    )
    expanded = expander.expand(hits)
    assert len(expanded) == 1
    assert expanded[0]["chunk_text"] == "only"
    assert expanded[0]["parent_expansion_applied"] is False


def test_expand_hits_collapses_duplicate_sections() -> None:
    hits = [
        {
            "document_id": "2401.00001",
            "chunk_index": 0,
            "section_breadcrumb": "Methods > Training",
            "chunk_text": "a",
            "score": 0.5,
        },
        {
            "document_id": "2401.00001",
            "chunk_index": 1,
            "section_breadcrumb": "Methods > Training",
            "chunk_text": "b",
            "score": 0.9,
        },
    ]

    expander = _expander_with_sections(
        lambda _doc, _crumb: [
            {"chunk_index": 0, "chunk_text": "hdr-a"},
            {"chunk_index": 1, "chunk_text": "hdr-b"},
        ],
    )
    expanded = expander.expand(hits)
    assert len(expanded) == 1
    assert expanded[0]["parent_expansion_applied"] is True
    assert expanded[0]["matched_child_chunk_index"] == 1
    assert expanded[0]["chunk_text"] == "hdr-ab"
