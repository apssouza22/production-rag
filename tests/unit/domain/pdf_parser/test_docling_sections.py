from types import SimpleNamespace

from docling_core.types.doc import DocItemLabel
from src.domain.pdf_parser.docling import extract_sections_from_document


def _item(label: DocItemLabel, text: str, level: int = 1) -> SimpleNamespace:
    return SimpleNamespace(label=label, text=text, level=level)


class TestExtractSectionsFromDocument:
    def test_builds_nested_breadcrumbs(self):
        doc = SimpleNamespace(
            iterate_items=lambda: iter(
                [
                    (_item(DocItemLabel.TITLE, "Paper Title", level=1), 1),
                    (_item(DocItemLabel.SECTION_HEADER, "Methods", level=1), 1),
                    (_item(DocItemLabel.PARAGRAPH, "We describe the setup."), 2),
                    (_item(DocItemLabel.SECTION_HEADER, "Training", level=2), 2),
                    (_item(DocItemLabel.PARAGRAPH, "We train for ten epochs."), 3),
                    (_item(DocItemLabel.SECTION_HEADER, "Results", level=1), 1),
                    (_item(DocItemLabel.PARAGRAPH, "Accuracy improves."), 2),
                ]
            )
        )

        sections = extract_sections_from_document(doc)

        assert len(sections) == 3
        assert sections[0].title == "Methods"
        assert sections[0].breadcrumb == "Paper Title > Methods"
        assert sections[0].content == "We describe the setup."
        assert sections[1].title == "Training"
        assert sections[1].breadcrumb == "Paper Title > Methods > Training"
        assert sections[1].level == 2
        assert sections[2].title == "Results"
        assert sections[2].breadcrumb == "Paper Title > Results"

    def test_captures_preamble_before_first_header(self):
        doc = SimpleNamespace(
            iterate_items=lambda: iter(
                [
                    (_item(DocItemLabel.PARAGRAPH, "Author list and affiliations."), 1),
                    (_item(DocItemLabel.SECTION_HEADER, "Introduction", level=1), 1),
                    (_item(DocItemLabel.PARAGRAPH, "This paper studies RAG."), 2),
                ]
            )
        )

        sections = extract_sections_from_document(doc)

        assert sections[0].title == "Content"
        assert sections[0].breadcrumb is None
        assert sections[0].content == "Author list and affiliations."
        assert sections[1].title == "Introduction"
        assert sections[1].breadcrumb == "Introduction"
