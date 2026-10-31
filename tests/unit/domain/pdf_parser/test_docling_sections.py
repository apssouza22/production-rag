from types import SimpleNamespace
from unittest.mock import MagicMock

from docling_core.types.doc import DocItemLabel, PictureItem
from src.domain.pdf_parser.docling import (
    _extract_sections_and_artifacts,
    assemble_markdown_from_sections,
    extract_sections_from_document,
)


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

    def test_figure_markdown_embedded_in_section_and_document(self, tmp_path):
        picture = MagicMock(spec=PictureItem)
        picture.captions = []
        doc = MagicMock()
        doc.iterate_items.return_value = [
            (SimpleNamespace(label=DocItemLabel.SECTION_HEADER, text="Results", level=1), 1),
            (picture, 2),
            (SimpleNamespace(label=DocItemLabel.PARAGRAPH, text="See the chart above."), 2),
        ]
        saved_image = MagicMock()
        picture.get_image.return_value = saved_image

        artifacts_dir = tmp_path / "paper_assets"
        sections, figures, _ = _extract_sections_and_artifacts(doc, artifacts_dir)

        assert len(figures) == 1
        assert len(sections) == 1
        assert sections[0].figure_paths
        assert "![Figure 1](paper_assets/figure_001.png)" in sections[0].content
        assert "See the chart above." in sections[0].content
        saved_image.save.assert_called_once()

        markdown = assemble_markdown_from_sections(sections)
        assert "# Results" in markdown
        assert "![Figure 1](paper_assets/figure_001.png)" in markdown
        assert "See the chart above." in markdown
