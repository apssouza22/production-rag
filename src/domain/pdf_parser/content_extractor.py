"""Build domain PdfContent from a Docling conversion result."""

from pathlib import Path
from typing import Any, Optional

from docling.datamodel.document import ConversionResult
from docling_core.types import DoclingDocument
from docling_core.types.doc import DocItemLabel, PictureItem, TableItem

from .artefact_extractor import ArtefactExtractor
from .models import PaperFigure, PaperSection, PaperTable, ParserType, PdfContent

_HEADER_LABELS = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}
_CONTENT_LABELS = {
    DocItemLabel.TEXT,
    DocItemLabel.PARAGRAPH,
    DocItemLabel.LIST_ITEM,
    DocItemLabel.CAPTION,
    DocItemLabel.CODE,
    DocItemLabel.FORMULA,
}


def assemble_markdown_from_sections(sections: list[PaperSection]) -> str:
    """Build a Markdown document from extracted sections, preserving inline figures and tables."""
    parts: list[str] = []
    for section in sections:
        if section.title == "Content" and section.breadcrumb is None:
            if section.content.strip():
                parts.append(section.content.strip())
            continue

        heading_level = min(max(section.level, 1), 6)
        parts.append(f"{'#' * heading_level} {section.title}")
        if section.content.strip():
            parts.append(section.content.strip())

    if not parts:
        return ""
    return "\n\n".join(parts).rstrip() + "\n"


def _extract_sections_and_artifacts(
    doc: DoclingDocument,
    artifacts_dir: Optional[Path] = None,
    *,
    extract_figures: bool = False,
    extract_tables: bool = False,
) -> tuple[list[PaperSection], list[PaperFigure], list[PaperTable]]:
    sections: list[PaperSection] = []
    assets_link_prefix = artifacts_dir.name if artifacts_dir is not None else None
    artefact_extractor = ArtefactExtractor(artifacts_dir, assets_link_prefix=assets_link_prefix)
    header_stack: list[tuple[int, str]] = []
    current: Optional[dict[str, Any]] = {
        "title": "Content",
        "content": "",
        "level": 1,
        "breadcrumb": None,
        "figure_paths": [],
        "table_paths": [],
    }

    items = list(doc.iterate_items())
    items_by_ref = {getattr(item, "self_ref", ""): item for item, _ in items}

    def flush_current() -> None:
        nonlocal current
        if current and (current["content"].strip() or current["figure_paths"] or current["table_paths"]):
            sections.append(
                PaperSection(
                    title=current["title"],
                    content=current["content"].strip(),
                    level=current["level"],
                    breadcrumb=current.get("breadcrumb"),
                    figure_paths=current["figure_paths"],
                    table_paths=current["table_paths"],
                )
            )

    for item, depth in items:
        label = getattr(item, "label", None)
        text = getattr(item, "text", "").strip()

        if label in _HEADER_LABELS:
            flush_current()

            if label == DocItemLabel.TITLE:
                # Keep the document title as the breadcrumb root across same-level section headers.
                level = 0
            else:
                level = getattr(item, "level", None) or max(depth, 1)

            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, text))

            breadcrumb = " > ".join(title for _, title in header_stack)
            current = {
                "title": text,
                "content": "",
                "level": level,
                "breadcrumb": breadcrumb,
                "figure_paths": [],
                "table_paths": [],
            }
        elif isinstance(item, PictureItem) and extract_figures:
            artefact_extractor.save_figure(item, doc, current, items_by_ref)
        elif isinstance(item, TableItem) and extract_tables:
            artefact_extractor.save_table(item, doc, current, items_by_ref)
        elif current is not None and label in _CONTENT_LABELS:
            if text:
                current["content"] += text + "\n"

    flush_current()
    return sections, artefact_extractor.figures, artefact_extractor.tables


def extract_sections_from_document(doc: DoclingDocument) -> list[PaperSection]:
    """Extract paper sections from a Docling document without figures or tables."""
    sections, _, _ = _extract_sections_and_artifacts(doc, extract_figures=False, extract_tables=False)
    return sections


class ContentExtractor:
    """Maps Docling conversion output to domain PdfContent and optional disk artifacts."""

    def __init__(
        self,
        do_table_structure: bool = False,
        do_figure_extraction: bool = False,
    ) -> None:
        self.do_table_structure = do_table_structure
        self.do_figure_extraction = do_figure_extraction

    def build_content_session(
        self,
        conversion_result: ConversionResult,
        *,
        artifacts_dir: Path | None,
    ) -> PdfContent:
        """Build PdfContent from a completed Docling conversion."""
        doc = conversion_result.document
        sections, figures, tables = _extract_sections_and_artifacts(
            doc,
            artifacts_dir,
            extract_figures=self.do_figure_extraction,
            extract_tables=self.do_table_structure,
        )
        markdown = assemble_markdown_from_sections(sections)

        return PdfContent(
            sections=sections,
            figures=figures,
            tables=tables,
            raw_text=doc.export_to_text(),
            markdown=markdown or doc.export_to_markdown(),
            references=[],
            parser_used=ParserType.DOCLING,
            metadata={"source": "docling", "note": "Content extracted from PDF, metadata comes from arXiv API"},
        )
