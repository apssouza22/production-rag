"""Build domain PdfContent from a Docling conversion result."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class SectionDraft:
    """Mutable section under construction while walking Docling document items."""

    title: str = "Content"
    content: str = ""
    level: int = 1
    breadcrumb: str | None = None
    figure_paths: list[str] = field(default_factory=list)
    table_paths: list[str] = field(default_factory=list)

    @classmethod
    def preamble(cls) -> SectionDraft:
        return cls()

    @classmethod
    def for_header(cls, *, title: str, level: int, breadcrumb: str) -> SectionDraft:
        return cls(title=title, level=level, breadcrumb=breadcrumb)

    def append_text(self, text: str) -> None:
        self.content += text + "\n"

    def has_extractable_body(self) -> bool:
        return bool(self.content.strip() or self.figure_paths or self.table_paths)

    def to_paper_section(self) -> PaperSection:
        return PaperSection(
            title=self.title,
            content=self.content.strip(),
            level=self.level,
            breadcrumb=self.breadcrumb,
            figure_paths=self.figure_paths,
            table_paths=self.table_paths,
        )


class DoclingSectionParser:
    """Single pass over Docling items to build sections and optional figure/table artifacts."""

    def __init__(
        self,
        doc: DoclingDocument,
        artefact_extractor: ArtefactExtractor,
        *,
        extract_figures: bool,
        extract_tables: bool,
    ) -> None:
        self._doc = doc
        self._artefact_extractor = artefact_extractor
        self._extract_figures = extract_figures
        self._extract_tables = extract_tables
        self._sections: list[PaperSection] = []
        self._header_stack: list[tuple[int, str]] = []
        self._current = SectionDraft.preamble()
        items = list(doc.iterate_items())
        self._items_by_ref = {getattr(item, "self_ref", ""): item for item, _ in items}
        self._items = items

    def parse(self) -> list[PaperSection]:
        for item, depth in self._items:
            self._handle_item(item, depth)
        self._flush_current()
        return self._sections

    def _flush_current(self) -> None:
        if self._current.has_extractable_body():
            self._sections.append(self._current.to_paper_section())

    def _handle_item(self, item: Any, depth: int) -> None:
        label = getattr(item, "label", None)
        if label in _HEADER_LABELS:
            self._handle_header(item, depth, label, getattr(item, "text", "").strip())
        elif isinstance(item, PictureItem) and self._extract_figures:
            self._artefact_extractor.save_figure(item, self._doc, self._current, self._items_by_ref)
        elif isinstance(item, TableItem) and self._extract_tables:
            self._artefact_extractor.save_table(item, self._doc, self._current, self._items_by_ref)
        elif label in _CONTENT_LABELS:
            text = getattr(item, "text", "").strip()
            if text:
                self._current.append_text(text)

    def _handle_header(self, item: Any, depth: int, label: DocItemLabel, text: str) -> None:
        self._flush_current()
        level = self._header_level(item, depth, label)
        self._push_header(level, text)
        breadcrumb = " > ".join(title for _, title in self._header_stack)
        self._current = SectionDraft.for_header(title=text, level=level, breadcrumb=breadcrumb)

    def _header_level(self, item: Any, depth: int, label: DocItemLabel) -> int:
        if label == DocItemLabel.TITLE:
            # Keep the document title as the breadcrumb root across same-level section headers.
            return 0
        return getattr(item, "level", None) or max(depth, 1)

    def _push_header(self, level: int, text: str) -> None:
        while self._header_stack and self._header_stack[-1][0] >= level:
            self._header_stack.pop()
        self._header_stack.append((level, text))


def extract_sections_from_document(doc: DoclingDocument) -> list[PaperSection]:
    """Extract paper sections from a Docling document without figures or tables."""
    sections, _, _ = ContentExtractor().extract_sections_and_artifacts(doc)
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

    def extract_sections_and_artifacts(
        self,
        doc: DoclingDocument,
        artifacts_dir: Optional[Path] = None,
    ) -> tuple[list[PaperSection], list[PaperFigure], list[PaperTable]]:
        assets_link_prefix = artifacts_dir.name if artifacts_dir is not None else None
        artefact_extractor = ArtefactExtractor(artifacts_dir, assets_link_prefix=assets_link_prefix)
        parser = DoclingSectionParser(
            doc,
            artefact_extractor,
            extract_figures=self.do_figure_extraction,
            extract_tables=self.do_table_structure,
        )
        sections = parser.parse()
        return sections, artefact_extractor.figures, artefact_extractor.tables

    def build_content_session(
        self,
        conversion_result: ConversionResult,
        *,
        artifacts_dir: Path | None,
    ) -> PdfContent:
        """Build PdfContent from a completed Docling conversion."""
        doc = conversion_result.document
        sections, figures, tables = self.extract_sections_and_artifacts(doc, artifacts_dir)

        return PdfContent(
            sections=sections,
            figures=figures,
            tables=tables,
            raw_text=doc.export_to_text(),
            markdown=None,
            references=[],
            parser_used=ParserType.DOCLING,
            metadata={
                "docling_markdown_fallback": doc.export_to_markdown(),
            },
        )
