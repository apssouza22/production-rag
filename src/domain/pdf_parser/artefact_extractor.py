"""Save Docling figures and tables and track their section paths."""

import logging
from pathlib import Path
from typing import Any, Optional

from docling_core.types.doc import DoclingDocument, PictureItem, TableItem

from .models import PaperFigure, PaperSection, PaperTable

logger = logging.getLogger(__name__)


class ArtefactExtractor:
    """Extract figure and table files while associating them with paper sections."""

    def __init__(
        self,
        artifacts_dir: Optional[Path] = None,
        assets_link_prefix: Optional[str] = None,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.assets_link_prefix = assets_link_prefix
        self.figures: list[PaperFigure] = []
        self.tables: list[PaperTable] = []

    def save_figure(
        self,
        item: PictureItem,
        doc: DoclingDocument,
        section: dict[str, Any],
        items_by_ref: dict[str, Any],
    ) -> PaperFigure:
        """Save a picture as PNG and attach its path to the current section."""
        number = len(self.figures) + 1
        path = None
        caption = self._caption_for_item(item, items_by_ref) or f"Figure {number}"

        if self.artifacts_dir is not None:
            image_path = self.artifacts_dir / f"figure_{number:03d}.png"
            try:
                image = item.get_image(doc)
                if image is not None:
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_path)
                    path = str(image_path.resolve())
                    section["figure_paths"].append(path)
                    if self.assets_link_prefix is not None:
                        rel_path = f"{self.assets_link_prefix}/figure_{number:03d}.png"
                        section["content"] += f"![{caption}]({rel_path})\n\n"
                else:
                    logger.warning("Docling did not provide image data for figure %d", number)
            except Exception:
                logger.exception("Failed to save figure %d", number)

        figure = PaperFigure(
            caption=caption,
            id=f"figure_{number:03d}",
            path=path,
        )
        self.figures.append(figure)
        return figure

    def save_table(
        self,
        item: TableItem,
        doc: DoclingDocument,
        section: dict[str, Any],
        items_by_ref: dict[str, Any],
    ) -> PaperTable:
        """Save a table as Markdown and attach its path to the current section."""
        number = len(self.tables) + 1
        path = None
        table_markdown = item.export_to_markdown(doc=doc)
        if table_markdown.strip():
            section["content"] += table_markdown.strip() + "\n\n"

        if self.artifacts_dir is not None:
            markdown_path = self.artifacts_dir / f"table_{number:03d}.md"
            try:
                markdown_path.parent.mkdir(parents=True, exist_ok=True)
                markdown_path.write_text(table_markdown, encoding="utf-8")
                path = str(markdown_path.resolve())
                section["table_paths"].append(path)
            except Exception:
                logger.exception("Failed to save table %d", number)

        table = PaperTable(
            caption=self._caption_for_item(item, items_by_ref) or f"Table {number}",
            id=f"table_{number:03d}",
            path=path,
        )
        self.tables.append(table)
        return table

    @staticmethod
    def _caption_for_item(item: Any, items_by_ref: dict[str, Any]) -> str:
        captions: list[str] = []
        for caption_ref in getattr(item, "captions", []):
            caption_item = items_by_ref.get(caption_ref.cref)
            caption_text = getattr(caption_item, "text", "").strip()
            if caption_text:
                captions.append(caption_text)
        return " ".join(captions)
