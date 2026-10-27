"""Markdown export of a Docling document, with figures and tables saved as assets.

The Markdown is assembled in a single pass over ``doc.iterate_items()``: text items become
Markdown blocks, tables stay inline as GFM pipe tables (and are also saved as CSV), and
pictures are saved as PNG and linked from the Markdown.
"""

import csv
import logging
import re
from pathlib import Path
from typing import Any, Optional

from docling_core.types.doc import DocItemLabel
from src.domain.pdf_parser.exceptions import PDFParsingException

logger = logging.getLogger(__name__)

_MAX_HEADING_LEVEL = 6
_SECTION_LABELS = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}
_FIGURE_CAPTION_PREFIX = "Figure"

# A GFM header separator row, e.g. "|:----|-----:|" or "| --- | --- |"
_GFM_SEPARATOR_CELL = re.compile(r"^:?-+:?$")


def _render_text_item(item: Any, depth: int) -> Optional[str]:
    """Render a Docling text item as a Markdown block.

    Section header levels follow the same rule as ``extract_sections_from_document``.

    :param item: Docling item visited by ``iterate_items``
    :param depth: Depth reported by ``iterate_items``
    :returns: Markdown block, or None when the item has no text
    """
    label = getattr(item, "label", None)
    text = getattr(item, "text", "").strip()
    if not text:
        return None

    if label == DocItemLabel.TITLE:
        return f"# {text}"
    if label == DocItemLabel.SECTION_HEADER:
        level = getattr(item, "level", None) or max(depth, 1)
        hashes = "#" * min(level + 1, _MAX_HEADING_LEVEL)
        return f"{hashes} {text}"
    if label == DocItemLabel.LIST_ITEM:
        return f"- {text}"
    if label == DocItemLabel.CODE:
        return f"```\n{text}\n```"
    return text


def _split_gfm_row(line: str) -> list[str]:
    """Split a GFM table row into stripped cells, keeping escaped pipes (``\\|``) inside a cell."""
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith("\\|"):
        row = row[:-1]

    cells: list[str] = []
    current = ""
    i = 0
    while i < len(row):
        if row[i] == "\\" and i + 1 < len(row) and row[i + 1] == "|":
            current += "|"
            i += 2
            continue
        if row[i] == "|":
            cells.append(current.strip())
            current = ""
        else:
            current += row[i]
        i += 1
    cells.append(current.strip())
    return cells


def _is_gfm_separator(line: str) -> bool:
    """Return True when the line is a GFM header separator row."""
    cells = _split_gfm_row(line)
    return bool(cells) and all(_GFM_SEPARATOR_CELL.match(cell) for cell in cells)


def _extract_gfm_table_lines(markdown: str) -> list[str]:
    """Keep only the pipe-table lines of a table's Markdown.

    ``TableItem.export_to_markdown(doc)`` also emits the table caption before the grid; the caption
    is rendered on its own when ``iterate_items`` visits it, so it is dropped here to avoid duplicates.
    """
    return [line for line in markdown.splitlines() if line.strip().startswith("|")]


def _parse_gfm_table(table_lines: list[str]) -> tuple[Optional[list[str]], list[list[str]]]:
    """Parse GFM pipe-table lines into an optional header and data rows.

    The first line is the header only when the second line is a header separator row;
    otherwise every line is a data row (headerless grid).

    :param table_lines: Pipe-table lines of a single table
    :returns: Tuple of (header or None, data rows)
    """
    if len(table_lines) >= 2 and _is_gfm_separator(table_lines[1]):
        return _split_gfm_row(table_lines[0]), [_split_gfm_row(line) for line in table_lines[2:]]
    return None, [_split_gfm_row(line) for line in table_lines]


def _write_table_csv(csv_path: Path, header: Optional[list[str]], rows: list[list[str]]) -> None:
    """Write a table as CSV (UTF-8, comma delimiter, minimal quoting).

    :param csv_path: Destination CSV file
    :param header: Header row, written first only when present
    :param rows: Data rows
    """
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        if header is not None:
            writer.writerow(header)
        writer.writerows(rows)


def _render_table(table: Any, doc: Any, csv_path: Path) -> str:
    """Render a table inline as GFM and save the same grid as CSV.

    :param table: Docling ``TableItem``
    :param doc: Docling document the table belongs to
    :param csv_path: Destination CSV file for this table
    :returns: GFM pipe table for the Markdown (without any link to the CSV)
    """
    table_lines = _extract_gfm_table_lines(table.export_to_markdown(doc))
    header, rows = _parse_gfm_table(table_lines)
    _write_table_csv(csv_path, header, rows)
    return "\n".join(table_lines)


def _figure_caption_text(item: Any) -> Optional[str]:
    """Return the normalized text of a ``CAPTION`` item starting with "Figure", or None."""
    if getattr(item, "label", None) != DocItemLabel.CAPTION:
        return None
    text = " ".join(getattr(item, "text", "").split())
    return text if text.startswith(_FIGURE_CAPTION_PREFIX) else None


def _resolve_alt(
    items: list[tuple[Any, int]],
    index: int,
    picture: Any,
    doc: Any,
    number: int,
    used_captions: Optional[set[int]] = None,
) -> str:
    """Resolve the alt text of a picture.

    Each "Figure" caption is used by at most one picture, because Docling may place several captions
    after a group of pictures, or a caption before its picture. In order:

    1. The next unused ``CAPTION`` starting with "Figure", searching forward until the end of the
       current section (the next ``TITLE`` or ``SECTION_HEADER``), even across other pictures and tables.
    2. Otherwise, the picture's own caption (``caption_text``) when it starts with "Figure" and is unused.
    3. Otherwise, the nearest unused "Figure" ``CAPTION`` before the picture, in the same section.
    4. Otherwise, ``Figure NNN``.

    :param items: All ``(item, depth)`` pairs from ``doc.iterate_items()``
    :param index: Position of the picture in ``items``
    :param picture: Docling ``PictureItem``
    :param doc: Docling document the picture belongs to
    :param number: Picture number (1-based), used in the fallback alt
    :param used_captions: Positions in ``items`` of captions already used; updated in place
    :returns: Alt text for the Markdown image
    """
    used = used_captions if used_captions is not None else set()
    used_texts = {_figure_caption_text(items[i][0]) for i in used}

    for position in range(index + 1, len(items)):
        item = items[position][0]
        if getattr(item, "label", None) in _SECTION_LABELS:
            break
        text = _figure_caption_text(item)
        if text and position not in used:
            used.add(position)
            return text

    caption_text = getattr(picture, "caption_text", None)
    if callable(caption_text):
        text = " ".join(caption_text(doc).split())
        if text.startswith(_FIGURE_CAPTION_PREFIX) and text not in used_texts:
            # Mark the matching caption item as used, so a later picture does not take it again
            used.update(i for i, (item, _depth) in enumerate(items) if _figure_caption_text(item) == text)
            return text

    for position in range(index - 1, -1, -1):
        item = items[position][0]
        if getattr(item, "label", None) in _SECTION_LABELS:
            break
        text = _figure_caption_text(item)
        if text and position not in used:
            used.add(position)
            return text

    return f"Figure {number:03d}"


def _render_picture(picture: Any, doc: Any, alt: str, png_path: Path, link: str) -> str:
    """Save a picture as PNG and render its Markdown image.

    When Docling has no image for the picture, nothing is saved and the alt text is rendered in
    italics instead, so the Markdown has no broken link.

    :param picture: Docling ``PictureItem``
    :param doc: Docling document the picture belongs to
    :param alt: Alt text
    :param png_path: Destination PNG file
    :param link: Link to the PNG, relative to the Markdown file
    :returns: Markdown block for the picture
    """
    image = picture.get_image(doc)
    if image is None:
        logger.warning("Picture has no image; skipping %s", png_path.name)
        return f"_{alt} (imagem indisponível)_"

    image.save(png_path, "PNG")
    return f"![{alt}]({link})"


def export_markdown(doc: Any, pdf_stem: str, output_dir: Path) -> tuple[Path, Path]:
    """Export a Docling document as ``{pdf_stem}.md`` plus a ``{pdf_stem}_assets/`` folder.

    The Markdown is assembled in a single pass over ``doc.iterate_items()``. Pictures are saved as
    ``figure_NNN.png`` and tables as ``table_NNN.csv``, each with its own counter starting at 001 in
    visit order. Existing files with the same names are overwritten.

    :param doc: Converted Docling document
    :param pdf_stem: PDF file name without extension
    :param output_dir: Folder where the Markdown and the assets folder are written
    :returns: Tuple of (Markdown file path, assets folder path)
    :raises PDFParsingException: If the files cannot be written
    """
    assets_dir = output_dir / f"{pdf_stem}_assets"
    markdown_path = output_dir / f"{pdf_stem}.md"
    figure_count = 0
    table_count = 0

    try:
        assets_dir.mkdir(parents=True, exist_ok=True)

        # Materialized once so the alt rule can look ahead; still a single assembly pass
        items = list(doc.iterate_items())
        used_captions: set[int] = set()
        blocks: list[str] = []
        for index, (item, depth) in enumerate(items):
            label = getattr(item, "label", None)
            if label == DocItemLabel.PICTURE:
                figure_count += 1
                file_name = f"figure_{figure_count:03d}.png"
                alt = _resolve_alt(items, index, item, doc, figure_count, used_captions)
                block = _render_picture(item, doc, alt, assets_dir / file_name, f"{assets_dir.name}/{file_name}")
            elif label == DocItemLabel.TABLE:
                table_count += 1
                block = _render_table(item, doc, assets_dir / f"table_{table_count:03d}.csv")
            else:
                block = _render_text_item(item, depth)

            if block:
                blocks.append(block)

        markdown_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    except OSError as e:
        logger.error("Failed to export Markdown to %s: %s", output_dir, e)
        raise PDFParsingException(f"Failed to export Markdown to {output_dir}: {e}") from e

    logger.info("Exported %s (%d figures, %d tables)", markdown_path, figure_count, table_count)
    return markdown_path, assets_dir
