"""Markdown export of a Docling document, with figures and tables saved as assets.

The Markdown is assembled in a single pass over ``doc.iterate_items()``: text items become
Markdown blocks, tables stay inline as GFM pipe tables (and are also saved as CSV), and
pictures are saved as PNG and linked from the Markdown (a picture without an image is rendered
as its italic alt text followed by ``(image unavailable)``).
"""

import bisect
import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from docling_core.types.doc import DocItemLabel
from src.domain.pdf_parser.exceptions import PDFParsingException

logger = logging.getLogger(__name__)

_MAX_HEADING_LEVEL = 6
_SECTION_LABELS = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}
# Docling labels a PictureItem as PICTURE or CHART and a TableItem as TABLE or DOCUMENT_INDEX
_PICTURE_LABELS = {DocItemLabel.PICTURE, DocItemLabel.CHART}
_TABLE_LABELS = {DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX}
_FIGURE_CAPTION_PREFIX = "Figure"

# A GFM header separator row, e.g. "|:----|-----:|" or "| --- | --- |"
_GFM_SEPARATOR_CELL = re.compile(r"^:?-+:?$")
# Asset files written by a previous export, removed before exporting again (counters may exceed 999)
_ASSET_FILE = re.compile(r"^(figure_\d{3,}\.png|table_\d{3,}\.csv)$")
# Characters that would change the meaning of inline Markdown inside the alt text (links, emphasis,
# code, HTML tags and entities)
_MARKDOWN_INLINE_SPECIALS = re.compile(r"([\\`*_\[\]<&])")
_BACKTICK_RUN = re.compile(r"`+")
_MISSING_IMAGE_NOTE = "(image unavailable)"


def _code_fence(text: str) -> str:
    """Return a backtick fence longer than any backtick run in the text."""
    longest_run = max((len(run) for run in _BACKTICK_RUN.findall(text)), default=0)
    return "`" * max(3, longest_run + 1)


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
        fence = _code_fence(text)
        return f"{fence}\n{text}\n{fence}"
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


def _normalize(text: str) -> str:
    """Collapse whitespace and line breaks into single spaces."""
    return " ".join(text.split())


@dataclass
class _CaptionIndex:
    """Positions of "Figure" captions and section starts in ``iterate_items`` order.

    Built once per document so each picture only looks at caption positions, not at every item.
    Captions are tracked by position: each one is used by at most one picture. Caption texts that
    were not visited (only linked to a picture) are tracked by text.
    """

    captions: dict[int, str]
    section_starts: list[int]
    used: set[int] = field(default_factory=set)
    used_unvisited_texts: set[str] = field(default_factory=set)
    positions: list[int] = field(init=False)
    by_text: dict[str, list[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.positions = sorted(self.captions)
        self.by_text = {}
        for position in self.positions:
            self.by_text.setdefault(self.captions[position], []).append(position)

    @classmethod
    def build(cls, items: list[tuple[Any, int]]) -> "_CaptionIndex":
        captions: dict[int, str] = {}
        section_starts: list[int] = []
        for position, (item, _depth) in enumerate(items):
            label = getattr(item, "label", None)
            if label in _SECTION_LABELS:
                section_starts.append(position)
            elif label == DocItemLabel.CAPTION:
                text = _normalize(getattr(item, "text", ""))
                if text.startswith(_FIGURE_CAPTION_PREFIX):
                    captions[position] = text
        return cls(captions=captions, section_starts=section_starts)

    def _section_bounds(self, index: int) -> tuple[int, Optional[int]]:
        """Return the positions of the section start before ``index`` (-1 if none) and of the next one (None if none)."""
        after = bisect.bisect_right(self.section_starts, index)
        start = self.section_starts[after - 1] if after > 0 else -1
        end = self.section_starts[after] if after < len(self.section_starts) else None
        return start, end

    def _take(self, position: int) -> str:
        self.used.add(position)
        return self.captions[position]

    def take_next(self, index: int) -> Optional[str]:
        """Take the first unused caption after ``index``, in the same section."""
        _start, end = self._section_bounds(index)
        for position in self.positions[bisect.bisect_right(self.positions, index) :]:
            if end is not None and position > end:
                break
            if position not in self.used:
                return self._take(position)
        return None

    def take_previous(self, index: int) -> Optional[str]:
        """Take the nearest unused caption before ``index``, in the same section."""
        start, _end = self._section_bounds(index)
        for i in range(bisect.bisect_left(self.positions, index) - 1, -1, -1):
            position = self.positions[i]
            if position < start:
                break
            if position not in self.used:
                return self._take(position)
        return None

    def take_text(self, text: str) -> Optional[str]:
        """Take a caption with exactly this text, unless it is already used.

        A text with no matching visited caption is returned once, then treated as used.
        """
        positions = self.by_text.get(text, [])
        if not positions:
            if text in self.used_unvisited_texts:
                return None
            self.used_unvisited_texts.add(text)
            return text
        for position in positions:
            if position not in self.used:
                return self._take(position)
        return None


def _linked_caption_texts(picture: Any, doc: Any) -> list[str]:
    """Return the normalized text of each caption Docling linked to the picture.

    Each caption is resolved on its own: ``caption_text`` would glue several captions together.
    """
    caption_refs = getattr(picture, "captions", None)
    if caption_refs:
        return [_normalize(getattr(ref.resolve(doc), "text", "")) for ref in caption_refs]
    caption_text = getattr(picture, "caption_text", None)
    return [_normalize(caption_text(doc))] if callable(caption_text) else []


def _resolve_alt(index: int, picture: Any, doc: Any, number: int, captions: _CaptionIndex) -> str:
    """Resolve the alt text of a picture.

    Each "Figure" caption is used by at most one picture, because Docling may place several captions
    after a group of pictures, or a caption before its picture. In order:

    1. The next unused ``CAPTION`` starting with "Figure", searching forward until the end of the
       current section (the next ``TITLE`` or ``SECTION_HEADER``), even across other pictures and tables.
    2. Otherwise, a caption linked to the picture (each of ``picture.captions``) that starts with "Figure"
       and is unused.
    3. Otherwise, the nearest unused "Figure" ``CAPTION`` before the picture, in the same section.
    4. Otherwise, ``Figure NNN``.

    :param index: Position of the picture in the ``iterate_items`` sequence
    :param picture: Docling ``PictureItem``
    :param doc: Docling document the picture belongs to
    :param number: Picture number (1-based), used in the fallback alt
    :param captions: Caption index of the document; the chosen caption is marked as used
    :returns: Alt text for the Markdown image (not escaped)
    """
    alt = captions.take_next(index)
    if alt:
        return alt

    for text in _linked_caption_texts(picture, doc):
        if text.startswith(_FIGURE_CAPTION_PREFIX):
            alt = captions.take_text(text)
            if alt:
                return alt

    alt = captions.take_previous(index)
    if alt:
        return alt

    return f"Figure {number:03d}"


def _escape_markdown_inline(text: str) -> str:
    """Escape characters that would change the meaning of inline Markdown (links, emphasis, code)."""
    return _MARKDOWN_INLINE_SPECIALS.sub(r"\\\1", text)


def _link_destination(link: str) -> str:
    """Percent-encode a relative link destination (spaces, parentheses, ``#``, ``%``, ``?``...), keeping ``/``."""
    return quote(link, safe="/")


def _render_picture(picture: Any, doc: Any, alt: str, png_path: Path, link: str) -> str:
    """Save a picture as PNG and render its Markdown image.

    When Docling has no image for the picture, nothing is saved and the alt text is rendered in
    italics instead, so the Markdown has no broken link.

    :param picture: Docling ``PictureItem``
    :param doc: Docling document the picture belongs to
    :param alt: Alt text (escaped here)
    :param png_path: Destination PNG file
    :param link: Link to the PNG, relative to the Markdown file
    :returns: Markdown block for the picture
    """
    escaped_alt = _escape_markdown_inline(alt)
    image = picture.get_image(doc)
    if image is None:
        logger.warning("Picture has no image; skipping %s", png_path.name)
        return f"_{escaped_alt} {_MISSING_IMAGE_NOTE}_"

    image.save(png_path, "PNG")
    return f"![{escaped_alt}]({_link_destination(link)})"


def _clear_previous_assets(assets_dir: Path) -> None:
    """Remove figure and table files left by a previous export of the same PDF."""
    for path in assets_dir.iterdir():
        if path.is_file() and _ASSET_FILE.match(path.name):
            path.unlink()


def export_markdown(doc: Any, pdf_stem: str, output_dir: Path) -> tuple[Path, Path]:
    """Export a Docling document as ``{pdf_stem}.md`` plus a ``{pdf_stem}_assets/`` folder.

    The Markdown is assembled in a single pass over ``doc.iterate_items()``. Pictures are saved as
    ``figure_NNN.png`` and tables as ``table_NNN.csv``, each with its own counter starting at 001 in
    visit order. A previous export of the same PDF in ``output_dir`` is replaced: its Markdown and its
    ``figure_NNN.png``/``table_NNN.csv`` files are removed first, so a failed export never leaves an
    old Markdown pointing at removed assets.

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
        markdown_path.unlink(missing_ok=True)
        _clear_previous_assets(assets_dir)

        # Materialized once so the alt rule can look around the picture; still a single assembly pass
        items = list(doc.iterate_items())
        captions = _CaptionIndex.build(items)
        blocks: list[str] = []
        for index, (item, depth) in enumerate(items):
            label = getattr(item, "label", None)
            if label in _PICTURE_LABELS:
                figure_count += 1
                file_name = f"figure_{figure_count:03d}.png"
                alt = _resolve_alt(index, item, doc, figure_count, captions)
                block = _render_picture(item, doc, alt, assets_dir / file_name, f"{assets_dir.name}/{file_name}")
            elif label in _TABLE_LABELS:
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
