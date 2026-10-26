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

logger = logging.getLogger(__name__)

_MAX_HEADING_LEVEL = 6

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
