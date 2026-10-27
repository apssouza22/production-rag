"""Markdown export with the real Docling pipeline on an arXiv PDF fixture.

Slow: converts the PDF once (Docling downloads its models on the first run).
"""

import csv
import re
from pathlib import Path

import pytest
from src.domain.pdf_parser.docling import DoclingParser
from src.domain.pdf_parser.markdown_export import _parse_gfm_table

FIXTURE = Path(__file__).parents[2] / "fixtures" / "arxiv_pdfs" / "2609.17429v1.pdf"
STEM = "2609.17429v1"
IMAGE_ALT = re.compile(r"^!\[((?:\\.|[^\]\\])*)\]\(", flags=re.MULTILINE)
FIGURE_CAPTION_PARAGRAPH = re.compile(r"^(Figure \d+:.*)$", flags=re.MULTILINE)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Parse the fixture once with the Markdown export and share the result across tests."""
    output_dir = tmp_path_factory.mktemp("markdown_export")
    # CPU keeps the run deterministic and avoids GPUs the installed torch build does not support
    parser = DoclingParser(max_pages=30, max_file_size_mb=20, accelerator_device="cpu")
    content = parser.parse_pdf(FIXTURE, markdown_output_dir=output_dir)
    return content, output_dir


@pytest.fixture(scope="module")
def markdown(exported):
    content, _ = exported
    return content.markdown_path.read_text(encoding="utf-8")


def _unescape(text: str) -> str:
    return re.sub(r"\\(.)", r"\1", text)


def test_markdown_and_assets_are_written(exported):
    content, output_dir = exported

    assert content.markdown_path == output_dir / f"{STEM}.md"
    assert content.artifacts_dir == output_dir / f"{STEM}_assets"
    assert content.markdown_path.is_file()
    assert content.artifacts_dir.is_dir()


def test_sections_and_raw_text_are_still_extracted(exported):
    content, _ = exported

    assert content.sections
    assert content.raw_text


def test_every_figure_is_saved_and_linked(exported, markdown):
    content, _ = exported
    figures = sorted(p.name for p in content.artifacts_dir.glob("figure_*.png"))
    links = re.findall(rf"!\[(?:\\.|[^\]\\])*\]\({re.escape(STEM)}_assets/(figure_\d{{3}}\.png)\)", markdown)

    assert figures, "the fixture has figures"
    assert figures[0] == "figure_001.png"
    assert figures == [f"figure_{n:03d}.png" for n in range(1, len(figures) + 1)]
    assert sorted(links) == figures


def test_every_figure_alt_comes_from_its_own_caption(markdown):
    """Q5: no figure falls back to "Figure NNN" and each "Figure N:" caption is used by one figure only."""
    alts = [_unescape(alt) for alt in IMAGE_ALT.findall(markdown)]
    captions = FIGURE_CAPTION_PARAGRAPH.findall(markdown)

    assert alts
    assert not [alt for alt in alts if re.fullmatch(r"Figure \d{3}", alt)]
    assert sorted(alts) == sorted(set(captions))


def test_every_table_has_a_csv_and_no_csv_link(exported, markdown):
    content, _ = exported
    csvs = sorted(p.name for p in content.artifacts_dir.glob("table_*.csv"))
    inline_tables = re.findall(r"^\|.*\|\n\|[\s:|-]+\|$", markdown, flags=re.MULTILINE)

    assert csvs, "the fixture has tables"
    assert csvs == [f"table_{n:03d}.csv" for n in range(1, len(csvs) + 1)]
    assert len(inline_tables) == len(csvs)
    assert ".csv" not in markdown


def test_first_csv_mirrors_the_first_inline_table(exported, markdown):
    content, _ = exported
    lines = markdown.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("|"))
    end = next((i for i in range(start, len(lines)) if not lines[i].startswith("|")), len(lines))
    header, rows = _parse_gfm_table(lines[start:end])

    with open(content.artifacts_dir / "table_001.csv", encoding="utf-8", newline="") as f:
        csv_rows = list(csv.reader(f))

    assert csv_rows == ([header] if header is not None else []) + rows
