"""Markdown export with the real Docling pipeline on an arXiv PDF fixture.

Slow: converts the PDF once (Docling downloads its models on the first run).
"""

import re
from pathlib import Path

import pytest
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from src.domain.pdf_parser.docling import DoclingParser

FIXTURE = Path(__file__).parents[2] / "fixtures" / "arxiv_pdfs" / "2609.17429v1.pdf"
STEM = "2609.17429v1"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Parse the fixture once with the Markdown export and share the result across tests."""
    output_dir = tmp_path_factory.mktemp("markdown_export")
    parser = DoclingParser(max_pages=30, max_file_size_mb=20)
    # Run on CPU: keeps the test deterministic and avoids GPUs the installed torch build does not support.
    # DOCLING_DEVICE would not work here: Docling reads it once, when its pipeline options module is imported.
    pipeline_options = parser._converter.format_to_options[InputFormat.PDF].pipeline_options
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)

    content = parser.parse_pdf(FIXTURE, markdown_output_dir=output_dir)
    return content, output_dir


@pytest.fixture(scope="module")
def markdown(exported):
    content, _ = exported
    return content.markdown_path.read_text(encoding="utf-8")


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
    links = re.findall(rf"!\[[^\]]*\]\({re.escape(STEM)}_assets/(figure_\d{{3}}\.png)\)", markdown)

    assert figures, "the fixture has figures"
    assert figures[0] == "figure_001.png"
    assert figures == [f"figure_{n:03d}.png" for n in range(1, len(figures) + 1)]
    assert sorted(links) == figures


def test_figure_alts_come_from_captions(markdown):
    alts = re.findall(r"^!\[([^\]]*)\]\(", markdown, flags=re.MULTILINE)

    assert any(re.match(r"Figure \d+:", alt) for alt in alts)
    assert len(set(alts)) == len(alts), "each caption is used by one figure only"


def test_every_table_has_a_csv_and_no_csv_link(exported, markdown):
    content, _ = exported
    csvs = sorted(p.name for p in content.artifacts_dir.glob("table_*.csv"))
    inline_tables = re.findall(r"^\|.*\|\n\|[\s:|-]+\|$", markdown, flags=re.MULTILINE)

    assert csvs, "the fixture has tables"
    assert csvs == [f"table_{n:03d}.csv" for n in range(1, len(csvs) + 1)]
    assert len(inline_tables) == len(csvs)
    assert ".csv" not in markdown
