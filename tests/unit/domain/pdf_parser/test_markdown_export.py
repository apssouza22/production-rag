import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument, ImageRef, TableCell, TableData
from PIL import Image
from src.domain.pdf_parser.exceptions import PDFParsingException
from src.domain.pdf_parser.markdown_export import (
    _CaptionIndex,
    _extract_gfm_table_lines,
    _parse_gfm_table,
    _render_picture,
    _render_table,
    _render_text_item,
    _resolve_alt,
    _write_table_csv,
    export_markdown,
)


def _item(label: DocItemLabel, text: str, level: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(label=label, text=text, level=level)


def _table_data(rows: list[list[str]], header: bool = True) -> TableData:
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=header and r == 0,
        )
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
    ]
    return TableData(num_rows=len(rows), num_cols=len(rows[0]), table_cells=cells)


class _FakeImage:
    def save(self, path, fmt):
        Path(path).write_bytes(b"png")


def _picture(caption: str = "", has_image: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        label=DocItemLabel.PICTURE,
        text="",
        get_image=lambda doc: _FakeImage() if has_image else None,
        caption_text=lambda doc: caption,
    )


def _table(markdown: str) -> SimpleNamespace:
    return SimpleNamespace(label=DocItemLabel.TABLE, text="", export_to_markdown=lambda doc: markdown)


def _fake_doc(items: list) -> SimpleNamespace:
    def _forbidden_export():
        raise AssertionError("doc.export_to_markdown() must not be called")

    return SimpleNamespace(iterate_items=lambda: iter((item, 1) for item in items), export_to_markdown=_forbidden_export)


def _add_picture_with_caption(doc: DoclingDocument, caption: str | None):
    """Mimic Docling's PDF pipeline: the caption is a child of the picture and listed in its captions."""
    picture = doc.add_picture(image=ImageRef.from_pil(Image.new("RGB", (8, 8), "white"), dpi=72))
    if caption is not None:
        caption_item = doc.add_text(label=DocItemLabel.CAPTION, text=caption, parent=picture)
        picture.captions.append(caption_item.get_ref())
    return picture


def _read_csv(path) -> str:
    # Read raw bytes: read_text() would translate the csv module's "\r\n" line endings
    return path.read_bytes().decode("utf-8")


class TestRenderTextItem:
    def test_title_is_top_level_heading(self):
        assert _render_text_item(_item(DocItemLabel.TITLE, "Paper Title"), depth=1) == "# Paper Title"

    def test_section_header_uses_item_level(self):
        assert _render_text_item(_item(DocItemLabel.SECTION_HEADER, "Methods", level=1), depth=1) == "## Methods"
        assert _render_text_item(_item(DocItemLabel.SECTION_HEADER, "Training", level=2), depth=2) == "### Training"

    def test_section_header_falls_back_to_depth(self):
        assert _render_text_item(_item(DocItemLabel.SECTION_HEADER, "Results"), depth=2) == "### Results"

    def test_section_header_level_is_capped(self):
        assert _render_text_item(_item(DocItemLabel.SECTION_HEADER, "Deep", level=9), depth=1) == "###### Deep"

    def test_list_item(self):
        assert _render_text_item(_item(DocItemLabel.LIST_ITEM, "First point"), depth=2) == "- First point"

    def test_code_is_fenced(self):
        assert _render_text_item(_item(DocItemLabel.CODE, "x = 1"), depth=1) == "```\nx = 1\n```"

    def test_other_text_labels_are_paragraphs(self):
        for label in (DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.CAPTION, DocItemLabel.FORMULA):
            assert _render_text_item(_item(label, "  Some text.  "), depth=1) == "Some text."

    def test_empty_text_is_skipped(self):
        assert _render_text_item(_item(DocItemLabel.PARAGRAPH, "   "), depth=1) is None


class TestParseGfmTable:
    def test_with_header_separator(self):
        lines = ["| Model | Acc |", "|:------|----:|", "| A     | 0.9 |"]

        header, rows = _parse_gfm_table(lines)

        assert header == ["Model", "Acc"]
        assert rows == [["A", "0.9"]]

    def test_headerless_grid_keeps_only_data_rows(self):
        lines = ["| a | b |", "| c | d |"]

        header, rows = _parse_gfm_table(lines)

        assert header is None
        assert rows == [["a", "b"], ["c", "d"]]

    def test_escaped_pipe_stays_inside_cell(self):
        header, rows = _parse_gfm_table(["| Expr | Value |", "|---|---|", "| a \\| b | 1 |"])

        assert header == ["Expr", "Value"]
        assert rows == [["a | b", "1"]]

    def test_caption_lines_are_dropped(self):
        markdown = "Table 1: scores\n\n| Model | Acc |\n|---|---|\n| A | 0.9 |"

        assert _extract_gfm_table_lines(markdown) == ["| Model | Acc |", "|---|---|", "| A | 0.9 |"]


class TestWriteTableCsv:
    def test_csv_format(self, tmp_path):
        csv_path = tmp_path / "table_001.csv"

        _write_table_csv(csv_path, ["Modelo", "Métrica"], [["A, v2", "0.9"], ["Ação", "1"]])

        assert _read_csv(csv_path) == 'Modelo,Métrica\r\n"A, v2",0.9\r\nAção,1\r\n'

    def test_headerless_csv_has_only_data_rows(self, tmp_path):
        csv_path = tmp_path / "table_001.csv"

        _write_table_csv(csv_path, None, [["a", "b"]])

        assert _read_csv(csv_path) == "a,b\r\n"


class TestRenderTable:
    def test_returns_table_markdown_without_csv_link(self, tmp_path):
        table = SimpleNamespace(export_to_markdown=lambda doc: "| Model | Acc |\n|---|---|\n| A | 0.9 |")
        csv_path = tmp_path / "table_001.csv"

        markdown = _render_table(table, doc=None, csv_path=csv_path)

        assert markdown == "| Model | Acc |\n|---|---|\n| A | 0.9 |"
        assert ".csv" not in markdown
        assert _read_csv(csv_path) == "Model,Acc\r\nA,0.9\r\n"

    def test_real_docling_table_drops_caption_and_mirrors_grid(self, tmp_path):
        doc = DoclingDocument(name="paper")
        table = doc.add_table(data=_table_data([["Model", "Acc"], ["A, v2", "0.9"]]))
        caption = doc.add_text(label=DocItemLabel.CAPTION, text="Table 1: scores", parent=table)
        table.captions.append(caption.get_ref())
        csv_path = tmp_path / "table_001.csv"

        markdown = _render_table(table, doc, csv_path)

        assert "Table 1: scores" not in markdown
        assert markdown.splitlines()[0].startswith("| Model")
        assert _read_csv(csv_path) == 'Model,Acc\r\n"A, v2",0.9\r\n'


class TestResolveAlt:
    def _alt(self, items, picture_index=0, doc=None):
        captions = _CaptionIndex.build([(item, 1) for item in items])
        return _resolve_alt(picture_index, items[picture_index], doc, picture_index + 1, captions)

    def test_next_figure_caption_after_text_and_table_in_same_section(self):
        items = [
            _picture(),
            _item(DocItemLabel.PARAGRAPH, "Some text."),
            _table("| a |\n|---|\n| b |"),
            _item(DocItemLabel.CAPTION, "Table 1: not a figure"),
            _item(DocItemLabel.CAPTION, "Figure 1: Architecture overview"),
        ]

        assert self._alt(items) == "Figure 1: Architecture overview"

    def test_caption_after_section_header_is_not_used(self):
        items = [
            _picture(),
            _item(DocItemLabel.SECTION_HEADER, "Results", level=1),
            _item(DocItemLabel.CAPTION, "Figure 2: Results"),
        ]

        assert self._alt(items) == "Figure 001"

    def test_falls_back_to_picture_caption_text(self):
        items = [_picture(caption="Figure 3: Pipeline"), _item(DocItemLabel.SECTION_HEADER, "Next", level=1)]

        assert self._alt(items) == "Figure 3: Pipeline"

    def test_picture_caption_without_figure_prefix_uses_number(self):
        items = [_picture(caption="Overview of the system")]

        assert self._alt(items) == "Figure 001"

    def test_picture_without_caption_takes_next_figure_caption_in_section(self):
        # Accepted behaviour (Q2): the search does not stop at the next picture
        items = [_picture(), _picture(), _item(DocItemLabel.CAPTION, "Figure 2: Second")]

        assert self._alt(items, picture_index=0) == "Figure 2: Second"

    def test_caption_whitespace_is_collapsed(self):
        items = [_picture(), _item(DocItemLabel.CAPTION, "Figure 1:\n  Two   lines")]

        assert self._alt(items) == "Figure 1: Two lines"

    def _alts(self, items):
        """Resolve the alt of every picture in order, sharing the used captions (as export_markdown does)."""
        captions = _CaptionIndex.build([(item, 1) for item in items])
        pictures = [i for i, item in enumerate(items) if item.label == DocItemLabel.PICTURE]
        return [_resolve_alt(i, items[i], None, n, captions) for n, i in enumerate(pictures, start=1)]

    def test_captions_grouped_after_pictures_are_used_once_each(self):
        # Seen in the arXiv fixture: two pictures followed by "Figure 1" and "Figure 2"
        items = [
            _picture(),
            _picture(caption="Figure 1: First"),
            _item(DocItemLabel.CAPTION, "Figure 1: First"),
            _item(DocItemLabel.CAPTION, "Figure 2: Second"),
        ]

        assert self._alts(items) == ["Figure 1: First", "Figure 2: Second"]

    def test_caption_before_picture_is_used_when_none_follows(self):
        # Seen in the arXiv fixture: "Figure 7" comes before its picture
        items = [
            _item(DocItemLabel.CAPTION, "Table 9: Not a figure"),
            _item(DocItemLabel.CAPTION, "Figure 7: Evidence"),
            _picture(),
        ]

        assert self._alts(items) == ["Figure 7: Evidence"]

    def test_look_back_stops_at_section_header(self):
        items = [
            _item(DocItemLabel.CAPTION, "Figure 1: Other section"),
            _item(DocItemLabel.SECTION_HEADER, "Results", level=1),
            _picture(),
        ]

        assert self._alts(items) == ["Figure 001"]

    def test_used_caption_is_not_taken_again_by_look_back_or_caption_text(self):
        items = [
            _picture(),
            _item(DocItemLabel.CAPTION, "Figure 1: Only caption"),
            _picture(caption="Figure 1: Only caption"),
        ]

        assert self._alts(items) == ["Figure 1: Only caption", "Figure 002"]


class TestRenderPicture:
    def test_saves_png_and_links_it(self, tmp_path):
        png_path = tmp_path / "figure_001.png"

        block = _render_picture(_picture(), None, "Figure 1: Arch", png_path, "paper_assets/figure_001.png")

        assert block == "![Figure 1: Arch](paper_assets/figure_001.png)"
        assert png_path.exists()

    def test_missing_image_renders_italic_alt_without_file(self, tmp_path):
        png_path = tmp_path / "figure_001.png"

        block = _render_picture(_picture(has_image=False), None, "Figure 001", png_path, "paper_assets/figure_001.png")

        assert block == "_Figure 001 (image unavailable)_"
        assert not png_path.exists()


class TestExportMarkdown:
    def test_mixed_document_order_counters_and_files(self, tmp_path):
        doc = _fake_doc(
            [
                _item(DocItemLabel.TITLE, "Paper"),
                _item(DocItemLabel.PARAGRAPH, "Intro."),
                _picture(),
                _item(DocItemLabel.CAPTION, "Figure 1: First"),
                _table("| a | b |\n|---|---|\n| 1 | 2 |"),
                _picture(has_image=False),
                _table("| c |\n|---|\n| 3 |"),
            ]
        )

        markdown_path, assets_dir = export_markdown(doc, "paper", tmp_path)

        assert markdown_path == tmp_path / "paper.md"
        assert assets_dir == tmp_path / "paper_assets"
        assert sorted(p.name for p in assets_dir.iterdir()) == ["figure_001.png", "table_001.csv", "table_002.csv"]
        assert markdown_path.read_text(encoding="utf-8") == (
            "# Paper\n\n"
            "Intro.\n\n"
            "![Figure 1: First](paper_assets/figure_001.png)\n\n"
            "Figure 1: First\n\n"
            "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
            "_Figure 002 (image unavailable)_\n\n"
            "| c |\n|---|\n| 3 |\n"
        )

    def test_document_without_assets_creates_empty_assets_dir(self, tmp_path):
        output_dir = tmp_path / "nested" / "out"

        markdown_path, assets_dir = export_markdown(_fake_doc([_item(DocItemLabel.PARAGRAPH, "Only text.")]), "paper", output_dir)

        assert markdown_path.read_text(encoding="utf-8") == "Only text.\n"
        assert assets_dir.is_dir()
        assert list(assets_dir.iterdir()) == []

    def test_overwrites_previous_export(self, tmp_path):
        export_markdown(_fake_doc([_item(DocItemLabel.PARAGRAPH, "Old.")]), "paper", tmp_path)

        markdown_path, _ = export_markdown(_fake_doc([_item(DocItemLabel.PARAGRAPH, "New.")]), "paper", tmp_path)

        assert markdown_path.read_text(encoding="utf-8") == "New.\n"

    def test_write_failure_raises_pdf_parsing_exception(self, tmp_path):
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            with pytest.raises(PDFParsingException) as exc_info:
                export_markdown(_fake_doc([_item(DocItemLabel.PARAGRAPH, "Text.")]), "paper", tmp_path)

        assert "Failed to export Markdown" in str(exc_info.value)
        assert "disk full" in str(exc_info.value)

    def test_real_docling_document(self, tmp_path):
        doc = DoclingDocument(name="paper")
        doc.add_title("Paper")
        doc.add_heading("Method", level=1)
        _add_picture_with_caption(doc, "Figure 1: Architecture")
        table = doc.add_table(data=_table_data([["Model", "Acc"], ["A", "0.9"]]))
        table_caption = doc.add_text(label=DocItemLabel.CAPTION, text="Table 1: Scores", parent=table)
        table.captions.append(table_caption.get_ref())
        _add_picture_with_caption(doc, None)

        markdown_path, assets_dir = export_markdown(doc, "paper", tmp_path)
        markdown = markdown_path.read_text(encoding="utf-8")

        assert (assets_dir / "figure_001.png").read_bytes().startswith(b"\x89PNG")
        assert (assets_dir / "figure_002.png").exists()
        assert "![Figure 1: Architecture](paper_assets/figure_001.png)" in markdown
        assert "![Figure 002](paper_assets/figure_002.png)" in markdown
        assert markdown.count("Table 1: Scores") == 1
        assert ".csv" not in markdown
        assert _read_csv(assets_dir / "table_001.csv") == "Model,Acc\r\nA,0.9\r\n"

    def test_table_without_grid_still_advances_counter(self, tmp_path):
        """A table whose Markdown has no pipe rows writes an empty CSV, adds no block and keeps numbering."""
        doc = _fake_doc([_table("Table 1: caption only"), _table("| c |\n|---|\n| 3 |")])

        markdown_path, assets_dir = export_markdown(doc, "paper", tmp_path)

        assert _read_csv(assets_dir / "table_001.csv") == ""
        assert _read_csv(assets_dir / "table_002.csv") == "c\r\n3\r\n"
        assert markdown_path.read_text(encoding="utf-8") == "| c |\n|---|\n| 3 |\n"


class TestResolveAltOrder:
    """Search order and section limits of the alt rule (Q5)."""

    def _alts(self, items):
        captions = _CaptionIndex.build([(item, 1) for item in items])
        pictures = [i for i, item in enumerate(items) if item.label == DocItemLabel.PICTURE]
        return [_resolve_alt(i, items[i], None, n, captions) for n, i in enumerate(pictures, start=1)]

    def test_forward_search_comes_before_caption_text(self):
        # Q5 order: a visited caption ahead wins over the picture's own caption_text, even when it was
        # meant for the next picture. In Docling's PDF pipeline the own caption is visited right after
        # the picture, so this only happens when that caption is missing from the iteration.
        items = [_picture(caption="Figure 1: A"), _picture(), _item(DocItemLabel.CAPTION, "Figure 2: B")]

        assert self._alts(items) == ["Figure 2: B", "Figure 002"]

    def test_look_back_takes_nearest_caption(self):
        items = [
            _item(DocItemLabel.CAPTION, "Figure 1: Far"),
            _item(DocItemLabel.CAPTION, "Figure 2: Near"),
            _picture(),
        ]

        assert self._alts(items) == ["Figure 2: Near"]

    def test_title_also_limits_both_searches(self):
        items = [
            _item(DocItemLabel.CAPTION, "Figure 1: Before title"),
            _item(DocItemLabel.TITLE, "Appendix"),
            _picture(),
            _item(DocItemLabel.TITLE, "Another part"),
            _item(DocItemLabel.CAPTION, "Figure 2: After title"),
        ]

        assert self._alts(items) == ["Figure 001"]

    def test_caption_text_takes_matching_caption_outside_the_section(self):
        # Docling linked the caption to the picture, but the caption was visited in another section
        items = [
            _item(DocItemLabel.CAPTION, "Figure 1: Linked"),
            _item(DocItemLabel.SECTION_HEADER, "Results", level=1),
            _picture(caption="Figure 1: Linked"),
            _picture(),
        ]

        assert self._alts(items) == ["Figure 1: Linked", "Figure 002"]


class TestMarkdownSafety:
    def test_alt_special_characters_are_escaped(self, tmp_path):
        block = _render_picture(
            _picture(), None, "Figure 1: values in [0, 1) for F_1 *x*", tmp_path / "f.png", "paper_assets/figure_001.png"
        )

        assert block == r"![Figure 1: values in \[0, 1) for F\_1 \*x\*](paper_assets/figure_001.png)"

    def test_missing_image_fallback_escapes_alt(self, tmp_path):
        block = _render_picture(_picture(has_image=False), None, "Figure 1: F_1", tmp_path / "f.png", "a/figure_001.png")

        assert block == r"_Figure 1: F\_1 (image unavailable)_"

    def test_link_is_percent_encoded(self, tmp_path):
        markdown_path, _ = export_markdown(_fake_doc([_picture()]), "my paper (v2)#<1>%?", tmp_path)

        assert "![Figure 001](my%20paper%20%28v2%29%23%3C1%3E%25%3F_assets/figure_001.png)" in markdown_path.read_text(
            encoding="utf-8"
        )

    def test_arxiv_stem_link_is_unchanged(self, tmp_path):
        markdown_path, _ = export_markdown(_fake_doc([_picture()]), "2609.17429v1", tmp_path)

        assert "![Figure 001](2609.17429v1_assets/figure_001.png)" in markdown_path.read_text(encoding="utf-8")

    def test_alt_backslash_backtick_bracket_html_and_entity_are_escaped(self, tmp_path):
        block = _render_picture(_picture(), None, r"Figure 1: a\b `c` d] <b> &amp;", tmp_path / "f.png", "a/figure_001.png")

        assert block == r"![Figure 1: a\\b \`c\` d\] \<b> \&amp;](a/figure_001.png)"

    def test_code_fence_is_longer_than_backticks_in_text(self):
        block = _render_text_item(_item(DocItemLabel.CODE, "Use ```json blocks```"), depth=1)

        assert block == "````\nUse ```json blocks```\n````"


class TestExportMarkdownLabelsAndReexport:
    def test_document_index_table_and_chart_picture_are_exported(self, tmp_path):
        doc = DoclingDocument(name="paper")
        doc.add_table(data=_table_data([["Section", "Page"], ["Intro", "1"]]), label=DocItemLabel.DOCUMENT_INDEX)
        chart = _add_picture_with_caption(doc, "Figure 1: A chart")
        chart.label = DocItemLabel.CHART

        markdown_path, assets_dir = export_markdown(doc, "paper", tmp_path)
        markdown = markdown_path.read_text(encoding="utf-8")

        assert sorted(p.name for p in assets_dir.iterdir()) == ["figure_001.png", "table_001.csv"]
        assert "| Section" in markdown
        assert "![Figure 1: A chart](paper_assets/figure_001.png)" in markdown

    def test_reexport_removes_assets_from_previous_export(self, tmp_path):
        export_markdown(_fake_doc([_picture(), _picture(), _table("| a |\n|---|\n| 1 |")]), "paper", tmp_path)
        (tmp_path / "paper_assets" / "notes.txt").write_text("keep me", encoding="utf-8")

        _, assets_dir = export_markdown(_fake_doc([_picture()]), "paper", tmp_path)

        assert sorted(p.name for p in assets_dir.iterdir()) == ["figure_001.png", "notes.txt"]

    def test_reexport_keeps_similar_names_and_folders_and_clears_large_counters(self, tmp_path):
        assets_dir = tmp_path / "paper_assets"
        assets_dir.mkdir()
        (assets_dir / "figure_1.png").write_bytes(b"x")
        (assets_dir / "table_001.csv").mkdir()
        (assets_dir / "figure_1000.png").write_bytes(b"x")
        (tmp_path / "paper.md").write_text("old", encoding="utf-8")

        markdown_path, _ = export_markdown(_fake_doc([_item(DocItemLabel.PARAGRAPH, "New.")]), "paper", tmp_path)

        assert sorted(p.name for p in assets_dir.iterdir()) == ["figure_1.png", "table_001.csv"]
        assert (assets_dir / "table_001.csv").is_dir()
        assert markdown_path.read_text(encoding="utf-8") == "New.\n"

    def test_failed_reexport_leaves_no_stale_markdown(self, tmp_path):
        export_markdown(_fake_doc([_picture()]), "paper", tmp_path)

        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            with pytest.raises(PDFParsingException):
                export_markdown(_fake_doc([_picture()]), "paper", tmp_path)

        assert not (tmp_path / "paper.md").exists()


class TestLinkedCaptions:
    """Captions Docling linked to a picture are matched one by one (not the glued caption_text)."""

    def test_picture_with_two_linked_captions_takes_the_visited_one(self, tmp_path):
        doc = DoclingDocument(name="paper")
        doc.add_heading("Method", level=1)
        first = _add_picture_with_caption(doc, "Figure 1: A")
        second_caption = doc.add_text(label=DocItemLabel.CAPTION, text="Figure 1b: B", parent=first)
        first.captions.append(second_caption.get_ref())

        markdown_path, _ = export_markdown(doc, "paper", tmp_path)
        alts = re.findall(r"^!\[([^\]]*)\]", markdown_path.read_text(encoding="utf-8"), flags=re.MULTILINE)

        assert alts == ["Figure 1: A"]

    def test_same_unvisited_linked_caption_is_used_once(self):
        items = [_picture(caption="Figure 9: Ghost"), _picture(caption="Figure 9: Ghost")]
        captions = _CaptionIndex.build([(item, 1) for item in items])

        alts = [_resolve_alt(i, items[i], None, i + 1, captions) for i in range(2)]

        assert alts == ["Figure 9: Ghost", "Figure 002"]

    def test_linked_caption_ref_is_used_when_forward_search_finds_none(self, tmp_path):
        doc = DoclingDocument(name="paper")
        caption = doc.add_text(label=DocItemLabel.CAPTION, text="Figure 3: Linked")
        doc.add_heading("Results", level=1)
        picture = doc.add_picture(image=ImageRef.from_pil(Image.new("RGB", (8, 8), "white"), dpi=72))
        picture.captions.append(caption.get_ref())

        markdown_path, _ = export_markdown(doc, "paper", tmp_path)

        assert "![Figure 3: Linked](paper_assets/figure_001.png)" in markdown_path.read_text(encoding="utf-8")
