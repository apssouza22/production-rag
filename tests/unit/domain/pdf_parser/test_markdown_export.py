from types import SimpleNamespace

from docling_core.types.doc import DocItemLabel, DoclingDocument, TableCell, TableData
from src.domain.pdf_parser.markdown_export import (
    _extract_gfm_table_lines,
    _parse_gfm_table,
    _render_table,
    _render_text_item,
    _write_table_csv,
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
