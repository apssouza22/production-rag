from src.domain.indexing.text_chunker import TextChunker


def _section(title: str, content: str, breadcrumb: str | None = None, level: int = 1) -> dict:
    return {
        "title": title,
        "content": content,
        "level": level,
        "breadcrumb": breadcrumb,
    }


class TestTextChunkerBreadcrumbs:
    def test_section_chunk_uses_breadcrumb_in_text_and_metadata(self):
        chunker = TextChunker(chunk_size=600, overlap_size=100, min_chunk_size=100)
        content = " ".join(["word"] * 120)

        chunks = chunker.chunk_paper(
            title="Sample Paper",
            abstract="An abstract about retrieval.",
            full_text=content,
            arxiv_id="2401.00001",
            paper_id="paper-1",
            sections=[
                _section(
                    title="Training",
                    content=content,
                    breadcrumb="Methods > Training",
                    level=2,
                )
            ],
        )

        assert len(chunks) == 1
        assert "Section: Methods > Training" in chunks[0].text
        assert chunks[0].metadata.section_title == "Training"
        assert chunks[0].metadata.section_breadcrumb == "Methods > Training"

    def test_parses_legacy_sections_without_breadcrumb(self):
        chunker = TextChunker(chunk_size=600, overlap_size=100, min_chunk_size=100)
        content = " ".join(["word"] * 120)

        chunks = chunker.chunk_paper(
            title="Sample Paper",
            abstract="An abstract about retrieval.",
            full_text=content,
            arxiv_id="2401.00002",
            paper_id="paper-2",
            sections={"Introduction": content},
        )

        assert len(chunks) == 1
        assert "Section: Introduction" in chunks[0].text
        assert chunks[0].metadata.section_breadcrumb is None
