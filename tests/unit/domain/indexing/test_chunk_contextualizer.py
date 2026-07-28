import pytest

from src.domain.indexing.chunk_contextualizer import ChunkContextualizer
from src.domain.indexing.models import ChunkMetadata, TextChunk


def _make_chunk(text: str = "chunk body", chunk_index: int = 0) -> TextChunk:
    return TextChunk(
        text=text,
        metadata=ChunkMetadata(
            chunk_index=chunk_index,
            start_char=0,
            end_char=len(text),
            word_count=len(text.split()),
            overlap_with_previous=0,
            overlap_with_next=0,
        ),
        arxiv_id="2301.00001",
        paper_id="1",
    )


class MockLLMClient:
    def __init__(self, response: str = "Context about transformers in NLP."):
        self.response = response
        self.calls = []

    async def generate(self, model: str, prompt: str, **kwargs):
        self.calls.append({"model": model, "prompt": prompt, **kwargs})
        return {"response": self.response}


@pytest.mark.asyncio
async def test_contextualize_chunks_attaches_context():
    llm = MockLLMClient(response="BERT is a transformer model for language understanding.")
    contextualizer = ChunkContextualizer(llm_client=llm, model="test-model", max_concurrent_requests=2)

    chunks = [_make_chunk("BERT uses bidirectional attention."), _make_chunk("Results on GLUE benchmark.")]

    result = await contextualizer.contextualize_chunks(
        chunks=chunks,
        document="Full paper about BERT and transformer architectures.",
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        abstract="We introduce BERT for NLP tasks.",
    )

    assert len(result) == 2
    assert all(chunk.metadata.chunk_context == "BERT is a transformer model for language understanding." for chunk in result)
    assert len(llm.calls) == 2
    assert "BERT: Pre-training" in llm.calls[0]["prompt"]
    assert "bidirectional attention" in llm.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_contextualize_chunks_continues_on_llm_failure():
    class FailingLLM:
        async def generate(self, model: str, prompt: str, **kwargs):
            raise RuntimeError("LLM unavailable")

    contextualizer = ChunkContextualizer(llm_client=FailingLLM(), model="test-model")
    chunks = [_make_chunk()]

    result = await contextualizer.contextualize_chunks(chunks=chunks, document="doc")

    assert len(result) == 1
    assert result[0].metadata.chunk_context is None


def test_get_contextualized_text_with_and_without_context():
    chunk = _make_chunk("original chunk text")
    assert chunk.get_contextualized_text() == "original chunk text"

    chunk.metadata.chunk_context = "Paper discusses neural networks."
    assert chunk.get_contextualized_text() == "Paper discusses neural networks.\n\noriginal chunk text"


def test_prepare_document_truncates_long_text():
    contextualizer = ChunkContextualizer(llm_client=MockLLMClient(), model="test-model", max_document_chars=100)
    prepared = contextualizer._prepare_document(
        document="x" * 200,
        title="Short Title",
        abstract="Short abstract.",
    )

    assert len(prepared) <= 100
    assert "[Document truncated" in prepared


def test_prepare_document_includes_title_and_abstract():
    contextualizer = ChunkContextualizer(llm_client=MockLLMClient(), model="test-model")
    prepared = contextualizer._prepare_document(
        document="Body content here.",
        title="My Paper",
        abstract="An important finding.",
    )

    assert "Title: My Paper" in prepared
    assert "Abstract: An important finding." in prepared
    assert "Body content here." in prepared
