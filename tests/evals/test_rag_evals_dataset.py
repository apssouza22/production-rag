"""Tests for Ragas dataset building."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from evals.rag_evals.config import RagEvalSettings
from evals.rag_evals.dataset import DatasetBuilder, load_queries
from evals.rag_evals.retrieval import RerankSearchClient
from evals.rag_evals.schemas import EvalQuery, RerankSearchDocument, RerankSearchResponse


@pytest.fixture
def sample_dataset_path(tmp_path: Path) -> Path:
    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "user_input": "What is RAG?",
                    "reference": "RAG combines retrieval with generation.",
                    "response": "Retrieval augmented generation.",
                },
                {
                    "user_input": "How does hybrid search work?",
                    "reference": "Hybrid search combines BM25 and vector retrieval.",
                },
            ]
        ),
        encoding="utf-8",
    )
    return dataset_path


def test_load_queries(sample_dataset_path: Path):
    queries = load_queries(sample_dataset_path)

    assert len(queries) == 2
    assert queries[0] == EvalQuery(
        user_input="What is RAG?",
        reference="RAG combines retrieval with generation.",
        response="Retrieval augmented generation.",
    )
    assert queries[1].response is None


@pytest.mark.asyncio
async def test_build_samples_uses_rerank_search_contexts(sample_dataset_path: Path):
    settings = RagEvalSettings(
        rag_eval_dataset_path=sample_dataset_path,
        rag_eval_generate_responses=True,
    )
    retrieval_client = Mock()
    retrieval_client.search = AsyncMock(
        side_effect=[
            RerankSearchResponse(
                query="What is RAG?",
                rerank_applied=True,
                after_rerank=[RerankSearchDocument(chunk_text="chunk about RAG", arxiv_id="1234.5678", rank=0)],
            ),
            RerankSearchResponse(
                query="How does hybrid search work?",
                rerank_applied=False,
                after_rerank=[RerankSearchDocument(chunk_text="chunk about hybrid search", arxiv_id="8765.4321", rank=0)],
            ),
        ]
    )
    openai_client = Mock()
    openai_client.chat = Mock()
    openai_client.chat.completions = Mock()
    openai_client.chat.completions.create = AsyncMock(
        return_value=Mock(choices=[Mock(message=Mock(content="Hybrid search combines BM25 and vectors."))])
    )

    builder = DatasetBuilder(settings=settings, retrieval_client=retrieval_client, openai_client=openai_client)
    samples = await builder.build_samples(
        [
            EvalQuery(
                user_input="What is RAG?",
                reference="RAG combines retrieval with generation.",
                response="Retrieval augmented generation.",
            ),
            EvalQuery(
                user_input="How does hybrid search work?",
                reference="Hybrid search combines BM25 and vector retrieval.",
            ),
        ]
    )

    assert len(samples) == 2
    assert samples[0].retrieved_contexts == ["chunk about RAG"]
    assert samples[0].reference == "RAG combines retrieval with generation."
    assert samples[0].response == "Retrieval augmented generation."
    assert samples[1].retrieved_contexts == ["chunk about hybrid search"]
    assert samples[1].response == "Hybrid search combines BM25 and vectors."
    assert retrieval_client.search.await_count == 2


def test_contexts_from_result_skips_empty_chunks():
    result = RerankSearchResponse(
        query="test",
        rerank_applied=False,
        after_rerank=[
            RerankSearchDocument(chunk_text="useful chunk", rank=0),
            RerankSearchDocument(chunk_text="   ", rank=1),
        ],
    )

    contexts = RerankSearchClient.contexts_from_result(result)

    assert contexts == ["useful chunk"]
