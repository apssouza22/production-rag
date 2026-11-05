"""Tests for experiment dataset retrieval modes."""

from unittest.mock import AsyncMock, Mock

import pytest
from evals.experiments.config import ExperimentSettings
from evals.experiments.dataset import ExperimentDataset
from evals.experiments.schemas import ExperimentQuery
from langchain_core.messages import AIMessage
from src.domain.kg.schemas import DualLevelKGResult, KgChunkHit
from src.domain.search.schemas import DocumentSearchResult, SearchDocument


@pytest.fixture
def experiment_settings() -> ExperimentSettings:
    return ExperimentSettings(retrieval_mode="rrf", experiment_top_k=3)


@pytest.fixture
def sample_query() -> ExperimentQuery:
    return ExperimentQuery(
        user_input="What is attention?",
        reference="Attention maps query-key pairs.",
        arxiv_id="1234.5678",
    )


@pytest.mark.asyncio
async def test_build_sample_rrf_uses_search_rrf(experiment_settings: ExperimentSettings, sample_query: ExperimentQuery):
    search_service = Mock()
    search_service.search_rrf = AsyncMock(
        return_value=DocumentSearchResult(
            query=sample_query.user_input,
            search_mode="hybrid",
            rerank_applied=True,
            before_rerank=[],
            after_rerank=[
                SearchDocument(arxiv_id="1234.5678", chunk_text="attention mechanism", rank=0),
                SearchDocument(arxiv_id="1234.5678", chunk_text="   ", rank=1),
            ],
        )
    )
    search_service.dual_search = AsyncMock()
    chat_model = Mock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="Answer from RRF."))

    dataset = ExperimentDataset(
        queries=[sample_query],
        rerank_search_service=search_service,
        chat_model=chat_model,
        settings=experiment_settings,
    )
    sample = await dataset._build_sample(sample_query)

    assert sample is not None
    assert sample.contexts == ["attention mechanism"]
    assert sample.metadata["retrieval_mode"] == "rrf"
    assert sample.metadata["search_mode"] == "hybrid"
    search_service.search_rrf.assert_awaited_once_with(sample_query.user_input, top_k=3)
    search_service.dual_search.assert_not_called()


@pytest.mark.asyncio
async def test_build_sample_kg_uses_dual_search_and_top_k(sample_query: ExperimentQuery):
    settings = ExperimentSettings(retrieval_mode="kg", experiment_top_k=2)
    search_service = Mock()
    search_service.dual_search = AsyncMock(
        return_value=DualLevelKGResult(
            query=sample_query.user_input,
            high_level_keywords=["transformers"],
            low_level_keywords=["attention"],
            entities=[],
            relations=[],
            chunks=[
                KgChunkHit(
                    chunk_key="1234.5678:0",
                    arxiv_id="1234.5678",
                    chunk_index=0,
                    chunk_text="first chunk",
                ),
                KgChunkHit(
                    chunk_key="1234.5678:1",
                    arxiv_id="1234.5678",
                    chunk_index=1,
                    chunk_text="second chunk",
                ),
                KgChunkHit(
                    chunk_key="1234.5678:2",
                    arxiv_id="1234.5678",
                    chunk_index=2,
                    chunk_text="third chunk",
                ),
            ],
        )
    )
    search_service.search_rrf = AsyncMock()
    chat_model = Mock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="Answer from KG."))

    dataset = ExperimentDataset(
        queries=[sample_query],
        rerank_search_service=search_service,
        chat_model=chat_model,
        settings=settings,
    )
    sample = await dataset._build_sample(sample_query)

    assert sample is not None
    assert sample.contexts == ["first chunk", "second chunk"]
    assert sample.metadata["retrieval_mode"] == "kg"
    assert sample.metadata["low_level_keywords"] == ["attention"]
    search_service.dual_search.assert_awaited_once_with(sample_query.user_input)
    search_service.search_rrf.assert_not_called()


@pytest.mark.asyncio
async def test_build_sample_kg_skips_when_no_chunks(sample_query: ExperimentQuery):
    settings = ExperimentSettings(retrieval_mode="kg")
    search_service = Mock()
    search_service.dual_search = AsyncMock(
        return_value=DualLevelKGResult(query=sample_query.user_input, chunks=[])
    )
    chat_model = Mock()

    dataset = ExperimentDataset(
        queries=[sample_query],
        rerank_search_service=search_service,
        chat_model=chat_model,
        settings=settings,
    )

    assert await dataset._build_sample(sample_query) is None
    chat_model.ainvoke.assert_not_called()
