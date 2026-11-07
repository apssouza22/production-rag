"""Tests for experiment runner retrieval wiring."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from evals.core.evaluator import EvaluationItemResult, EvaluationResponse
from evals.core.report_publisher import make_null_report_publisher
from evals.core.schemas import EvalSample, MetricResult
from evals.experiments.config import ExperimentRunConfig, ExperimentSettings, QuerySettings
from evals.experiments.runner import ExperimentRunner
from evals.experiments.schemas import ExperimentQuery
from src.config import ChunkingSettings, PDFParserSettings


def test_runner_rejects_kg_mode_when_kg_disabled() -> None:
    run_config = ExperimentRunConfig(
        pdf_parser=PDFParserSettings(),
        chunking=ChunkingSettings(),
        query=QuerySettings(),
        experiment=ExperimentSettings(retrieval_mode="kg"),
        config_path=Path("evals/experiments/data/settings/experiment-config.json"),
    )
    app_settings = Mock()
    app_settings.kg.enabled = False

    with (
        patch("evals.experiments.runner.get_app_settings", return_value=app_settings),
        pytest.raises(RuntimeError, match="retrieval_mode=kg"),
    ):
        ExperimentRunner(run_config=run_config, metrics=[])


@pytest.mark.asyncio
async def test_experiment_report_metadata_includes_retrieval_mode() -> None:
    run_config = ExperimentRunConfig(
        pdf_parser=PDFParserSettings(),
        chunking=ChunkingSettings(),
        query=QuerySettings(),
        experiment=ExperimentSettings(retrieval_mode="rrf"),
        config_path=Path("evals/experiments/data/settings/experiment-config.json"),
    )
    sample = EvalSample(
        input="What is X?",
        output="X is described in the paper.",
        reference="X is a method.",
        contexts=["chunk one"],
        metadata={"retrieval_mode": "rrf"},
    )
    eval_response = EvaluationResponse(
        total_items=1,
        evaluated_items=1,
        duration_seconds=0.5,
        metrics=[],
        items=[
            EvaluationItemResult(
                item_id="item-1",
                sample=sample,
                metrics=[MetricResult(name="context_relevance", value=0.9)],
            )
        ],
    )
    question_generator = Mock()
    question_generator.build_queries = AsyncMock(
        return_value=[ExperimentQuery(user_input="q", reference="r")]
    )

    with (
        patch("evals.experiments.runner.make_hybrid_rrf_search", return_value=Mock()),
        patch("evals.experiments.runner.make_rerank_search_service", return_value=Mock()),
        patch("evals.experiments.runner.make_embeddings_service"),
        patch("evals.experiments.runner.make_reranker_client"),
        patch("evals.experiments.runner.make_eval_chat_model", return_value=Mock()),
        patch("evals.experiments.runner.QuestionGenerator", return_value=question_generator),
        patch(
            "evals.experiments.runner.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value={"skipped": True, "chunk_count": 3, "ingestion_name": "evals-experiments"},
        ),
    ):
        runner = ExperimentRunner(
            run_config=run_config,
            metrics=[],
            report_publisher=make_null_report_publisher(),
        )
        runner.evaluator.evaluate = AsyncMock(return_value=eval_response)
        report = await runner.run_eval()

    assert report["metadata"]["retrieval"]["retrieval_mode"] == "rrf"
    assert report["items"][0]["metadata"]["retrieval_mode"] == "rrf"
