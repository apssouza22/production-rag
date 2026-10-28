"""Tests for Ragas evaluator reporting."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from evals.rag_evals.config import RagEvalSettings
from evals.rag_evals.evaluator import RagEvaluator
from evals.rag_evals.metrics import MetricDefinition
from evals.rag_evals.schemas import EvalSample, MetricScore


@pytest.mark.asyncio
async def test_evaluator_aggregates_metric_scores(tmp_path):
    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "user_input": "What is RAG?",
                    "reference": "RAG combines retrieval with generation.",
                    "response": "RAG combines retrieval and generation.",
                }
            ]
        )
    )

    settings = RagEvalSettings(rag_eval_dataset_path=dataset_path, rag_eval_generate_responses=False)
    sample = EvalSample(
        user_input="What is RAG?",
        reference="RAG combines retrieval with generation.",
        response="RAG combines retrieval and generation.",
        retrieved_contexts=["A chunk about retrieval augmented generation."],
        rerank_applied=True,
    )

    async def fake_scorer(current_sample: EvalSample) -> MetricScore:
        return MetricScore(name="context_relevance", value=0.9)

    metric_definition = MetricDefinition(
        name="context_relevance",
        description="test",
        requires_reference=False,
        scorer=fake_scorer,
    )

    with (
        patch("evals.rag_evals.evaluator.create_openai_client", return_value=Mock()),
        patch("evals.rag_evals.evaluator.RerankSearchClient"),
        patch("evals.rag_evals.evaluator.DatasetBuilder") as dataset_builder_cls,
        patch("evals.rag_evals.evaluator.build_metrics", return_value=[metric_definition]),
        patch("evals.rag_evals.evaluator.load_queries", return_value=[Mock()]),
    ):
        dataset_builder_cls.return_value.build_samples = AsyncMock(return_value=[sample])
        evaluator = RagEvaluator(settings=settings)
        report = await evaluator.run(generate_report_file=False)

    assert report["evaluated_samples"] == 1
    assert report["metrics_summary"]["context_relevance"]["success_count"] == 1
    assert report["metrics_summary"]["context_relevance"]["avg_score"] == 0.9
    assert report["sample_results"][0]["context_count"] == 1
    assert report["sample_results"][0]["metrics"] == [{"name": "context_relevance", "value": 0.9, "error": None}]


@pytest.mark.asyncio
async def test_evaluator_writes_json_report(tmp_path):
    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "user_input": "What is RAG?",
                    "reference": "RAG combines retrieval with generation.",
                    "response": "RAG combines retrieval and generation.",
                }
            ]
        )
    )
    reports_dir = tmp_path / "reports"

    settings = RagEvalSettings(rag_eval_dataset_path=dataset_path, rag_eval_generate_responses=False)
    sample = EvalSample(
        user_input="What is RAG?",
        reference="RAG combines retrieval with generation.",
        response="RAG combines retrieval and generation.",
        retrieved_contexts=["A chunk about retrieval augmented generation."],
        rerank_applied=True,
    )

    async def fake_scorer(current_sample: EvalSample) -> MetricScore:
        return MetricScore(name="context_relevance", value=0.9)

    metric_definition = MetricDefinition(
        name="context_relevance",
        description="test",
        requires_reference=False,
        scorer=fake_scorer,
    )

    with (
        patch("evals.rag_evals.evaluator.create_openai_client", return_value=Mock()),
        patch("evals.rag_evals.evaluator.RerankSearchClient"),
        patch("evals.rag_evals.evaluator.DatasetBuilder") as dataset_builder_cls,
        patch("evals.rag_evals.evaluator.build_metrics", return_value=[metric_definition]),
        patch("evals.rag_evals.evaluator.load_queries", return_value=[Mock()]),
        patch("evals.rag_evals.evaluator.REPORTS_DIR", reports_dir),
    ):
        dataset_builder_cls.return_value.build_samples = AsyncMock(return_value=[sample])
        evaluator = RagEvaluator(settings=settings)
        report = await evaluator.run(generate_report_file=True)

    report_path = report["report_path"]
    with open(report_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["evaluated_samples"] == 1
    assert payload["sample_results"][0]["metrics"][0]["name"] == "context_relevance"
    assert payload["metrics_summary"]["context_relevance"]["description"] == "test"
    assert payload["metrics"][0]["description"] == "test"
    assert payload["metrics"][0]["requires_reference"] is False


@pytest.mark.asyncio
async def test_evaluator_skips_reference_metrics_without_reference(tmp_path):
    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "user_input": "What is RAG?",
                    "reference": "RAG combines retrieval with generation.",
                    "response": "RAG combines retrieval and generation.",
                }
            ]
        )
    )

    settings = RagEvalSettings(rag_eval_dataset_path=dataset_path, rag_eval_generate_responses=False)
    sample = EvalSample(
        user_input="What is RAG?",
        reference="",
        response="RAG combines retrieval and generation.",
        retrieved_contexts=["A chunk about retrieval augmented generation."],
        rerank_applied=True,
    )

    async def fake_scorer(current_sample: EvalSample) -> MetricScore:
        return MetricScore(name="context_precision", value=0.8)

    metric_definition = MetricDefinition(
        name="context_precision",
        description="test",
        requires_reference=True,
        scorer=fake_scorer,
    )

    with (
        patch("evals.rag_evals.evaluator.create_openai_client", return_value=Mock()),
        patch("evals.rag_evals.evaluator.RerankSearchClient"),
        patch("evals.rag_evals.evaluator.DatasetBuilder") as dataset_builder_cls,
        patch("evals.rag_evals.evaluator.build_metrics", return_value=[metric_definition]),
        patch("evals.rag_evals.evaluator.load_queries", return_value=[Mock()]),
    ):
        dataset_builder_cls.return_value.build_samples = AsyncMock(return_value=[sample])
        evaluator = RagEvaluator(settings=settings)
        report = await evaluator.run(generate_report_file=False)

    assert report["metrics_summary"]["context_precision"]["failure_count"] == 1
    assert report["sample_results"][0]["metrics"][0]["error"] == "reference is required"
