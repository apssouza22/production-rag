"""Tests for Langfuse experiment integration in rag_evals."""

import json
from unittest.mock import Mock, patch

import pytest
from evals.rag_evals.config import RagEvalSettings
from evals.rag_evals.langfuse_experiment import (
    RagLangfuseExperiment,
    build_langfuse_evaluators,
    experiment_result_to_report,
    extract_item_expected_output,
    extract_item_input,
    extract_item_preset_response,
    make_ragas_evaluator,
    queries_to_experiment_items,
)
from evals.rag_evals.metrics import MetricDefinition
from evals.rag_evals.schemas import EvalQuery, EvalSample, MetricScore
from langfuse.experiment import Evaluation, ExperimentItemResult, ExperimentResult


def test_queries_to_experiment_items_maps_reference_and_optional_response():
    queries = [
        EvalQuery(
            user_input="What is RAG?",
            reference="RAG combines retrieval with generation.",
            response="Retrieval augmented generation.",
        ),
        EvalQuery(
            user_input="How does hybrid search work?",
            reference="Hybrid search combines BM25 and vectors.",
        ),
    ]

    items = queries_to_experiment_items(queries)

    assert items == [
        {
            "input": "What is RAG?",
            "expected_output": "RAG combines retrieval with generation.",
            "metadata": {"response": "Retrieval augmented generation."},
        },
        {
            "input": "How does hybrid search work?",
            "expected_output": "Hybrid search combines BM25 and vectors.",
            "metadata": None,
        },
    ]


def test_extract_item_fields_support_local_and_structured_inputs():
    local_item = {
        "input": {"user_input": "What is RAG?", "response": "A RAG answer"},
        "expected_output": "Reference answer",
        "metadata": {"response": "Metadata response"},
    }

    assert extract_item_input(local_item) == "What is RAG?"
    assert extract_item_expected_output(local_item) == "Reference answer"
    assert extract_item_preset_response(local_item) == "Metadata response"


@pytest.mark.asyncio
async def test_make_ragas_evaluator_returns_langfuse_evaluation():
    async def fake_scorer(sample: EvalSample) -> MetricScore:
        assert sample.user_input == "What is RAG?"
        assert sample.reference == "Reference"
        return MetricScore(name="context_relevance", value=0.9)

    metric = MetricDefinition(
        name="context_relevance",
        description="test",
        requires_reference=False,
        scorer=fake_scorer,
    )
    evaluator = make_ragas_evaluator(metric)
    result = await evaluator(
        input="What is RAG?",
        output={
            "response": "RAG combines retrieval and generation.",
            "retrieved_contexts": ["chunk"],
            "rerank_applied": True,
        },
        expected_output="Reference",
        metadata=None,
    )

    assert result.name == "context_relevance"
    assert result.value == 0.9


@pytest.mark.asyncio
async def test_make_ragas_evaluator_handles_missing_reference():
    async def fake_scorer(sample: EvalSample) -> MetricScore:
        return MetricScore(name="context_precision", value=0.8)

    metric = MetricDefinition(
        name="context_precision",
        description="test",
        requires_reference=True,
        scorer=fake_scorer,
    )
    evaluator = make_ragas_evaluator(metric)
    result = await evaluator(
        input="What is RAG?",
        output={"response": "answer", "retrieved_contexts": ["chunk"], "rerank_applied": False},
        expected_output=None,
        metadata=None,
    )

    assert result.value == 0
    assert result.comment == "Reference answer is required"


def test_experiment_result_to_report_aggregates_scores():
    settings = RagEvalSettings()
    result = ExperimentResult(
        name="rag-retrieval-eval",
        run_name="rag-retrieval-eval-20260101",
        description="test run",
        item_results=[
            ExperimentItemResult(
                item={"input": "What is RAG?", "expected_output": "Reference"},
                output={
                    "response": "RAG answer",
                    "retrieved_contexts": ["chunk"],
                    "rerank_applied": True,
                },
                evaluations=[Evaluation(name="context_relevance", value=0.9)],
                trace_id="trace-1",
                dataset_run_id=None,
            )
        ],
        run_evaluations=[],
        experiment_id="exp-1",
        dataset_run_id="run-1",
        dataset_run_url="https://cloud.langfuse.com/project/run-1",
    )

    report = experiment_result_to_report(result, settings, dataset_path="evals/rag_evals/queries.json")

    assert report["evaluated_samples"] == 1
    assert report["metrics_summary"]["context_relevance"]["avg_score"] == 0.9
    assert report["sample_results"][0]["trace_id"] == "trace-1"
    assert report["dataset_run_url"] == "https://cloud.langfuse.com/project/run-1"


def test_rag_langfuse_experiment_runs_local_dataset(tmp_path):
    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "user_input": "What is RAG?",
                    "reference": "RAG combines retrieval with generation.",
                    "response": "Retrieval augmented generation.",
                }
            ]
        )
    )
    settings = RagEvalSettings(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        rag_eval_dataset_path=dataset_path,
        rag_eval_generate_responses=False,
    )
    experiment_result = ExperimentResult(
        name="rag-retrieval-eval",
        run_name="rag-retrieval-eval-local",
        description="test",
        item_results=[],
        run_evaluations=[],
        experiment_id="exp-1",
    )

    with (
        patch("evals.rag_evals.langfuse_experiment.Langfuse") as langfuse_cls,
        patch("evals.rag_evals.langfuse_experiment.create_openai_client", return_value=Mock()),
        patch("evals.rag_evals.langfuse_experiment.RerankSearchClient"),
        patch("evals.rag_evals.langfuse_experiment.DatasetBuilder"),
        patch("evals.rag_evals.langfuse_experiment.build_metrics", return_value=[]),
    ):
        langfuse_client = langfuse_cls.return_value
        langfuse_client.run_experiment.return_value = experiment_result
        runner = RagLangfuseExperiment(settings=settings)
        result = runner.run(generate_report_file=False)

    langfuse_client.run_experiment.assert_called_once()
    call_kwargs = langfuse_client.run_experiment.call_args.kwargs
    assert call_kwargs["name"] == "rag-retrieval-eval"
    assert call_kwargs["data"][0]["input"] == "What is RAG?"
    assert call_kwargs["metadata"]["top_k"] == "5"
    assert result.experiment_id == "exp-1"


def test_rag_langfuse_experiment_uses_hosted_dataset(tmp_path):
    settings = RagEvalSettings(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        rag_eval_langfuse_dataset="hosted-rag-eval",
    )
    experiment_result = ExperimentResult(
        name="rag-retrieval-eval",
        run_name="rag-retrieval-eval-hosted",
        description="test",
        item_results=[],
        run_evaluations=[],
        experiment_id="exp-2",
        dataset_run_url="https://cloud.langfuse.com/project/run-2",
    )
    dataset_client = Mock()
    dataset_client.run_experiment.return_value = experiment_result

    with (
        patch("evals.rag_evals.langfuse_experiment.Langfuse") as langfuse_cls,
        patch("evals.rag_evals.langfuse_experiment.create_openai_client", return_value=Mock()),
        patch("evals.rag_evals.langfuse_experiment.RerankSearchClient"),
        patch("evals.rag_evals.langfuse_experiment.DatasetBuilder"),
        patch("evals.rag_evals.langfuse_experiment.build_metrics", return_value=[]),
    ):
        langfuse_client = langfuse_cls.return_value
        langfuse_client.get_dataset.return_value = dataset_client
        runner = RagLangfuseExperiment(settings=settings)
        result = runner.run(generate_report_file=False)

    langfuse_client.get_dataset.assert_called_once_with("hosted-rag-eval")
    dataset_client.run_experiment.assert_called_once()
    assert result.dataset_run_url == "https://cloud.langfuse.com/project/run-2"


def test_build_langfuse_evaluators_creates_one_evaluator_per_metric():
    async def fake_scorer(sample: EvalSample) -> MetricScore:
        return MetricScore(name="metric", value=1.0)

    metrics = [
        MetricDefinition(name="context_relevance", description="a", requires_reference=False, scorer=fake_scorer),
        MetricDefinition(name="faithfulness", description="b", requires_reference=False, scorer=fake_scorer),
    ]

    evaluators = build_langfuse_evaluators(metrics)

    assert len(evaluators) == 2


def test_rag_langfuse_experiment_requires_credentials():
    settings = RagEvalSettings(langfuse_public_key="", langfuse_secret_key="")

    with pytest.raises(ValueError, match="Langfuse credentials are required"):
        RagLangfuseExperiment(settings=settings)
