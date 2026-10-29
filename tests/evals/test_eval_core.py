"""Tests for the shared evaluation contracts."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from evals.core.evaluator import Evaluator
from evals.core.metrics import MetricDefinition, SampleField
from evals.core.reporting import build_report
from evals.core.runner import EvalRunner
from evals.core.schemas import EvalSample, MetricResult
from evals.custom_evals.config import Settings
from evals.custom_evals.dataset import LangfuseTraceDataset
from evals.custom_evals.runner import TraceEvaluationRunner


def test_eval_runner_is_abstract() -> None:
    with pytest.raises(TypeError):
        EvalRunner()


def test_eval_sample_normalizes_legacy_rag_fields() -> None:
    sample = EvalSample(
        user_input="What is RAG?",
        response="An answer",
        retrieved_contexts=["A context"],
        rerank_applied=True,
    )

    assert sample.input == "What is RAG?"
    assert sample.output == "An answer"
    assert sample.contexts == ["A context"]
    assert sample.metadata["rerank_applied"] is True


@pytest.mark.asyncio
async def test_metric_requirements_are_validated_before_scoring() -> None:
    scorer_called = False

    async def scorer(sample: EvalSample) -> MetricResult:
        nonlocal scorer_called
        scorer_called = True
        return MetricResult(name="groundedness", value=1.0)

    metric = MetricDefinition(
        name="groundedness",
        description="test",
        scorer=scorer,
        required_fields=frozenset({SampleField.OUTPUT, SampleField.CONTEXTS}),
    )

    result = await metric.score(EvalSample(input="question"))

    assert result.error == "contexts, output are required"
    assert scorer_called is False


@pytest.mark.asyncio
async def test_langfuse_trace_dataset_returns_normalized_samples() -> None:
    trace = Mock(
        id="trace-1",
        input={"query": "What is RAG?"},
        output={"answer": "Retrieval-augmented generation."},
        scores=[],
    )
    langfuse = Mock()
    langfuse.api.trace.list.return_value.data = [trace]

    samples = await LangfuseTraceDataset(langfuse).load()

    assert samples == [
        EvalSample(
            input="What is RAG?",
            output="Retrieval-augmented generation.",
            source_id="trace-1",
            metadata={"source": "langfuse_trace"},
        )
    ]


@pytest.mark.asyncio
async def test_trace_runner_consumes_shared_dataset_and_metric() -> None:
    sample = EvalSample(input="question", output="answer", source_id="trace-1")
    dataset = Mock()
    dataset.load = AsyncMock(return_value=[sample])

    async def scorer(current_sample: EvalSample) -> MetricResult:
        assert current_sample is sample
        return MetricResult(name="helpfulness", value=0.8, reasoning="Useful answer")

    metric = MetricDefinition(
        name="helpfulness",
        description="test",
        scorer=scorer,
        required_fields=frozenset({SampleField.INPUT, SampleField.OUTPUT}),
    )
    settings = Settings(evaluation_sleep_time=0)

    with (
        patch("evals.custom_evals.runner.get_settings", return_value=settings),
        patch("evals.custom_evals.runner.make_eval_chat_model"),
        patch("evals.custom_evals.runner.Langfuse") as langfuse_cls,
    ):
        runner = TraceEvaluationRunner(dataset=dataset, metrics=[metric])
        report = await runner.run_eval(generate_report_file=False)

    assert report["schema_version"] == "1.0"
    assert report["run_type"] == "trace_evaluation"
    assert report["successful_items"] == 1
    assert report["metrics_summary"]["helpfulness"]["average_score"] == 0.8
    langfuse_cls.return_value.create_score.assert_called_once_with(
        trace_id="trace-1",
        name="helpfulness",
        data_type="NUMERIC",
        value=0.8,
        comment="Useful answer",
    )


@pytest.mark.asyncio
async def test_shared_evaluator_response_builds_report() -> None:
    sample = EvalSample(input="question", output="answer")
    dataset = Mock(total_items=2)
    dataset.load = AsyncMock(return_value=[sample])

    async def scorer(current_sample: EvalSample) -> MetricResult:
        return MetricResult(name="quality", value=0.75)

    metric = MetricDefinition(name="quality", description="Quality score", scorer=scorer)
    response = await Evaluator([metric]).evaluate(dataset)

    assert response.total_items == 2
    assert response.evaluated_items == 1
    assert response.items[0].sample == sample

    report = build_report(
        response=response,
        run_type="test",
        evaluation_model="judge-model",
        dataset_source="test_dataset",
    )

    assert report.total_items == 2
    assert report.metrics_summary["quality"].average_score == 0.75
