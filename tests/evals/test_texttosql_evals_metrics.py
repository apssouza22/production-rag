"""Tests for text-to-SQL execution-accuracy metrics."""

import pytest
from evals.core.schemas import EvalSample
from evals.texttosql_evals.metrics import build_metrics, score_execution_accuracy


@pytest.mark.asyncio
async def test_execution_accuracy_scores_matching_results() -> None:
    sample = EvalSample(input="question", reference="[(1000,)]", output="[(1000,)]")
    result = await score_execution_accuracy(sample)
    assert result.value == 1.0
    assert result.error is None


@pytest.mark.asyncio
async def test_execution_accuracy_scores_mismatch() -> None:
    sample = EvalSample(input="question", reference="[(1000,)]", output="[(999,)]")
    result = await score_execution_accuracy(sample)
    assert result.value == 0.0
    assert result.error is None


@pytest.mark.asyncio
async def test_execution_accuracy_scores_empty_output() -> None:
    sample = EvalSample(input="question", reference="[(1000,)]", output="")
    result = await score_execution_accuracy(sample)
    assert result.value == 0.0
    assert "no executable" in (result.reasoning or "").lower()


@pytest.mark.asyncio
async def test_execution_accuracy_allows_empty_output_via_metric_wrapper() -> None:
    metric = build_metrics()[0]
    sample = EvalSample(input="question", reference="[(1000,)]", output="")
    result = await metric.score(sample)
    assert result.value == 0.0
    assert result.error is None