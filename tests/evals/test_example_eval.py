"""Tests for the minimal evaluation example."""

import pytest
from evals.example_eval.main import ExampleEvaluationRunner


@pytest.mark.asyncio
async def test_example_eval_uses_shared_report_schema() -> None:
    report = await ExampleEvaluationRunner().run_eval(generate_report_file=False)

    assert report["run_type"] == "example_evaluation"
    assert report["dataset_source"] == "inline_example"
    assert report["evaluated_items"] == 1
    assert report["metrics_summary"]["exact_match"]["average_score"] == 1.0
