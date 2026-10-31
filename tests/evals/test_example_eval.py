"""Tests for the minimal evaluation example."""

import pytest
from evals.example_eval.main import ExampleEvaluationRunner


@pytest.mark.asyncio
async def test_example_eval_uses_shared_report_schema() -> None:
    runner = ExampleEvaluationRunner()
    response = await runner.evaluator.evaluate(runner.dataset)
    payload = response.model_dump(mode="json")

    assert payload["total_items"] == 3
    assert payload["evaluated_items"] == 3
    assert payload["metrics"] == [
        {
            "name": "exact_match",
            "description": "Whether output and reference are equal after normalization.",
            "required_fields": ["output", "reference"],
        }
    ]
    assert payload["items"] == [
        {
            "item_id": "example-1",
            "sample": {
                "input": "What is the capital of France?",
                "output": "Paris",
                "reference": "Paris",
                "contexts": ["France is a country in Europe."],
                "source_id": "example-1",
                "metadata": {"region": "europe"},
            },
            "metrics": [
                {
                    "name": "exact_match",
                    "value": 1.0,
                    "reasoning": "Output matches the reference.",
                    "error": None,
                }
            ],
        },
        {
            "item_id": "example-2",
            "sample": {
                "input": "What is 2+2?",
                "output": "5",
                "reference": "4",
                "contexts": ["Basic arithmetic."],
                "source_id": "example-2",
                "metadata": {"difficulty": "easy"},
            },
            "metrics": [
                {
                    "name": "exact_match",
                    "value": 0.0,
                    "reasoning": "Output does not match the reference.",
                    "error": None,
                }
            ],
        },
        {
            "item_id": "example-3",
            "sample": {
                "input": "What does RAG stand for?",
                "output": "Retrieval-augmented generation",
                "reference": "Retrieval-augmented generation",
                "contexts": ["RAG combines retrieval with generation."],
                "source_id": "example-3",
                "metadata": {"topic": "rag"},
            },
            "metrics": [
                {
                    "name": "exact_match",
                    "value": 1.0,
                    "reasoning": "Output matches the reference.",
                    "error": None,
                }
            ],
        },
    ]

    report = await runner.run_eval(generate_report_file=False)

    assert report["run_type"] == "example_evaluation"
    assert report["dataset_source"] == "inline_example"
    assert report["evaluated_items"] == 3
    assert report["metrics_summary"]["exact_match"]["average_score"] == pytest.approx(2 / 3, rel=1e-4)
