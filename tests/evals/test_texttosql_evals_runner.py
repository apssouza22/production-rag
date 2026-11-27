"""Tests for text-to-SQL evaluation runner reporting."""

import json
from unittest.mock import AsyncMock, Mock

import pytest
from evals.core.report_publisher import make_file_report_publisher, make_null_report_publisher
from evals.core.schemas import EvalSample
from evals.texttosql_evals.config import TextToSQLEvalSettings
from evals.texttosql_evals.runner import TextToSQLEvaluationRunner


@pytest.mark.asyncio
async def test_evaluator_aggregates_execution_accuracy(tmp_path) -> None:
    dataset_path = tmp_path / "queries.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "user_input": "How many films?",
                    "reference_sql": "SELECT COUNT(*) FROM film",
                }
            ]
        )
    )

    settings = TextToSQLEvalSettings(text_to_sql_eval_dataset_path=dataset_path)
    sample = EvalSample(
        input="How many films?",
        reference="[(1000,)]",
        output="[(1000,)]",
        metadata={"agent_sql": "SELECT COUNT(*) FROM film"},
    )

    service = Mock()
    service.agent_config = Mock(model="gpt-4o-mini", dialect="postgresql", include_tables=["film"])

    dataset = Mock(total_items=1)
    dataset.load = AsyncMock(return_value=[sample])

    runner = TextToSQLEvaluationRunner(
        settings=settings,
        service=service,
        sql_executor=Mock(),
        dataset=dataset,
        report_publisher=make_null_report_publisher(),
    )
    report = await runner.run_eval()

    assert report["schema_version"] == "1.0"
    assert report["run_type"] == "texttosql_evaluation"
    assert report["evaluated_items"] == 1
    assert report["metrics_summary"]["execution_accuracy"]["success_count"] == 1
    assert report["metrics_summary"]["execution_accuracy"]["average_score"] == 1.0


@pytest.mark.asyncio
async def test_evaluator_writes_json_report(tmp_path) -> None:
    settings = TextToSQLEvalSettings()
    sample = EvalSample(
        input="How many films?",
        reference="[(1000,)]",
        output="[(1000,)]",
    )

    service = Mock()
    service.agent_config = Mock(model="gpt-4o-mini", dialect="postgresql", include_tables=["film"])

    dataset = Mock(total_items=1)
    dataset.load = AsyncMock(return_value=[sample])
    reports_dir = tmp_path / "reports"

    runner = TextToSQLEvaluationRunner(
        settings=settings,
        service=service,
        sql_executor=Mock(),
        dataset=dataset,
        report_publisher=make_file_report_publisher(reports_dir=reports_dir),
    )
    report = await runner.run_eval()

    assert report["report_path"] is not None
    written = reports_dir / report["report_path"].split("/")[-1]
    assert written.exists()
    assert written.name.startswith("evaluation_report_texttosql_evaluation_")
    assert written.name.endswith(".json")


@pytest.mark.asyncio
async def test_run_eval_raises_when_no_samples() -> None:
    settings = TextToSQLEvalSettings()
    service = Mock()
    service.agent_config = Mock(model="gpt-4o-mini", dialect="postgresql", include_tables=["film"])

    dataset = Mock(total_items=0)
    dataset.load = AsyncMock(return_value=[])

    runner = TextToSQLEvaluationRunner(
        settings=settings,
        service=service,
        sql_executor=Mock(),
        dataset=dataset,
        report_publisher=make_null_report_publisher(),
    )
    with pytest.raises(RuntimeError, match="No evaluation samples"):
        await runner.run_eval()


@pytest.mark.asyncio
async def test_evaluator_continues_when_one_sample_has_empty_output() -> None:
    settings = TextToSQLEvalSettings()
    samples = [
        EvalSample(input="q1", reference="[(1,)]", output="[(1,)]"),
        EvalSample(input="q2", reference="[(2,)]", output=""),
    ]

    service = Mock()
    service.agent_config = Mock(model="gpt-4o-mini", dialect="postgresql", include_tables=["film"])

    dataset = Mock(total_items=2)
    dataset.load = AsyncMock(return_value=samples)

    runner = TextToSQLEvaluationRunner(
        settings=settings,
        service=service,
        sql_executor=Mock(),
        dataset=dataset,
        report_publisher=make_null_report_publisher(),
    )
    report = await runner.run_eval()

    assert report["evaluated_items"] == 2
    assert report["metrics_summary"]["execution_accuracy"]["average_score"] == 0.5