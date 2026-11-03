"""Tests for evaluation report publishers."""

import json

from evals.core.report_publisher import FileReportPublisher, NullReportPublisher
from evals.core.reporting import EvaluationReport, MetricSummary


def test_file_report_publisher_writes_json_and_sets_path(tmp_path) -> None:
    report = EvaluationReport(
        run_type="unit_test",
        evaluation_model="mock",
        dataset_source="test",
        total_items=1,
        evaluated_items=1,
        successful_items=1,
        failed_items=0,
        duration_seconds=0.1,
        metrics_summary={"quality": MetricSummary(average_score=1.0, success_count=1)},
        items=[],
    )
    publisher = FileReportPublisher(reports_dir=tmp_path)

    path = publisher.publish(report)

    assert path == report.report_path
    assert path.startswith(str(tmp_path))
    assert "evaluation_report_unit_test_" in path
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["run_type"] == "unit_test"
    assert payload["report_path"] == path


def test_null_report_publisher_leaves_report_path_unset() -> None:
    report = EvaluationReport(
        run_type="unit_test",
        evaluation_model="mock",
        dataset_source="test",
        total_items=0,
        evaluated_items=0,
        successful_items=0,
        failed_items=0,
        duration_seconds=0.0,
        metrics_summary={},
        items=[],
    )
    publisher = NullReportPublisher()

    assert publisher.publish(report) == ""
    assert report.report_path is None
