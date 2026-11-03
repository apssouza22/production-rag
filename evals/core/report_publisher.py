"""Publish evaluation reports through pluggable backends."""

import json
from datetime import datetime
from pathlib import Path
from typing import Protocol

from evals.core.reporting import DEFAULT_REPORTS_DIR, EvaluationReport


class ReportPublisher(Protocol):
    """Write or emit a built :class:`EvaluationReport`."""

    def publish(self, report: EvaluationReport, *, filename_prefix: str | None = None) -> str:
        """Persist or deliver the report and return a stable location identifier."""
        ...


class FileReportPublisher:
    """Write unified reports as timestamped JSON files under a directory."""

    def __init__(self, *, reports_dir: Path = DEFAULT_REPORTS_DIR) -> None:
        self.reports_dir = reports_dir

    def publish(self, report: EvaluationReport, *, filename_prefix: str | None = None) -> str:
        """Write the report to disk and set ``report.report_path``."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_prefix = filename_prefix or f"evaluation_report_{report.run_type}"
        report_path = self.reports_dir / f"{resolved_prefix}_{timestamp}.json"
        report.report_path = str(report_path)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report.model_dump(mode="json"), handle, indent=2)
        return str(report_path)


class NullReportPublisher:
    """Accept reports without persisting them (``report_path`` stays unset)."""

    def publish(self, report: EvaluationReport, *, filename_prefix: str | None = None) -> str:
        return ""


def make_file_report_publisher(*, reports_dir: Path = DEFAULT_REPORTS_DIR) -> FileReportPublisher:
    """Construct the default on-disk report publisher."""
    return FileReportPublisher(reports_dir=reports_dir)


def make_null_report_publisher() -> NullReportPublisher:
    """Construct a publisher that skips report output."""
    return NullReportPublisher()


def make_report_publisher(*, enabled: bool = True, reports_dir: Path = DEFAULT_REPORTS_DIR) -> ReportPublisher:
    """Select a file or no-op publisher for CLI and runner wiring."""
    if enabled:
        return make_file_report_publisher(reports_dir=reports_dir)
    return make_null_report_publisher()
