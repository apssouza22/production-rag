"""Unified report models and aggregation for every evaluation runner."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from evals.core.evaluator import EvaluationResponse
from evals.core.schemas import EvalSample, MetricResult

REPORT_SCHEMA_VERSION = "1.0"
DEFAULT_REPORTS_DIR = Path(__file__).parent.parent / "reports"


class MetricSummary(BaseModel):
    """Aggregate results for one metric."""

    description: str = ""
    required_fields: list[str] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    average_score: float = 0.0


class ReportItem(BaseModel):
    """One evaluated item in a report."""

    id: str
    input_preview: str = ""
    output_preview: str = ""
    reference_preview: str = ""
    context_count: int = 0
    metrics: list[MetricResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Stable report shape emitted by every evaluation path."""

    schema_version: str = REPORT_SCHEMA_VERSION
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    run_type: str
    evaluation_model: str
    dataset_source: str
    dataset_location: str | None = None
    total_items: int
    evaluated_items: int
    successful_items: int
    failed_items: int
    duration_seconds: float
    metrics_summary: dict[str, MetricSummary]
    items: list[ReportItem]
    metadata: dict[str, Any] = Field(default_factory=dict)
    report_path: str | None = None


def get_dataset_source_name(dataset: Any) -> str:
    """Resolve a stable source name from any dataset implementation."""
    source_name = getattr(dataset, "source_name", None)
    return source_name if isinstance(source_name, str) else type(dataset).__name__


def get_dataset_location(dataset: Any) -> str | None:
    """Resolve an optional source location from a dataset implementation."""
    for attribute in ("source_location", "dataset_path"):
        location = getattr(dataset, attribute, None)
        if isinstance(location, (str, Path)):
            return str(location)
    return None


def make_report_item(
    sample: EvalSample,
    metrics: list[MetricResult],
    *,
    item_id: str,
    metadata: dict[str, Any] | None = None,
) -> ReportItem:
    """Create a consistently redacted item representation."""
    item_metadata = dict(sample.metadata)
    if metadata:
        item_metadata.update(metadata)
    return ReportItem(
        id=item_id,
        input_preview=(sample.input or "")[:200],
        output_preview=(sample.output or "")[:200],
        reference_preview=(sample.reference or "")[:200],
        context_count=len(sample.contexts),
        metrics=metrics,
        metadata=item_metadata,
    )


def build_report(
    *,
    response: EvaluationResponse,
    run_type: str,
    evaluation_model: str,
    dataset_source: str,
    dataset_location: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    """Build a report from the provider-neutral evaluator response."""
    items = [make_report_item(item.sample, item.metrics, item_id=item.item_id) for item in response.items]
    definitions = {metric.name: metric for metric in response.metrics}
    metric_names = set(definitions)
    metric_names.update(result.name for item in items for result in item.metrics)

    summaries: dict[str, MetricSummary] = {}
    for name in sorted(metric_names):
        definition = definitions.get(name)
        values: list[float] = []
        failure_count = 0
        for item in items:
            result = next((result for result in item.metrics if result.name == name), None)
            if result is None or result.value is None or result.error:
                failure_count += 1
            else:
                values.append(result.value)
        summaries[name] = MetricSummary(
            description=definition.description if definition else "",
            required_fields=definition.required_fields if definition else [],
            success_count=len(values),
            failure_count=failure_count,
            average_score=round(sum(values) / len(values), 4) if values else 0.0,
        )

    successful_items = 0
    for item in items:
        results_by_name = {result.name: result for result in item.metrics}
        if metric_names and all(
            name in results_by_name and results_by_name[name].value is not None and not results_by_name[name].error
            for name in metric_names
        ):
            successful_items += 1
    return EvaluationReport(
        run_type=run_type,
        evaluation_model=evaluation_model,
        dataset_source=dataset_source,
        dataset_location=dataset_location,
        total_items=response.total_items,
        evaluated_items=response.evaluated_items,
        successful_items=successful_items,
        failed_items=len(items) - successful_items,
        duration_seconds=response.duration_seconds,
        metrics_summary=summaries,
        items=items,
        metadata=metadata or {},
    )


def write_report(
    report: EvaluationReport,
    *,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    filename_prefix: str | None = None,
) -> str:
    """Write a unified report and include its path in the serialized payload."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_prefix = filename_prefix or f"evaluation_report_{report.run_type}"
    report_path = reports_dir / f"{resolved_prefix}_{timestamp}.json"
    report.report_path = str(report_path)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report.model_dump(mode="json"), handle, indent=2)
    return str(report_path)
