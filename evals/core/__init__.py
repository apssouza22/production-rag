"""Shared contracts for evaluation datasets and metrics."""

from evals.core.datasets import Dataset
from evals.core.metrics import MetricDefinition, MetricScorer, SampleField, evaluate_metric
from evals.core.reporting import (
    EvaluationReport,
    MetricSummary,
    ReportItem,
    build_report,
    get_dataset_location,
    get_dataset_source_name,
    make_report_item,
    write_report,
)
from evals.core.schemas import EvalSample, MetricResult

__all__ = [
    "Dataset",
    "EvalSample",
    "EvaluationReport",
    "MetricDefinition",
    "MetricResult",
    "MetricScorer",
    "MetricSummary",
    "ReportItem",
    "SampleField",
    "build_report",
    "evaluate_metric",
    "get_dataset_location",
    "get_dataset_source_name",
    "make_report_item",
    "write_report",
]
