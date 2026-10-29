"""Shared contracts for evaluation datasets and metrics."""

from evals.core.datasets import Dataset
from evals.core.evaluator import (
    EvaluationItemResult,
    EvaluationResponse,
    Evaluator,
    MetricMetadata,
    MetricResultHandler,
)
from evals.core.metrics import Metric, MetricDefinition, MetricScorer, PromptMetric, SampleField
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
from evals.core.runner import EvalRunner
from evals.core.schemas import EvalSample, MetricResult

__all__ = [
    "Dataset",
    "EvalSample",
    "EvalRunner",
    "EvaluationItemResult",
    "EvaluationReport",
    "EvaluationResponse",
    "Evaluator",
    "Metric",
    "MetricDefinition",
    "MetricResult",
    "MetricScorer",
    "MetricSummary",
    "MetricMetadata",
    "MetricResultHandler",
    "PromptMetric",
    "ReportItem",
    "SampleField",
    "build_report",
    "get_dataset_location",
    "get_dataset_source_name",
    "make_report_item",
    "write_report",
]
