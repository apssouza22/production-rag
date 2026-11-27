"""Deterministic metrics for text-to-SQL evaluation."""

from evals.core.metrics import MetricDefinition, SampleField
from evals.core.schemas import EvalSample, MetricResult


async def score_execution_accuracy(sample: EvalSample) -> MetricResult:
    """Score 1.0 when agent and reference normalized results match."""
    if not (sample.output or "").strip():
        return MetricResult(
            name="execution_accuracy",
            value=0.0,
            reasoning="Agent produced no executable SQL result.",
        )

    if sample.output == sample.reference:
        return MetricResult(
            name="execution_accuracy",
            value=1.0,
            reasoning="Agent SQL result matches reference SQL result.",
        )

    return MetricResult(
        name="execution_accuracy",
        value=0.0,
        reasoning="Agent SQL result differs from reference SQL result.",
    )


def build_metrics() -> list[MetricDefinition]:
    """Return the default metric set for text-to-SQL eval."""
    return [
        MetricDefinition(
            name="execution_accuracy",
            description="Whether the agent's last SQL returns the same normalized result as reference SQL.",
            scorer=score_execution_accuracy,
            required_fields=frozenset({SampleField.REFERENCE}),
        )
    ]