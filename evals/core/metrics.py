"""Metric contracts and shared requirement validation."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from evals.core.schemas import EvalSample, MetricResult

MetricScorer = Callable[[EvalSample], Awaitable[MetricResult]]


class SampleField(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    REFERENCE = "reference"
    CONTEXTS = "contexts"


@dataclass(frozen=True, init=False)
class MetricDefinition:
    """Metadata and scorer for one metric."""

    name: str
    description: str
    scorer: MetricScorer
    required_fields: frozenset[SampleField] = field(default_factory=frozenset)

    def __init__(
        self,
        name: str,
        description: str,
        scorer: MetricScorer,
        required_fields: frozenset[SampleField] | None = None,
        requires_reference: bool | None = None,
    ) -> None:
        resolved_fields = required_fields or frozenset()
        if requires_reference:
            resolved_fields = resolved_fields | {SampleField.REFERENCE}
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "scorer", scorer)
        object.__setattr__(self, "required_fields", frozenset(resolved_fields))

    @property
    def requires_reference(self) -> bool:
        """Compatibility helper for reports and integrations."""
        return SampleField.REFERENCE in self.required_fields


def missing_required_fields(metric: MetricDefinition, sample: EvalSample) -> list[SampleField]:
    """Return fields whose values are absent from a sample."""
    missing: list[SampleField] = []
    for required_field in metric.required_fields:
        value = getattr(sample, required_field.value)
        if value is None or value == "" or value == []:
            missing.append(required_field)
    return sorted(missing, key=str)


async def evaluate_metric(metric: MetricDefinition, sample: EvalSample) -> MetricResult:
    """Validate a sample and invoke a metric scorer."""
    missing = missing_required_fields(metric, sample)
    if missing:
        names = ", ".join(field.value for field in missing)
        suffix = "is" if len(missing) == 1 else "are"
        return MetricResult(name=metric.name, error=f"{names} {suffix} required")
    return await metric.scorer(sample)
