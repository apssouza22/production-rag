"""Metric contracts and shared requirement validation."""

from abc import ABC, abstractmethod
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


class Metric(ABC):
    """Object-oriented contract implemented by every evaluation metric."""

    name: str
    description: str
    required_fields: frozenset[SampleField]

    @property
    def requires_reference(self) -> bool:
        return SampleField.REFERENCE in self.required_fields

    async def score(self, sample: EvalSample) -> MetricResult:
        """Validate required fields and score one normalized sample."""
        missing = [
            required_field for required_field in self.required_fields if getattr(sample, required_field.value) in (None, "", [])
        ]
        if missing:
            names = ", ".join(field.value for field in sorted(missing, key=str))
            suffix = "is" if len(missing) == 1 else "are"
            return MetricResult(name=self.name, error=f"{names} {suffix} required")
        return await self._score(sample)

    @abstractmethod
    async def _score(self, sample: EvalSample) -> MetricResult:
        """Implement provider-specific scoring for a valid sample."""


@dataclass(frozen=True, init=False)
class MetricDefinition(Metric):
    """Callable-backed metric adapter, useful for extensions and tests."""

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

    async def _score(self, sample: EvalSample) -> MetricResult:
        return await self.scorer(sample)
