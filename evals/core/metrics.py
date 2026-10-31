"""Metric contracts and shared requirement validation."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from evals.core.schemas import EvalSample, MetricResult, ScoreSchema

MetricScorer = Callable[[EvalSample], Awaitable[MetricResult]]
logger = logging.getLogger(__name__)


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


class PromptMetric(Metric):
    """LLM judge driven by a system prompt."""

    required_fields = frozenset({SampleField.INPUT, SampleField.OUTPUT})

    def __init__(
        self,
        *,
        name: str,
        prompt: str,
        chat_model: BaseChatModel,
        max_retries: int = 3,
        retry_delay_seconds: float = 10,
    ) -> None:
        self.name = name
        self.description = f"LLM judge using the {name} prompt."
        self.prompt = prompt
        self.chat_model = chat_model
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    async def _score(self, sample: EvalSample) -> MetricResult:
        for attempt in range(self.max_retries):
            try:
                judge = self.chat_model.with_structured_output(ScoreSchema)
                response = await judge.ainvoke(
                    [
                        SystemMessage(content=self.prompt),
                        HumanMessage(content=f"Input: {sample.input}\nGeneration: {sample.output}"),
                    ]
                )
                parsed = response if isinstance(response, ScoreSchema) else ScoreSchema.model_validate(response)
                return MetricResult(name=self.name, value=parsed.score, reasoning=parsed.reasoning)
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    logger.exception("Metric %s failed after retries", self.name)
                    return MetricResult(name=self.name, error=str(exc))
                logger.error(
                    "Metric %s failed (retrying in %ss): %s",
                    self.name,
                    self.retry_delay_seconds,
                    exc,
                )
                await asyncio.sleep(self.retry_delay_seconds)
        return MetricResult(name=self.name, error="scoring failed")


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
