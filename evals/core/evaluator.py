"""Unified metric evaluator, independent from reporting and providers."""

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, Field
from tqdm import tqdm

from evals.core.datasets import Dataset
from evals.core.metrics import Metric
from evals.core.schemas import EvalSample, MetricResult

logger = logging.getLogger(__name__)


class MetricMetadata(BaseModel):
    """Serializable metric information included in an evaluation response."""

    name: str
    description: str
    required_fields: list[str] = Field(default_factory=list)


class EvaluationItemResult(BaseModel):
    """Metric results for one normalized sample."""

    item_id: str
    sample: EvalSample
    metrics: list[MetricResult]


class EvaluationResponse(BaseModel):
    """Provider-neutral output produced by the shared evaluator."""

    total_items: int
    evaluated_items: int
    duration_seconds: float
    metrics: list[MetricMetadata]
    items: list[EvaluationItemResult]


class MetricResultHandler(Protocol):
    """React to a metric result without coupling it to the evaluator."""

    async def handle(self, sample: EvalSample, result: MetricResult) -> None:
        """Process one metric result."""
        ...


class Evaluator:
    """Apply any collection of metrics to samples from any dataset."""

    def __init__(
        self,
        metrics: Sequence[Metric],
        *,
        result_handlers: Sequence[MetricResultHandler] = (),
        item_delay_seconds: float = 0,
        progress_description: str = "Evaluating samples",
    ) -> None:
        self.metrics = list(metrics)
        self.result_handlers = list(result_handlers)
        self.item_delay_seconds = item_delay_seconds
        self.progress_description = progress_description

    async def evaluate(self, dataset: Dataset) -> EvaluationResponse:
        """Load a dataset and return metric results without building a report."""
        start_time = time.time()
        samples = await dataset.load()
        items: list[EvaluationItemResult] = []

        for index, sample in enumerate(tqdm(samples, desc=self.progress_description)):
            metric_results: list[MetricResult] = []
            for metric in self.metrics:
                try:
                    logger.info("Evaluating sample %s with metric %s", sample.source_id, metric.name)
                    result = await metric.score(sample)
                except Exception as exc:
                    logger.exception("Metric %s evaluation failed", metric.name)
                    result = MetricResult(name=metric.name, error=str(exc))

                metric_results.append(result)
                for handler in self.result_handlers:
                    await handler.handle(sample, result)

            items.append(
                EvaluationItemResult(
                    item_id=sample.source_id or f"sample-{index}",
                    sample=sample,
                    metrics=metric_results,
                )
            )
            if self.item_delay_seconds:
                await asyncio.sleep(self.item_delay_seconds)

        total_items = getattr(dataset, "total_items", len(samples))
        if not isinstance(total_items, int):
            total_items = len(samples)
        return EvaluationResponse(
            total_items=total_items,
            evaluated_items=len(items),
            duration_seconds=round(time.time() - start_time, 2),
            metrics=[
                MetricMetadata(
                    name=metric.name,
                    description=metric.description,
                    required_fields=sorted(field.value for field in metric.required_fields),
                )
                for metric in self.metrics
            ],
            items=items,
        )
