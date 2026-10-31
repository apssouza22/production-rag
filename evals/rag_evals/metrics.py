"""Ragas metrics for retrieval and response evaluation."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from ragas.metrics.collections import (
    ContextPrecisionWithReference,
    ContextRecall,
    ContextRelevance,
    ContextUtilization,
    FactualCorrectness,
    Faithfulness,
)

from evals.core.llm import RagasChatModel
from evals.core.metrics import Metric, SampleField
from evals.core.schemas import EvalSample, MetricResult


class ContextRelevanceMetric(Metric):
    name = "context_relevance"
    description = "Checks whether retrieved chunks are relevant to the query (0–1, higher is better)."
    required_fields = frozenset({SampleField.INPUT, SampleField.CONTEXTS})

    def __init__(self, llm: Any) -> None:
        self.evaluator = ContextRelevance(llm=llm)

    async def _score(self, sample: EvalSample) -> MetricResult:
        result = await self.evaluator.ascore(
            user_input=sample.input,
            retrieved_contexts=sample.contexts,
        )
        return MetricResult(name=self.name, value=float(result.value))


class ContextUtilizationMetric(Metric):
    name = "context_utilization"
    description = (
        "Measures retrieval ranking quality by checking if useful chunks appear before "
        "irrelevant ones, using the generated response instead of a reference answer."
    )
    required_fields = frozenset({SampleField.INPUT, SampleField.OUTPUT, SampleField.CONTEXTS})

    def __init__(self, llm: Any) -> None:
        self.evaluator = ContextUtilization(llm=llm)

    async def _score(self, sample: EvalSample) -> MetricResult:
        result = await self.evaluator.ascore(
            user_input=sample.input,
            response=sample.output,
            retrieved_contexts=sample.contexts,
        )
        return MetricResult(name=self.name, value=float(result.value))


class FaithfulnessMetric(Metric):
    name = "faithfulness"
    description = "Checks whether the response claims are supported by the retrieved contexts (0–1, higher is better)."
    required_fields = frozenset({SampleField.INPUT, SampleField.OUTPUT, SampleField.CONTEXTS})

    def __init__(self, llm: Any) -> None:
        self.evaluator = Faithfulness(llm=llm)

    async def _score(self, sample: EvalSample) -> MetricResult:
        result = await self.evaluator.ascore(
            user_input=sample.input,
            response=sample.output,
            retrieved_contexts=sample.contexts,
        )
        return MetricResult(name=self.name, value=float(result.value))


class ContextPrecisionMetric(Metric):
    name = "context_precision"
    description = (
        "Measures whether relevant chunks are ranked above irrelevant ones, "
        "using the reference answer to judge chunk usefulness (0–1, higher is better)."
    )
    required_fields = frozenset({SampleField.INPUT, SampleField.REFERENCE, SampleField.CONTEXTS})

    def __init__(self, llm: Any) -> None:
        self.evaluator = ContextPrecisionWithReference(llm=llm)

    async def _score(self, sample: EvalSample) -> MetricResult:
        result = await self.evaluator.ascore(
            user_input=sample.input,
            reference=sample.reference,
            retrieved_contexts=sample.contexts,
        )
        return MetricResult(name=self.name, value=float(result.value))


class ContextRecallMetric(Metric):
    name = "context_recall"
    description = "Measures how much of the reference answer can be attributed to the retrieved contexts (0–1, higher is better)."
    required_fields = frozenset({SampleField.INPUT, SampleField.REFERENCE, SampleField.CONTEXTS})

    def __init__(self, llm: Any) -> None:
        self.evaluator = ContextRecall(llm=llm)

    async def _score(self, sample: EvalSample) -> MetricResult:
        result = await self.evaluator.ascore(
            user_input=sample.input,
            reference=sample.reference,
            retrieved_contexts=sample.contexts,
        )
        return MetricResult(name=self.name, value=float(result.value))


class FactualCorrectnessMetric(Metric):
    name = "factual_correctness"
    description = (
        "Compares claims in the generated response against the reference answer "
        "using claim-level verification (0–1, higher is better)."
    )
    required_fields = frozenset({SampleField.OUTPUT, SampleField.REFERENCE})

    def __init__(self, llm: Any) -> None:
        self.evaluator = FactualCorrectness(llm=llm)

    async def _score(self, sample: EvalSample) -> MetricResult:
        result = await self.evaluator.ascore(
            response=sample.output,
            reference=sample.reference,
        )
        return MetricResult(name=self.name, value=float(result.value))


def build_metrics(chat_model: BaseChatModel) -> list[Metric]:
    """Create object-oriented Ragas metrics."""
    ragas_llm = RagasChatModel(chat_model)
    return [
        ContextRelevanceMetric(ragas_llm),
        ContextUtilizationMetric(ragas_llm),
        FaithfulnessMetric(ragas_llm),
        ContextPrecisionMetric(ragas_llm),
        ContextRecallMetric(ragas_llm),
        FactualCorrectnessMetric(ragas_llm),
    ]
