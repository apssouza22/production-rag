"""Ragas metrics for retrieval and response evaluation."""

from dataclasses import dataclass
from typing import Awaitable, Callable

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecisionWithReference,
    ContextRecall,
    ContextRelevance,
    ContextUtilization,
    FactualCorrectness,
    Faithfulness,
)

from rag_evals.config import RagEvalSettings, build_eval_openai_client_kwargs
from rag_evals.schemas import EvalSample, MetricScore

MetricScorer = Callable[[EvalSample], Awaitable[MetricScore]]


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    description: str
    requires_reference: bool
    scorer: MetricScorer


def build_metrics(settings: RagEvalSettings, openai_client: AsyncOpenAI) -> list[MetricDefinition]:
    """Create Ragas scorers for retrieval and response evaluation."""
    ragas_llm = llm_factory(settings.evaluation_llm, client=openai_client)

    context_relevance = ContextRelevance(llm=ragas_llm)
    context_utilization = ContextUtilization(llm=ragas_llm)
    faithfulness = Faithfulness(llm=ragas_llm)
    context_precision = ContextPrecisionWithReference(llm=ragas_llm)
    context_recall = ContextRecall(llm=ragas_llm)
    factual_correctness = FactualCorrectness(llm=ragas_llm)

    async def score_context_relevance(sample: EvalSample) -> MetricScore:
        result = await context_relevance.ascore(
            user_input=sample.user_input,
            retrieved_contexts=sample.retrieved_contexts,
        )
        return MetricScore(name="context_relevance", value=float(result.value))

    async def score_context_utilization(sample: EvalSample) -> MetricScore:
        result = await context_utilization.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        )
        return MetricScore(name="context_utilization", value=float(result.value))

    async def score_faithfulness(sample: EvalSample) -> MetricScore:
        result = await faithfulness.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=sample.retrieved_contexts,
        )
        return MetricScore(name="faithfulness", value=float(result.value))

    async def score_context_precision(sample: EvalSample) -> MetricScore:
        result = await context_precision.ascore(
            user_input=sample.user_input,
            reference=sample.reference,
            retrieved_contexts=sample.retrieved_contexts,
        )
        return MetricScore(name="context_precision", value=float(result.value))

    async def score_context_recall(sample: EvalSample) -> MetricScore:
        result = await context_recall.ascore(
            user_input=sample.user_input,
            reference=sample.reference,
            retrieved_contexts=sample.retrieved_contexts,
        )
        return MetricScore(name="context_recall", value=float(result.value))

    async def score_factual_correctness(sample: EvalSample) -> MetricScore:
        result = await factual_correctness.ascore(
            response=sample.response,
            reference=sample.reference,
        )
        return MetricScore(name="factual_correctness", value=float(result.value))

    return [
        MetricDefinition(
            name="context_relevance",
            description="Checks whether retrieved chunks are relevant to the query (0–1, higher is better).",
            requires_reference=False,
            scorer=score_context_relevance,
        ),
        MetricDefinition(
            name="context_utilization",
            description=(
                "Measures retrieval ranking quality by checking if useful chunks appear before "
                "irrelevant ones, using the generated response instead of a reference answer."
            ),
            requires_reference=False,
            scorer=score_context_utilization,
        ),
        MetricDefinition(
            name="faithfulness",
            description="Checks whether the response claims are supported by the retrieved contexts (0–1, higher is better).",
            requires_reference=False,
            scorer=score_faithfulness,
        ),
        MetricDefinition(
            name="context_precision",
            description=(
                "Measures whether relevant chunks are ranked above irrelevant ones, "
                "using the reference answer to judge chunk usefulness (0–1, higher is better)."
            ),
            requires_reference=True,
            scorer=score_context_precision,
        ),
        MetricDefinition(
            name="context_recall",
            description=(
                "Measures how much of the reference answer can be attributed to the retrieved contexts "
                "(0–1, higher is better)."
            ),
            requires_reference=True,
            scorer=score_context_recall,
        ),
        MetricDefinition(
            name="factual_correctness",
            description=(
                "Compares claims in the generated response against the reference answer "
                "using claim-level verification (0–1, higher is better)."
            ),
            requires_reference=True,
            scorer=score_factual_correctness,
        ),
    ]


def create_openai_client(settings: RagEvalSettings) -> AsyncOpenAI:
    return AsyncOpenAI(**build_eval_openai_client_kwargs(settings))
