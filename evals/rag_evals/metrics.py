"""Reference-free Ragas metrics for retrieval evaluation."""

from dataclasses import dataclass
from typing import Awaitable, Callable

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRelevance, ContextUtilization, Faithfulness

from rag_evals.config import RagEvalSettings, build_eval_openai_client_kwargs
from rag_evals.schemas import EvalSample, MetricScore

MetricScorer = Callable[[EvalSample], Awaitable[MetricScore]]


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    description: str
    scorer: MetricScorer


def build_reference_free_metrics(settings: RagEvalSettings, openai_client: AsyncOpenAI) -> list[MetricDefinition]:
    """Create Ragas scorers that do not require reference answers or contexts."""
    ragas_llm = llm_factory(settings.evaluation_llm, client=openai_client)

    context_relevance = ContextRelevance(llm=ragas_llm)
    context_utilization = ContextUtilization(llm=ragas_llm)
    faithfulness = Faithfulness(llm=ragas_llm)

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

    return [
        MetricDefinition(
            name="context_relevance",
            description="Measures whether retrieved contexts are pertinent to the query.",
            scorer=score_context_relevance,
        ),
        MetricDefinition(
            name="context_utilization",
            description="Context precision without reference; ranks useful retrieved chunks using the response.",
            scorer=score_context_utilization,
        ),
        MetricDefinition(
            name="faithfulness",
            description="Measures whether the response is supported by retrieved contexts.",
            scorer=score_faithfulness,
        ),
    ]


def create_openai_client(settings: RagEvalSettings) -> AsyncOpenAI:
    return AsyncOpenAI(**build_eval_openai_client_kwargs(settings))
