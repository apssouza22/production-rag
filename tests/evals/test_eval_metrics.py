"""Tests for object-oriented metric implementations."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from evals.core.schemas import EvalSample
from evals.custom_evals.metrics import PromptMetric
from evals.custom_evals.schemas import ScoreSchema
from evals.rag_evals.config import RagEvalSettings
from evals.rag_evals.metrics import (
    ContextPrecisionMetric,
    ContextRecallMetric,
    ContextRelevanceMetric,
    ContextUtilizationMetric,
    FactualCorrectnessMetric,
    FaithfulnessMetric,
    build_metrics,
)


@pytest.mark.asyncio
async def test_prompt_metric_scores_sample() -> None:
    client = Mock()
    client.beta.chat.completions.parse = AsyncMock(
        return_value=Mock(choices=[Mock(message=Mock(parsed=ScoreSchema(score=0.8, reasoning="Relevant response")))])
    )
    metric = PromptMetric(
        name="relevancy",
        prompt="Judge relevance.",
        client=client,
        model="judge-model",
    )

    result = await metric.score(EvalSample(input="question", output="answer"))

    assert result.name == "relevancy"
    assert result.value == 0.8
    assert result.reasoning == "Relevant response"


def test_build_ragas_metrics_returns_concrete_metric_objects() -> None:
    evaluator = Mock()
    with (
        patch("evals.rag_evals.metrics.llm_factory", return_value=Mock()),
        patch("evals.rag_evals.metrics.ContextRelevance", return_value=evaluator),
        patch("evals.rag_evals.metrics.ContextUtilization", return_value=evaluator),
        patch("evals.rag_evals.metrics.Faithfulness", return_value=evaluator),
        patch("evals.rag_evals.metrics.ContextPrecisionWithReference", return_value=evaluator),
        patch("evals.rag_evals.metrics.ContextRecall", return_value=evaluator),
        patch("evals.rag_evals.metrics.FactualCorrectness", return_value=evaluator),
    ):
        metrics = build_metrics(RagEvalSettings(), Mock())

    assert [type(metric) for metric in metrics] == [
        ContextRelevanceMetric,
        ContextUtilizationMetric,
        FaithfulnessMetric,
        ContextPrecisionMetric,
        ContextRecallMetric,
        FactualCorrectnessMetric,
    ]
