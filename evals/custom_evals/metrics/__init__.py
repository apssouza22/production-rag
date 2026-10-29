"""Prompt-based metrics for trace evaluations."""

import asyncio
import logging
from pathlib import Path

from openai import AsyncOpenAI

from evals.core.metrics import MetricDefinition, MetricScorer, SampleField
from evals.core.schemas import EvalSample, MetricResult
from evals.custom_evals.schemas import ScoreSchema

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_metric_prompts() -> dict[str, str]:
    """Load metric prompts in deterministic name order."""
    return {prompt_path.stem: prompt_path.read_text(encoding="utf-8") for prompt_path in sorted(PROMPTS_DIR.glob("*.md"))}


def build_metrics(client: AsyncOpenAI, model: str) -> list[MetricDefinition]:
    """Create prompt judges that implement the shared metric contract."""

    def make_scorer(name: str, prompt: str) -> MetricScorer:
        async def score(sample: EvalSample) -> MetricResult:
            for attempt in range(3):
                try:
                    response = await client.beta.chat.completions.parse(
                        model=model,
                        messages=[
                            {"role": "system", "content": prompt},
                            {
                                "role": "user",
                                "content": f"Input: {sample.input}\nGeneration: {sample.output}",
                            },
                        ],
                        response_format=ScoreSchema,
                    )
                    parsed = response.choices[0].message.parsed
                    if parsed is None:
                        return MetricResult(name=name, error="judge returned no parsed score")
                    return MetricResult(name=name, value=parsed.score, reasoning=parsed.reasoning)
                except Exception as exc:
                    if attempt == 2:
                        logger.exception("Metric %s failed after retries", name)
                        return MetricResult(name=name, error=str(exc))
                    logger.error("Metric %s failed (retrying in 10s): %s", name, exc)
                    await asyncio.sleep(10)
            return MetricResult(name=name, error="scoring failed")

        return score

    required_fields = frozenset({SampleField.INPUT, SampleField.OUTPUT})
    return [
        MetricDefinition(
            name=name,
            description=f"LLM judge using the {name} prompt.",
            scorer=make_scorer(name, prompt),
            required_fields=required_fields,
        )
        for name, prompt in load_metric_prompts().items()
    ]


# Kept for callers that only need prompt metadata.
metrics = [{"name": name, "prompt": prompt} for name, prompt in load_metric_prompts().items()]
