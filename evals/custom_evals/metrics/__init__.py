"""Prompt-based metrics for trace evaluations."""

import asyncio
import logging
from pathlib import Path

from openai import AsyncOpenAI

from evals.core.metrics import Metric, SampleField
from evals.core.schemas import EvalSample, MetricResult
from evals.custom_evals.schemas import ScoreSchema

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_metric_prompts() -> dict[str, str]:
    """Load metric prompts in deterministic name order."""
    return {prompt_path.stem: prompt_path.read_text(encoding="utf-8") for prompt_path in sorted(PROMPTS_DIR.glob("*.md"))}


class PromptMetric(Metric):
    """LLM judge driven by a markdown system prompt."""

    required_fields = frozenset({SampleField.INPUT, SampleField.OUTPUT})

    def __init__(
        self,
        *,
        name: str,
        prompt: str,
        client: AsyncOpenAI,
        model: str,
        max_retries: int = 3,
        retry_delay_seconds: float = 10,
    ) -> None:
        self.name = name
        self.description = f"LLM judge using the {name} prompt."
        self.prompt = prompt
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    async def _score(self, sample: EvalSample) -> MetricResult:
        for attempt in range(self.max_retries):
            try:
                response = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {
                            "role": "user",
                            "content": f"Input: {sample.input}\nGeneration: {sample.output}",
                        },
                    ],
                    response_format=ScoreSchema,
                )
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    return MetricResult(name=self.name, error="judge returned no parsed score")
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


def build_metrics(client: AsyncOpenAI, model: str) -> list[Metric]:
    """Create object-oriented prompt judges."""
    return [PromptMetric(name=name, prompt=prompt, client=client, model=model) for name, prompt in load_metric_prompts().items()]


# Kept for callers that only need prompt metadata.
metrics = [{"name": name, "prompt": prompt} for name, prompt in load_metric_prompts().items()]
