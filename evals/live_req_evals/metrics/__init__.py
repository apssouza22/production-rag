"""Prompt-based metrics for trace evaluations."""

from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from evals.core.metrics import Metric, PromptMetric

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_metric_prompts() -> dict[str, str]:
    """Load metric prompts in deterministic name order."""
    return {prompt_path.stem: prompt_path.read_text(encoding="utf-8") for prompt_path in sorted(PROMPTS_DIR.glob("*.md"))}


def build_metrics(chat_model: BaseChatModel) -> list[Metric]:
    """Create object-oriented prompt judges."""
    return [PromptMetric(name=name, prompt=prompt, chat_model=chat_model) for name, prompt in load_metric_prompts().items()]


# Kept for callers that only need prompt metadata.
metrics = [{"name": name, "prompt": prompt} for name, prompt in load_metric_prompts().items()]
