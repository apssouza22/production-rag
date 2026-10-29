"""Evaluator for Langfuse traces."""

import logging
import time
from time import sleep

import openai
from langfuse import Langfuse
from tqdm import tqdm

from evals.core.datasets import Dataset
from evals.core.metrics import MetricDefinition, evaluate_metric
from evals.core.schemas import MetricResult
from evals.custom_evals.config import build_openai_client_kwargs, get_settings
from evals.custom_evals.dataset import LangfuseTraceDataset
from evals.custom_evals.helpers import (
    calculate_avg_scores,
    generate_report,
    initialize_metrics_summary,
    initialize_report,
    process_trace_results,
    update_failure_metrics,
    update_success_metrics,
)
from evals.custom_evals.metrics import build_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates model outputs using predefined metrics."""

    def __init__(
        self,
        dataset: Dataset | None = None,
        metrics: list[MetricDefinition] | None = None,
    ) -> None:
        """Initialize Evaluator with OpenAI and Langfuse clients."""
        self.settings = get_settings()

        client_kwargs = build_openai_client_kwargs(
            self.settings,
            api_key=self.settings.evaluation_api_key or self.settings.openai_api_key or None,
            base_url=self.settings.evaluation_base_url or None,
        )
        self.client = openai.AsyncOpenAI(**client_kwargs)
        self.langfuse = Langfuse(
            public_key=self.settings.langfuse_public_key,
            secret_key=self.settings.langfuse_secret_key,
            host=self.settings.langfuse_host,
            timeout=60,
        )
        self.dataset: Dataset = dataset or LangfuseTraceDataset(self.langfuse)
        self.metrics = metrics if metrics is not None else build_metrics(self.client, self.settings.evaluation_llm)
        self.report = initialize_report(self.settings.evaluation_llm)
        initialize_metrics_summary(self.report, [{"name": metric.name} for metric in self.metrics])

    async def run(self, generate_report_file: bool = True) -> dict:
        """Fetch traces from Langfuse, evaluate them, and push scores back."""
        start_time = time.time()
        samples = await self.dataset.load()
        self.report["total_traces"] = len(samples)

        trace_results = {}

        for index, sample in enumerate(tqdm(samples, desc="Evaluating traces")):
            trace_id = sample.source_id or f"sample-{index}"
            trace_results[trace_id] = {
                "success": False,
                "metrics_evaluated": 0,
                "metrics_succeeded": 0,
                "metrics_results": {},
            }

            for metric in tqdm(self.metrics, desc=f"Applying metrics to trace {trace_id[:8]}...", leave=False):
                metric_name = metric.name
                try:
                    score = await evaluate_metric(metric, sample)
                except Exception as exc:
                    logger.exception("Metric %s evaluation failed", metric_name)
                    score = MetricResult(name=metric_name, error=str(exc))

                if score.value is not None and not score.error:
                    if sample.source_id:
                        self._push_to_langfuse(sample.source_id, score)
                    update_success_metrics(self.report, trace_id, metric_name, score, trace_results)
                else:
                    logger.error("Metric %s evaluation failed: %s", metric_name, score.error)
                    update_failure_metrics(self.report, trace_id, metric_name, trace_results)

                trace_results[trace_id]["metrics_evaluated"] += 1

            process_trace_results(self.report, trace_id, trace_results, len(self.metrics))
            sleep(self.settings.evaluation_sleep_time)

        self.report["duration_seconds"] = round(time.time() - start_time, 2)
        calculate_avg_scores(self.report)

        if generate_report_file:
            generate_report(self.report)

        logger.info(
            "Evaluation completed: total_traces=%s successful_traces=%s failed_traces=%s duration_seconds=%s",
            self.report["total_traces"],
            self.report["successful_traces"],
            self.report["failed_traces"],
            self.report["duration_seconds"],
        )
        return self.report

    def _push_to_langfuse(self, trace_id: str, score: MetricResult) -> None:
        """Push evaluation score to Langfuse."""
        self.langfuse.create_score(
            trace_id=trace_id,
            name=score.name,
            data_type="NUMERIC",
            value=score.value,
            comment=score.reasoning,
        )
