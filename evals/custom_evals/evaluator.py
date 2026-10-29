"""Evaluator for Langfuse traces."""

import logging
import time
from time import sleep

import openai
from langfuse import Langfuse
from tqdm import tqdm

from evals.core.datasets import Dataset
from evals.core.metrics import Metric
from evals.core.reporting import (
    EvaluationReport,
    ReportItem,
    build_report,
    get_dataset_location,
    get_dataset_source_name,
    make_report_item,
    write_report,
)
from evals.core.schemas import MetricResult
from evals.custom_evals.config import build_openai_client_kwargs, get_settings
from evals.custom_evals.dataset import LangfuseTraceDataset
from evals.custom_evals.metrics import build_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates model outputs using predefined metrics."""

    def __init__(
        self,
        dataset: Dataset | None = None,
        metrics: list[Metric] | None = None,
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
        self.dataset: Dataset = dataset or LangfuseTraceDataset(
            self.langfuse,
            source_location=self.settings.langfuse_host,
        )
        self.metrics = metrics if metrics is not None else build_metrics(self.client, self.settings.evaluation_llm)
        self.report: EvaluationReport | None = None

    async def run(self, generate_report_file: bool = True) -> dict:
        """Fetch traces from Langfuse, evaluate them, and push scores back."""
        start_time = time.time()
        samples = await self.dataset.load()
        report_items: list[ReportItem] = []

        for index, sample in enumerate(tqdm(samples, desc="Evaluating traces")):
            trace_id = sample.source_id or f"sample-{index}"
            metric_results: list[MetricResult] = []

            for metric in tqdm(self.metrics, desc=f"Applying metrics to trace {trace_id[:8]}...", leave=False):
                try:
                    score = await metric.score(sample)
                except Exception as exc:
                    logger.exception("Metric %s evaluation failed", metric.name)
                    score = MetricResult(name=metric.name, error=str(exc))

                if score.value is not None and not score.error:
                    if sample.source_id:
                        self._push_to_langfuse(sample.source_id, score)
                else:
                    logger.error("Metric %s evaluation failed: %s", metric.name, score.error)
                metric_results.append(score)

            report_items.append(make_report_item(sample, metric_results, item_id=trace_id))
            sleep(self.settings.evaluation_sleep_time)

        self.report = build_report(
            run_type="trace_evaluation",
            evaluation_model=self.settings.evaluation_llm,
            dataset_source=get_dataset_source_name(self.dataset),
            dataset_location=get_dataset_location(self.dataset),
            total_items=len(samples),
            items=report_items,
            metrics=self.metrics,
            duration_seconds=round(time.time() - start_time, 2),
        )

        if generate_report_file:
            write_report(self.report)

        logger.info(
            "Evaluation completed: total_items=%s successful_items=%s failed_items=%s duration_seconds=%s",
            self.report.total_items,
            self.report.successful_items,
            self.report.failed_items,
            self.report.duration_seconds,
        )
        return self.report.model_dump(mode="json")

    def _push_to_langfuse(self, trace_id: str, score: MetricResult) -> None:
        """Push evaluation score to Langfuse."""
        self.langfuse.create_score(
            trace_id=trace_id,
            name=score.name,
            data_type="NUMERIC",
            value=score.value,
            comment=score.reasoning,
        )
