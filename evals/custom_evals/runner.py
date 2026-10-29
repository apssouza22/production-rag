"""Evaluator for Langfuse traces."""

import logging

from langfuse import Langfuse

from evals.core.datasets import Dataset
from evals.core.evaluator import Evaluator, MetricResultHandler
from evals.core.llm import make_eval_chat_model
from evals.core.metrics import Metric
from evals.core.reporting import (
    EvaluationReport,
    build_report,
    get_dataset_location,
    get_dataset_source_name,
    write_report,
)
from evals.core.runner import EvalRunner
from evals.core.schemas import EvalSample, MetricResult
from evals.custom_evals.config import get_settings
from evals.custom_evals.dataset import LangfuseTraceDataset
from evals.custom_evals.metrics import build_metrics

logger = logging.getLogger(__name__)


class LangfuseScorePublisher(MetricResultHandler):
    """Publish successful metric results for trace-backed samples."""

    def __init__(self, langfuse: Langfuse) -> None:
        self.langfuse = langfuse

    async def handle(self, sample: EvalSample, result: MetricResult) -> None:
        if sample.source_id is None or result.value is None or result.error:
            return
        self.langfuse.create_score(
            trace_id=sample.source_id,
            name=result.name,
            data_type="NUMERIC",
            value=result.value,
            comment=result.reasoning,
        )


class TraceEvaluationRunner(EvalRunner):
    """Configure trace evaluation and turn the shared response into a report."""

    def __init__(
        self,
        dataset: Dataset | None = None,
        metrics: list[Metric] | None = None,
    ) -> None:
        """Initialize trace dataset, metrics, and integrations."""
        self.settings = get_settings()

        self.chat_model = make_eval_chat_model(
            model=self.settings.evaluation_llm,
            api_key=self.settings.evaluation_api_key or self.settings.openai_api_key,
            base_url=self.settings.evaluation_base_url,
        )
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
        self.metrics = metrics if metrics is not None else build_metrics(self.chat_model)
        self.evaluator = Evaluator(
            self.metrics,
            result_handlers=[LangfuseScorePublisher(self.langfuse)],
            item_delay_seconds=self.settings.evaluation_sleep_time,
            progress_description="Evaluating traces",
        )
        self.report: EvaluationReport | None = None

    async def run_eval(self, generate_report_file: bool = True) -> dict:
        """Fetch traces from Langfuse, evaluate them, and push scores back."""
        response = await self.evaluator.evaluate(self.dataset)
        self.report = build_report(
            response=response,
            run_type="trace_evaluation",
            evaluation_model=self.settings.evaluation_llm,
            dataset_source=get_dataset_source_name(self.dataset),
            dataset_location=get_dataset_location(self.dataset),
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
