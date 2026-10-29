"""Run Ragas evaluation over rerank-search retrieval samples."""

import logging
import time
from pathlib import Path

from tqdm import tqdm

from evals.core.datasets import Dataset
from evals.core.metrics import MetricDefinition, evaluate_metric
from evals.core.reporting import (
    EvaluationReport,
    ReportItem,
    build_report,
    get_dataset_location,
    get_dataset_source_name,
    make_report_item,
    write_report,
)
from evals.core.schemas import EvalSample, MetricResult
from evals.rag_evals.config import RagEvalSettings, get_settings
from evals.rag_evals.dataset import DatasetBuilder, RagDataset
from evals.rag_evals.metrics import build_metrics, create_openai_client
from evals.rag_evals.retrieval import RerankSearchClient

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"


class RagEvaluator:
    """Evaluate retrieval and response quality with Ragas metrics."""

    def __init__(
        self,
        settings: RagEvalSettings | None = None,
        dataset: Dataset | None = None,
        metrics: list[MetricDefinition] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.openai_client = create_openai_client(self.settings)
        self.retrieval_client = RerankSearchClient(self.settings)
        self.dataset_builder = DatasetBuilder(
            settings=self.settings,
            retrieval_client=self.retrieval_client,
            openai_client=self.openai_client,
        )
        self.dataset: Dataset = dataset or RagDataset(self.settings.rag_eval_dataset_path, self.dataset_builder)
        self.metrics = metrics if metrics is not None else build_metrics(self.settings, self.openai_client)
        self.report: EvaluationReport | None = None

    async def run(self, generate_report_file: bool = True) -> dict:
        """Build the dataset from rerank search and score each sample."""
        start_time = time.time()
        samples = await self.dataset.load()
        total_items = getattr(self.dataset, "total_queries", len(samples))

        if not samples:
            raise RuntimeError("No evaluation samples were built. Check the dataset and RAG API availability.")

        report_items: list[ReportItem] = []
        for sample in tqdm(samples, desc="Evaluating samples"):
            report_items.append(await self._evaluate_sample(sample))

        self.report = build_report(
            run_type="rag_evaluation",
            evaluation_model=self.settings.evaluation_llm,
            dataset_source=get_dataset_source_name(self.dataset),
            dataset_location=get_dataset_location(self.dataset),
            total_items=total_items,
            items=report_items,
            metrics=self.metrics,
            duration_seconds=round(time.time() - start_time, 2),
            metadata={
                "retrieval": {
                    "top_k": self.settings.rag_eval_top_k,
                    "use_hybrid": self.settings.rag_eval_use_hybrid,
                    "rerank_enabled": self.settings.rag_eval_rerank_enabled,
                    "api_base_url": self.settings.rag_api_base_url,
                }
            },
        )

        if generate_report_file:
            write_report(self.report, reports_dir=REPORTS_DIR)

        logger.info(
            "Ragas evaluation completed: samples=%s duration_seconds=%s",
            self.report.evaluated_items,
            self.report.duration_seconds,
        )
        return self.report.model_dump(mode="json")

    async def _evaluate_sample(self, sample: EvalSample) -> ReportItem:
        metric_scores: list[MetricResult] = []
        for metric in self.metrics:
            try:
                metric_scores.append(await evaluate_metric(metric, sample))
            except Exception as exc:
                logger.exception("Metric %s failed for query preview=%r", metric.name, sample.user_input[:80])
                metric_scores.append(MetricResult(name=metric.name, error=str(exc)))

        return make_report_item(
            sample,
            metric_scores,
            item_id=sample.source_id or sample.user_input[:80],
        )
