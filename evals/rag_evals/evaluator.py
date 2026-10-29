"""Run Ragas evaluation over rerank-search retrieval samples."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from evals.core.datasets import Dataset
from evals.core.metrics import MetricDefinition, evaluate_metric
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
        self.report = self._initialize_report()

    async def run(self, generate_report_file: bool = True) -> dict[str, Any]:
        """Build the dataset from rerank search and score each sample."""
        start_time = time.time()
        samples = await self.dataset.load()

        self.report["total_queries"] = getattr(self.dataset, "total_queries", len(samples))
        self.report["evaluated_samples"] = len(samples)

        if not samples:
            raise RuntimeError("No evaluation samples were built. Check the dataset and RAG API availability.")

        for sample in tqdm(samples, desc="Evaluating samples"):
            sample_result = await self._evaluate_sample(sample)
            self._update_report(sample_result)

        self.report["duration_seconds"] = round(time.time() - start_time, 2)
        self._finalize_metric_averages()

        if generate_report_file:
            self._write_report_file()

        logger.info(
            "Ragas evaluation completed: samples=%s duration_seconds=%s",
            self.report["evaluated_samples"],
            self.report["duration_seconds"],
        )
        return self.report

    async def _evaluate_sample(self, sample: EvalSample) -> dict[str, Any]:
        metric_scores: list[MetricResult] = []
        for metric in self.metrics:
            try:
                metric_scores.append(await evaluate_metric(metric, sample))
            except Exception as exc:
                logger.exception("Metric %s failed for query preview=%r", metric.name, sample.user_input[:80])
                metric_scores.append(MetricResult(name=metric.name, error=str(exc)))

        return {
            "user_input": sample.user_input,
            "reference_preview": (sample.reference or "")[:200],
            "response_preview": sample.response[:200],
            "context_count": len(sample.contexts),
            "rerank_applied": sample.rerank_applied,
            "metrics": [score.model_dump(exclude={"reasoning"}) for score in metric_scores],
        }

    def _initialize_report(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(),
            "evaluation_llm": self.settings.evaluation_llm,
            "dataset_path": str(self.settings.rag_eval_dataset_path),
            "retrieval": {
                "top_k": self.settings.rag_eval_top_k,
                "use_hybrid": self.settings.rag_eval_use_hybrid,
                "rerank_enabled": self.settings.rag_eval_rerank_enabled,
                "api_base_url": self.settings.rag_api_base_url,
            },
            "metrics": [
                {
                    "name": metric.name,
                    "description": metric.description,
                    "requires_reference": metric.requires_reference,
                }
                for metric in self.metrics
            ],
            "total_queries": 0,
            "evaluated_samples": 0,
            "duration_seconds": 0,
            "metrics_summary": {
                metric.name: {
                    "description": metric.description,
                    "requires_reference": metric.requires_reference,
                    "success_count": 0,
                    "failure_count": 0,
                    "avg_score": 0.0,
                }
                for metric in self.metrics
            },
            "sample_results": [],
        }

    def _update_report(self, sample_result: dict[str, Any]) -> None:
        self.report["sample_results"].append(sample_result)
        for metric_score in sample_result["metrics"]:
            summary = self.report["metrics_summary"][metric_score["name"]]
            if metric_score.get("value") is None:
                summary["failure_count"] += 1
                continue
            summary["success_count"] += 1
            summary["avg_score"] += metric_score["value"]

    def _finalize_metric_averages(self) -> None:
        for summary in self.report["metrics_summary"].values():
            if summary["success_count"] > 0:
                summary["avg_score"] = round(summary["avg_score"] / summary["success_count"], 4)

    def _write_report_file(self) -> str:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"ragas_evaluation_report_{timestamp}.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(self.report, handle, indent=2)
        self.report["report_path"] = str(report_path)
        logger.info("Ragas evaluation report written to %s", report_path)
        return str(report_path)
