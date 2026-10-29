"""Run Ragas evaluation over rerank-search retrieval samples."""

import logging
from pathlib import Path

from evals.core.datasets import Dataset
from evals.core.evaluator import Evaluator
from evals.core.metrics import Metric
from evals.core.reporting import (
    EvaluationReport,
    build_report,
    get_dataset_location,
    get_dataset_source_name,
    write_report,
)
from evals.core.runner import EvalRunner
from evals.rag_evals.config import RagEvalSettings, get_settings
from evals.rag_evals.dataset import DatasetBuilder, RagDataset
from evals.rag_evals.metrics import build_metrics, create_openai_client
from evals.rag_evals.retrieval import RerankSearchClient

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"


class RagEvaluationRunner(EvalRunner):
    """Configure RAG evaluation and turn the shared response into a report."""

    def __init__(
        self,
        settings: RagEvalSettings | None = None,
        dataset: Dataset | None = None,
        metrics: list[Metric] | None = None,
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
        self.evaluator = Evaluator(self.metrics, progress_description="Evaluating RAG samples")
        self.report: EvaluationReport | None = None

    async def run_eval(self, generate_report_file: bool = True) -> dict:
        """Build the dataset from rerank search and score each sample."""
        response = await self.evaluator.evaluate(self.dataset)
        if not response.items:
            raise RuntimeError("No evaluation samples were built. Check the dataset and RAG API availability.")

        self.report = build_report(
            response=response,
            run_type="rag_evaluation",
            evaluation_model=self.settings.evaluation_llm,
            dataset_source=get_dataset_source_name(self.dataset),
            dataset_location=get_dataset_location(self.dataset),
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
