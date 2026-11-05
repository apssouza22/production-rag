"""Run the local-PDF experiment: ingest, generate queries, retrieve, and score with Ragas."""

import asyncio
import logging
from pathlib import Path
from typing import Any

from src.domain.jinaai.factory import make_embeddings_service, make_reranker_client
from src.domain.rrf.factory import make_paper_rrf_search
from src.domain.search.factory import make_rerank_search_service

from evals.core.evaluator import Evaluator
from evals.core.llm import make_eval_chat_model
from evals.core.metrics import Metric
from evals.core.report_publisher import ReportPublisher, make_file_report_publisher
from evals.core.reporting import (
    EvaluationReport,
    build_report,
    get_dataset_location,
    get_dataset_source_name,
)
from evals.core.runner import EvalRunner
from evals.experiments.config import ExperimentRunConfig, get_settings
from evals.experiments.dataset import ExperimentDataset
from evals.experiments.ingestion import ensure_ingested
from evals.experiments.question_gen import QuestionGenerator
from evals.rag_evals.metrics import build_metrics

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"


class ExperimentRunner(EvalRunner):
    """Wire ingestion, query generation, and retrieval into the shared evaluator."""

    def __init__(
        self,
        run_config: ExperimentRunConfig | None = None,
        metrics: list[Metric] | None = None,
        *,
        force_ingest: bool = False,
        regenerate_queries: bool = False,
        report_publisher: ReportPublisher | None = None,
    ) -> None:
        self.run_config = run_config or get_settings()
        self.settings = self.run_config.experiment
        self.force_ingest = force_ingest
        self.regenerate_queries = regenerate_queries

        self.chat_model = make_eval_chat_model(
            model=self.settings.evaluation_llm,
            api_key=self.settings.judge_api_key,
            base_url=self.settings.evaluation_base_url,
        )
        self.paper_rrf_search = make_paper_rrf_search(
            ingestion_name=self.settings.experiment_ingestion_name,
        )
        self.rerank_search_service = make_rerank_search_service(
            paper_rrf_search=self.paper_rrf_search,
            embeddings_client=make_embeddings_service(),
            reranker_client=make_reranker_client(),
            config=self.run_config.query.to_rerank_config(),
        )
        self.question_generator = QuestionGenerator(
            paper_rrf_search=self.paper_rrf_search,
            chat_model=self.chat_model,
            settings=self.settings,
        )
        self.metrics = metrics if metrics is not None else build_metrics(self.chat_model)
        self.evaluator = Evaluator(self.metrics, progress_description="Evaluating experiment samples")
        self.report_publisher = report_publisher or make_file_report_publisher(reports_dir=REPORTS_DIR)
        self.report: EvaluationReport | None = None

    async def run_eval(self) -> dict[str, Any]:
        """Ingest if needed, build the dataset from retrieval, and score every sample."""
        # The ingestion pipeline is synchronous and calls asyncio.run() internally.
        ingestion = await asyncio.to_thread(
            ensure_ingested,
            self.paper_rrf_search,
            self.settings,
            force=self.force_ingest,
        )
        queries = await self.question_generator.build_queries(regenerate=self.regenerate_queries)

        dataset = ExperimentDataset(
            queries=queries,
            rerank_search_service=self.rerank_search_service,
            chat_model=self.chat_model,
            settings=self.settings,
        )
        response = await self.evaluator.evaluate(dataset)
        if not response.items:
            raise RuntimeError("No evaluation samples were built. Check that the experiment partition is indexed.")

        query_settings = self.run_config.query.model_dump(mode="json")
        self.report = build_report(
            response=response,
            run_type="experiment_evaluation",
            evaluation_model=self.settings.evaluation_llm,
            dataset_source=get_dataset_source_name(dataset),
            dataset_location=get_dataset_location(dataset),
            metadata={
                "settings_path": str(self.run_config.config_path),
                "ingestion": {
                    key: ingestion[key]
                    for key in ("skipped", "chunk_count", "ingestion_name", "kg")
                    if key in ingestion
                },
                "retrieval": {
                    "top_k": self.settings.experiment_top_k,
                    "ingestion_name": self.settings.experiment_ingestion_name,
                    **query_settings,
                },
            },
        )

        self.report_publisher.publish(self.report)

        logger.info(
            "Experiment evaluation completed: samples=%s duration_seconds=%s",
            self.report.evaluated_items,
            self.report.duration_seconds,
        )
        return self.report.model_dump(mode="json")
