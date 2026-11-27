"""Run text-to-SQL execution-accuracy evaluation."""

import logging
from pathlib import Path
from typing import Any

from src.agents.texttosql import TextToSQLService, make_text_to_sql_service
from src.config import get_settings as get_app_settings

from evals.core.datasets import Dataset
from evals.core.evaluator import Evaluator
from evals.core.metrics import Metric
from evals.core.report_publisher import ReportPublisher, make_file_report_publisher
from evals.core.reporting import (
    EvaluationReport,
    build_report,
    get_dataset_location,
    get_dataset_source_name,
)
from evals.core.runner import EvalRunner
from evals.texttosql_evals.config import TextToSQLEvalSettings, get_settings
from evals.texttosql_evals.dataset import DatasetBuilder, TextToSQLDataset
from evals.texttosql_evals.metrics import build_metrics
from evals.texttosql_evals.sql_executor import SqlResultExecutor

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"


class TextToSQLEvaluationRunner(EvalRunner):
    """Configure text-to-SQL evaluation and emit a unified report."""

    def __init__(
        self,
        settings: TextToSQLEvalSettings | None = None,
        service: TextToSQLService | None = None,
        sql_executor: SqlResultExecutor | None = None,
        dataset: Dataset | None = None,
        metrics: list[Metric] | None = None,
        report_publisher: ReportPublisher | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        app_settings = get_app_settings()
        self.service = service or make_text_to_sql_service(langfuse_tracer=None)
        agent_config = self.service.agent_config

        if sql_executor is None:
            sql_executor = SqlResultExecutor(
                app_settings.text_to_sql.database_url,
                include_tables=agent_config.include_tables,
            )
        self.sql_executor = sql_executor

        self.dataset_builder = DatasetBuilder(
            settings=self.settings,
            service=self.service,
            sql_executor=self.sql_executor,
        )
        self.dataset: Dataset = dataset or TextToSQLDataset(
            self.settings.text_to_sql_eval_dataset_path,
            self.dataset_builder,
        )
        self.metrics = metrics if metrics is not None else build_metrics()
        self.evaluator = Evaluator(
            self.metrics,
            progress_description="Evaluating text-to-SQL samples",
            item_delay_seconds=self.settings.text_to_sql_eval_item_delay_seconds,
        )
        self.report_publisher = report_publisher or make_file_report_publisher(reports_dir=REPORTS_DIR)
        self.report: EvaluationReport | None = None

    async def run_eval(self) -> dict[str, Any]:
        """Build the dataset from the agent and score execution accuracy."""
        response = await self.evaluator.evaluate(self.dataset)
        if not response.items:
            raise RuntimeError("No evaluation samples were built. Check the dataset and database connectivity.")

        agent_config = self.service.agent_config
        self.report = build_report(
            response=response,
            run_type="texttosql_evaluation",
            evaluation_model=self.settings.text_to_sql_eval_model or agent_config.model,
            dataset_source=get_dataset_source_name(self.dataset),
            dataset_location=get_dataset_location(self.dataset),
            metadata={
                "text_to_sql": {
                    "dialect": agent_config.dialect,
                    "include_tables": agent_config.include_tables,
                    "database_url_redacted": True,
                }
            },
        )

        self.report_publisher.publish(self.report)

        logger.info(
            "Text-to-SQL evaluation completed: samples=%s duration_seconds=%s",
            self.report.evaluated_items,
            self.report.duration_seconds,
        )
        return self.report.model_dump(mode="json")