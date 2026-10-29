"""Run Ragas RAG evaluation as a Langfuse experiment."""

import logging
from pathlib import Path
from typing import Any

from langfuse import Evaluation, Langfuse
from langfuse.experiment import EvaluatorFunction, ExperimentResult, LocalExperimentItem

from evals.core.metrics import Metric
from evals.core.reporting import EvaluationReport, ReportItem, build_report, make_report_item, write_report
from evals.core.schemas import EvalSample, MetricResult
from evals.rag_evals.config import RagEvalSettings, get_settings
from evals.rag_evals.dataset import DatasetBuilder, load_queries
from evals.rag_evals.metrics import build_metrics, create_openai_client
from evals.rag_evals.retrieval import RerankSearchClient
from evals.rag_evals.schemas import EvalQuery

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def queries_to_experiment_items(queries: list[EvalQuery]) -> list[LocalExperimentItem]:
    """Convert local eval queries into Langfuse local experiment items."""
    items: list[LocalExperimentItem] = []
    for query in queries:
        metadata: dict[str, Any] = {}
        if query.response:
            metadata["response"] = query.response
        items.append(
            {
                "input": query.user_input,
                "expected_output": query.reference,
                "metadata": metadata or None,
            }
        )
    return items


def experiment_item_to_sample(item: Any, output: Any = None) -> EvalSample:
    """Normalize a local or hosted Langfuse experiment item."""
    output_payload = output if isinstance(output, dict) else {}
    raw_metadata = item.get("metadata") if isinstance(item, dict) else getattr(item, "metadata", None)
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    metadata["rerank_applied"] = bool(output_payload.get("rerank_applied", False))
    source_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
    return EvalSample(
        input=extract_item_input(item),
        output=output_payload.get("response") or extract_item_preset_response(item),
        reference=extract_item_expected_output(item),
        contexts=list(output_payload.get("retrieved_contexts", [])),
        source_id=str(source_id) if source_id else None,
        metadata=metadata,
    )


def extract_item_input(item: Any) -> str:
    """Read the query text from a local or hosted Langfuse dataset item."""
    raw_input = item["input"] if isinstance(item, dict) else item.input
    if isinstance(raw_input, dict):
        for key in ("user_input", "query", "question"):
            if raw_input.get(key):
                return str(raw_input[key])
    return str(raw_input)


def extract_item_expected_output(item: Any) -> str:
    """Read the reference answer from a local or hosted Langfuse dataset item."""
    if isinstance(item, dict):
        raw_expected = item.get("expected_output")
        metadata = item.get("metadata") or {}
    else:
        raw_expected = item.expected_output
        metadata = item.metadata or {}

    if raw_expected:
        return str(raw_expected)
    if isinstance(metadata, dict) and metadata.get("reference"):
        return str(metadata["reference"])
    return ""


def extract_item_preset_response(item: Any) -> str | None:
    """Read an optional pre-generated response from item metadata."""
    if isinstance(item, dict):
        metadata = item.get("metadata") or {}
        raw_input = item.get("input")
    else:
        metadata = item.metadata or {}
        raw_input = item.input

    if isinstance(metadata, dict) and metadata.get("response"):
        return str(metadata["response"])
    if isinstance(raw_input, dict) and raw_input.get("response"):
        return str(raw_input["response"])
    return None


def sample_to_task_output(sample: EvalSample) -> dict[str, Any]:
    """Serialize a Ragas sample as Langfuse task output for evaluators."""
    return {
        "response": sample.output,
        "retrieved_contexts": sample.contexts,
        "rerank_applied": sample.rerank_applied,
    }


def make_ragas_evaluator(metric: Metric) -> EvaluatorFunction:
    """Wrap a Ragas metric scorer as a Langfuse item-level evaluator."""

    async def evaluator(
        *,
        input: Any,
        output: Any,
        expected_output: Any = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Evaluation:
        if not isinstance(output, dict):
            return Evaluation(name=metric.name, value=0, comment="ERROR: Task output is missing retrieval payload")

        sample = experiment_item_to_sample(
            {
                "input": input,
                "expected_output": expected_output,
                "metadata": metadata,
            },
            output,
        )
        try:
            score = await metric.score(sample)
        except Exception as exc:
            logger.exception("Metric %s failed for input preview=%r", metric.name, str(input)[:80])
            return Evaluation(name=metric.name, value=0, comment=f"ERROR: {exc}")

        if score.error or score.value is None:
            comment = "Reference answer is required" if score.error == "reference is required" else score.error
            return Evaluation(name=metric.name, value=0, comment=f"ERROR: {comment or 'scoring failed'}")
        return Evaluation(name=metric.name, value=score.value)

    return evaluator


def build_langfuse_evaluators(metrics: list[Metric]) -> list[EvaluatorFunction]:
    """Create Langfuse evaluators for all configured Ragas metrics."""
    return [make_ragas_evaluator(metric) for metric in metrics]


def experiment_result_to_report(
    result: ExperimentResult,
    settings: RagEvalSettings,
    dataset_path: str | None = None,
    metrics: list[Metric] | None = None,
) -> EvaluationReport:
    """Convert a Langfuse experiment result into the unified report shape."""
    report_items: list[ReportItem] = []

    for item_result in result.item_results:
        metric_results: list[MetricResult] = []
        for evaluation in item_result.evaluations:
            is_error = bool(evaluation.comment and evaluation.comment.startswith("ERROR: "))
            is_numeric = isinstance(evaluation.value, (int, float))
            error = None
            if is_error:
                error = (evaluation.comment or "").removeprefix("ERROR: ")
            elif not is_numeric:
                error = evaluation.comment or "non-numeric evaluation"
            metric_results.append(
                MetricResult(
                    name=evaluation.name,
                    value=float(evaluation.value) if is_numeric and not is_error else None,
                    reasoning=evaluation.comment if not is_error else None,
                    error=error,
                )
            )
        sample = experiment_item_to_sample(item_result.item, item_result.output)
        report_items.append(
            make_report_item(
                sample,
                metric_results,
                item_id=item_result.trace_id,
                metadata={"trace_id": item_result.trace_id, "dataset_run_id": item_result.dataset_run_id},
            )
        )

    run_evaluations = [
        {
            "name": evaluation.name,
            "value": evaluation.value,
            "comment": evaluation.comment,
        }
        for evaluation in result.run_evaluations
    ]

    return build_report(
        run_type="langfuse_experiment",
        evaluation_model=settings.evaluation_llm,
        dataset_source="langfuse_dataset" if dataset_path is None else "json_rag",
        dataset_location=settings.rag_eval_langfuse_dataset or dataset_path,
        total_items=len(result.item_results),
        items=report_items,
        metrics=metrics,
        metadata={
            "experiment_name": result.name,
            "run_name": result.run_name,
            "description": result.description,
            "experiment_id": result.experiment_id,
            "dataset_run_id": result.dataset_run_id,
            "dataset_run_url": result.dataset_run_url,
            "run_evaluations": run_evaluations,
            "formatted_summary": result.format(),
        },
    )


class RagLangfuseExperiment:
    """Run retrieval + Ragas scoring through Langfuse's experiment runner."""

    def __init__(self, settings: RagEvalSettings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.langfuse_public_key or not self.settings.langfuse_secret_key:
            raise ValueError("Langfuse credentials are required. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env")

        self.langfuse = Langfuse(
            public_key=self.settings.langfuse_public_key,
            secret_key=self.settings.langfuse_secret_key,
            host=self.settings.langfuse_host,
            timeout=60,
        )
        self.openai_client = create_openai_client(self.settings)
        self.retrieval_client = RerankSearchClient(self.settings)
        self.dataset_builder = DatasetBuilder(
            settings=self.settings,
            retrieval_client=self.retrieval_client,
            openai_client=self.openai_client,
        )
        self.metrics = build_metrics(self.settings, self.openai_client)
        self.report: EvaluationReport | None = None

    def run(
        self,
        *,
        experiment_name: str | None = None,
        run_name: str | None = None,
        description: str | None = None,
        dataset_name: str | None = None,
        generate_report_file: bool = True,
    ) -> ExperimentResult:
        """Execute the experiment on a local JSON dataset or a hosted Langfuse dataset."""
        resolved_name = experiment_name or self.settings.rag_eval_experiment_name
        resolved_description = description or self.settings.rag_eval_experiment_description
        resolved_dataset_name = dataset_name or self.settings.rag_eval_langfuse_dataset
        evaluators = build_langfuse_evaluators(self.metrics)
        task = self._build_task()
        metadata = self._build_metadata()

        if resolved_dataset_name:
            logger.info("Running Langfuse experiment '%s' on dataset '%s'", resolved_name, resolved_dataset_name)
            dataset = self.langfuse.get_dataset(resolved_dataset_name)
            result = dataset.run_experiment(
                name=resolved_name,
                run_name=run_name,
                description=resolved_description,
                task=task,
                evaluators=evaluators,
                max_concurrency=self.settings.rag_eval_max_concurrency,
                metadata=metadata,
            )
            dataset_path = None
        else:
            queries = load_queries(self.settings.rag_eval_dataset_path)
            data = queries_to_experiment_items(queries)
            if not data:
                raise RuntimeError("No experiment items found. Check the dataset file.")

            logger.info(
                "Running Langfuse experiment '%s' on %d local items from %s",
                resolved_name,
                len(data),
                self.settings.rag_eval_dataset_path,
            )
            result = self.langfuse.run_experiment(
                name=resolved_name,
                run_name=run_name,
                description=resolved_description,
                data=data,
                task=task,
                evaluators=evaluators,
                max_concurrency=self.settings.rag_eval_max_concurrency,
                metadata=metadata,
            )
            dataset_path = str(self.settings.rag_eval_dataset_path)

        self.langfuse.flush()

        self.report = experiment_result_to_report(
            result,
            self.settings,
            dataset_path=dataset_path,
            metrics=self.metrics,
        )
        if generate_report_file:
            write_report(self.report, reports_dir=REPORTS_DIR)

        logger.info(
            "Langfuse experiment completed: name=%s items=%s dataset_run_url=%s",
            result.name,
            len(result.item_results),
            result.dataset_run_url,
        )
        return result

    def _build_task(self):
        dataset_builder = self.dataset_builder

        async def rag_task(*, item, **kwargs):
            base_sample = experiment_item_to_sample(item)
            sample = await dataset_builder.build_sample_from_inputs(
                user_input=base_sample.user_input,
                reference=base_sample.reference or "",
                response=base_sample.output,
            )
            if sample is None:
                raise RuntimeError(f"No retrieval sample built for query: {base_sample.user_input}")
            return sample_to_task_output(sample)

        return rag_task

    def _build_metadata(self) -> dict[str, str]:
        return {
            "evaluation_llm": self.settings.evaluation_llm,
            "rag_api_base_url": self.settings.rag_api_base_url,
            "top_k": str(self.settings.rag_eval_top_k),
            "use_hybrid": str(self.settings.rag_eval_use_hybrid).lower(),
            "rerank_enabled": str(self.settings.rag_eval_rerank_enabled).lower(),
            "generate_responses": str(self.settings.rag_eval_generate_responses).lower(),
        }
