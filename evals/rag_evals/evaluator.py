"""Evaluator for Langfuse traces."""

import logging
import time
from datetime import datetime, timedelta
from time import sleep

import openai
from langfuse import Langfuse
from langfuse.api import TraceWithDetails
from tqdm import tqdm

from rag_evals.config import build_openai_client_kwargs, get_settings
from rag_evals.helpers import (
    calculate_avg_scores,
    generate_report,
    get_input_output,
    initialize_metrics_summary,
    initialize_report,
    process_trace_results,
    update_failure_metrics,
    update_success_metrics,
)
from rag_evals.metrics import metrics
from rag_evals.schemas import ScoreSchema

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates model outputs using predefined metrics."""

    def __init__(self):
        """Initialize Evaluator with OpenAI and Langfuse clients."""
        self.settings = get_settings()

        use_bifrost = self.settings.bifrost_enabled or self.settings.llm_provider == "bifrost"
        if use_bifrost:
            client_kwargs = build_openai_client_kwargs(self.settings)
        else:
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
        self.report = initialize_report(self.settings.evaluation_llm)
        initialize_metrics_summary(self.report, metrics)

    async def run(self, generate_report_file=True):
        """Fetch traces from Langfuse, evaluate them, and push scores back."""
        start_time = time.time()
        traces = self.__fetch_traces()
        self.report["total_traces"] = len(traces)

        trace_results = {}

        for trace in tqdm(traces, desc="Evaluating traces"):
            trace_id = trace.id
            trace_results[trace_id] = {
                "success": False,
                "metrics_evaluated": 0,
                "metrics_succeeded": 0,
                "metrics_results": {},
            }

            for metric in tqdm(metrics, desc=f"Applying metrics to trace {trace_id[:8]}...", leave=False):
                metric_name = metric["name"]
                input_text, output_text = get_input_output(trace)
                score = await self._run_metric_evaluation(metric, input_text, output_text)

                if score:
                    self._push_to_langfuse(trace, score, metric)
                    update_success_metrics(self.report, trace_id, metric_name, score, trace_results)
                else:
                    update_failure_metrics(self.report, trace_id, metric_name, trace_results)

                trace_results[trace_id]["metrics_evaluated"] += 1

            process_trace_results(self.report, trace_id, trace_results, len(metrics))
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

    def _push_to_langfuse(self, trace: TraceWithDetails, score: ScoreSchema, metric: dict):
        """Push evaluation score to Langfuse."""
        self.langfuse.create_score(
            trace_id=trace.id,
            name=metric["name"],
            data_type="NUMERIC",
            value=score.score,
            comment=score.reasoning,
        )

    async def _run_metric_evaluation(self, metric: dict, input_text: str, output_text: str) -> ScoreSchema | None:
        """Evaluate a single trace against a specific metric."""
        metric_name = metric["name"]
        if not metric:
            logger.error("Metric %s not found", metric_name)
            return None
        system_metric_prompt = metric["prompt"]

        if not input_text or not output_text:
            logger.error("Metric %s evaluation failed: missing input or output", metric_name)
            return None
        score = await self._call_openai(system_metric_prompt, input_text, output_text)
        if score:
            logger.info("Metric %s evaluation completed successfully", metric_name)
        else:
            logger.error("Metric %s evaluation failed", metric_name)
        return score

    async def _call_openai(self, metric_system_prompt: str, input_text: str, output_text: str) -> ScoreSchema | None:
        """Call OpenAI API to evaluate a trace."""
        num_retries = 3
        for _ in range(num_retries):
            try:
                response = await self.client.beta.chat.completions.parse(
                    model=self.settings.evaluation_llm,
                    messages=[
                        {"role": "system", "content": metric_system_prompt},
                        {"role": "user", "content": f"Input: {input_text}\nGeneration: {output_text}"},
                    ],
                    response_format=ScoreSchema,
                )
                return response.choices[0].message.parsed
            except Exception as e:
                sleep_time = 10
                logger.error("Error calling OpenAI: %s (retrying in %ss)", e, sleep_time)
                sleep(sleep_time)
                continue
        return None

    def __fetch_traces(self) -> list[TraceWithDetails]:
        """Fetch traces from the past 24 hours without scores."""
        last_24_hours = datetime.now() - timedelta(hours=24)
        logger.info("Fetching Langfuse traces from %s", last_24_hours)
        try:
            traces = self.langfuse.api.trace.list(
                from_timestamp=last_24_hours, order_by="timestamp.asc", limit=100
            ).data
            return [trace for trace in traces if not trace.scores]
        except Exception as e:
            logger.error("Error fetching traces: %s", e)
            return []
