"""Helper functions for the evaluation process."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langfuse.api import TraceWithDetails

from rag_evals.schemas import ScoreSchema

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent / "reports"

INPUT_KEYS = ("query", "question", "input", "prompt", "content", "text")
OUTPUT_KEYS = ("answer", "response", "content", "text", "generation", "final_answer", "output")


def _stringify_value(value: Any) -> Optional[str]:
    """Convert trace payload values into a readable string."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, indent=2, default=str)
    if isinstance(value, list):
        parts = [part for part in (_stringify_value(item) for item in value) if part]
        return "\n".join(parts) if parts else None
    return str(value)


def _extract_from_mapping(payload: dict, keys: tuple[str, ...]) -> Optional[str]:
    """Return the first populated string value for the given keys."""
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        rendered = _stringify_value(value)
        if rendered:
            return rendered
    return None


def format_messages(messages: list[dict]) -> str:
    """Format a list of LangGraph/LangChain-style messages for evaluation."""
    formatted_messages = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            rendered = _stringify_value(message)
            if rendered:
                formatted_messages.append(rendered)
            continue

        message_type = message.get("type") or message.get("role") or "message"
        content = message.get("content")

        if message_type == "tool":
            previous_message = messages[idx - 1] if idx > 0 else {}
            tool_call = previous_message.get("additional_kwargs", {}).get("tool_calls", [])
            if tool_call:
                args = tool_call[0].get("function", {}).get("arguments")
            else:
                previous_tool_calls = previous_message.get("tool_calls") or []
                args = previous_tool_calls[0].get("args") if previous_tool_calls else {}
            tool_content = content or ""
            formatted_messages.append(
                f"tool {message.get('name')} input: {args} {tool_content[:100]}..."
                if len(tool_content) > 100
                else f"tool {message.get('name')}: {tool_content}"
            )
        elif content:
            formatted_messages.append(f"{message_type}: {content}")

    return "\n".join(formatted_messages)


def _extract_from_messages(messages: list[Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract input/output from a LangGraph-style messages list."""
    if not messages:
        return None, None

    if len(messages) == 1:
        only_message = messages[0]
        if isinstance(only_message, dict):
            content = only_message.get("content")
            if content:
                return None, str(content)
        return None, _stringify_value(only_message)

    input_messages = messages[:-1]
    output_message = messages[-1]
    input_text = format_messages([msg for msg in input_messages if isinstance(msg, dict)])
    if isinstance(output_message, dict):
        output_text = format_messages([output_message])
    else:
        output_text = _stringify_value(output_message)

    return input_text or None, output_text or None


def get_input_output(trace: TraceWithDetails) -> Tuple[Optional[str], Optional[str]]:
    """Extract and format input and output from a Langfuse trace.

    Supports:
    - RAG API traces with top-level input/output dicts (query + answer)
    - LangGraph callback traces that store messages under output.messages
    - Plain string payloads as a fallback
    """
    input_payload = trace.input
    output_payload = trace.output

    input_text = None
    output_text = None

    if isinstance(input_payload, dict):
        input_text = _extract_from_mapping(input_payload, INPUT_KEYS)
        if not input_text and input_payload.get("messages"):
            input_text, _ = _extract_from_messages(input_payload["messages"])
    else:
        input_text = _stringify_value(input_payload)

    if isinstance(output_payload, dict):
        if output_payload.get("error"):
            output_text = f"Error: {output_payload['error']}"
        else:
            output_text = _extract_from_mapping(output_payload, OUTPUT_KEYS)

        if not output_text and output_payload.get("messages"):
            _, output_text = _extract_from_messages(output_payload["messages"])
    else:
        output_text = _stringify_value(output_payload)

    if input_text and output_text:
        return input_text, output_text

    # Legacy layout: entire conversation stored under output.messages
    if isinstance(output_payload, dict) and output_payload.get("messages"):
        legacy_input, legacy_output = _extract_from_messages(output_payload["messages"])
        return legacy_input or input_text, legacy_output or output_text

    return input_text, output_text


def initialize_report(model_name: str) -> Dict[str, Any]:
    """Initialize report data structure."""
    return {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "total_traces": 0,
        "successful_traces": 0,
        "failed_traces": 0,
        "duration_seconds": 0,
        "metrics_summary": {},
        "successful_traces_details": [],
        "failed_traces_details": [],
    }


def initialize_metrics_summary(report: Dict[str, Any], metrics: List[Dict[str, str]]) -> None:
    """Initialize metrics summary in the report."""
    for metric in metrics:
        report["metrics_summary"][metric["name"]] = {"success_count": 0, "failure_count": 0, "avg_score": 0.0}


def update_success_metrics(
    report: Dict[str, Any], trace_id: str, metric_name: str, score: ScoreSchema, trace_results: Dict[str, Any]
) -> None:
    """Update metrics for a successful evaluation."""
    trace_results[trace_id]["metrics_succeeded"] += 1
    trace_results[trace_id]["metrics_results"][metric_name] = {
        "success": True,
        "score": score.score,
        "reasoning": score.reasoning,
    }
    report["metrics_summary"][metric_name]["success_count"] += 1
    report["metrics_summary"][metric_name]["avg_score"] += score.score


def update_failure_metrics(
    report: Dict[str, Any], trace_id: str, metric_name: str, trace_results: Dict[str, Any]
) -> None:
    """Update metrics for a failed evaluation."""
    trace_results[trace_id]["metrics_results"][metric_name] = {"success": False}
    report["metrics_summary"][metric_name]["failure_count"] += 1


def process_trace_results(
    report: Dict[str, Any], trace_id: str, trace_results: Dict[str, Any], metrics_count: int
) -> None:
    """Process results for a single trace."""
    if trace_results[trace_id]["metrics_succeeded"] == metrics_count:
        trace_results[trace_id]["success"] = True
        report["successful_traces"] += 1
        report["successful_traces_details"].append(
            {"trace_id": trace_id, "metrics_results": trace_results[trace_id]["metrics_results"]}
        )
    else:
        report["failed_traces"] += 1
        report["failed_traces_details"].append(
            {
                "trace_id": trace_id,
                "metrics_evaluated": trace_results[trace_id]["metrics_evaluated"],
                "metrics_succeeded": trace_results[trace_id]["metrics_succeeded"],
                "metrics_results": trace_results[trace_id]["metrics_results"],
            }
        )


def calculate_avg_scores(report: Dict[str, Any]) -> None:
    """Calculate average scores for each metric."""
    for _, data in report["metrics_summary"].items():
        if data["success_count"] > 0:
            data["avg_score"] = round(data["avg_score"] / data["success_count"], 2)


def generate_report(report: Dict[str, Any]) -> str:
    """Generate a JSON report file with evaluation results."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"evaluation_report_{timestamp}.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    report["generate_report_path"] = str(report_path)

    logger.info("Evaluation report generated at %s", report_path)
    return str(report_path)
