"""Helper functions for the evaluation process."""

import json
from typing import Any, Optional, Tuple

from langfuse.api import TraceWithDetails

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
