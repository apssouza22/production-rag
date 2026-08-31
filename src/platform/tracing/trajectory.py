"""Capture the full execution trajectory of a LangGraph invoke."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langgraph.types import Send


def _sanitize_for_json(value: Any) -> Any:
    """Convert callback payloads into JSON/Pydantic-serializable structures."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, Send):
        payload: dict[str, Any] = {
            "type": "Send",
            "node": value.node,
            "arg": _sanitize_for_json(value.arg),
        }
        if value.timeout is not None:
            payload["timeout"] = value.timeout
        return payload

    if isinstance(value, BaseMessage):
        message_payload: dict[str, Any] = {
            "type": getattr(value, "type", value.__class__.__name__),
            "content": getattr(value, "content", str(value)),
        }
        tool_calls = getattr(value, "tool_calls", None)
        if tool_calls:
            message_payload["tool_calls"] = _sanitize_for_json(tool_calls)
        return message_payload

    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(item) for item in value]

    return str(value)


def _truncate(value: Any, max_length: int) -> Any:
    if value is None:
        return None

    sanitized = _sanitize_for_json(value)

    if max_length <= 0:
        return sanitized

    if isinstance(sanitized, str):
        if len(sanitized) <= max_length:
            return sanitized
        return f"{sanitized[:max_length]}..."

    rendered = json.dumps(sanitized, default=str)
    if len(rendered) <= max_length:
        return sanitized
    return f"{rendered[:max_length]}..."


def _serialize_messages(messages: list[list[BaseMessage]]) -> list[list[dict[str, Any]]]:
    serialized_batches: list[list[dict[str, Any]]] = []
    for batch in messages:
        serialized_batch: list[dict[str, Any]] = []
        for message in batch:
            payload: dict[str, Any] = {
                "type": getattr(message, "type", message.__class__.__name__),
                "content": getattr(message, "content", str(message)),
            }
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                payload["tool_calls"] = tool_calls
            serialized_batch.append(payload)
        serialized_batches.append(serialized_batch)
    return serialized_batches


def _extract_run_name(
    serialized: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> str:
    if metadata:
        for key in ("langgraph_node", "langgraph_triggers", "checkpoint_ns"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, list) and value:
                return str(value[-1])

    if not serialized:
        return "unknown"

    name = serialized.get("name")
    if isinstance(name, str) and name:
        return name

    run_id = serialized.get("id")
    if isinstance(run_id, list) and run_id:
        return str(run_id[-1])
    if isinstance(run_id, str) and run_id:
        return run_id

    return serialized.get("type", "unknown")


def _extract_llm_output(response: LLMResult) -> Any:
    if not response.generations:
        return None

    generation = response.generations[0][0]
    message = getattr(generation, "message", None)
    if message is not None:
        payload: dict[str, Any] = {
            "type": getattr(message, "type", message.__class__.__name__),
            "content": getattr(message, "content", ""),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            payload["tool_calls"] = tool_calls
        return payload

    text = getattr(generation, "text", None)
    if text is not None:
        return text

    return str(generation)


@dataclass
class TrajectoryEvent:
    """A single event in a graph execution trajectory."""

    event_type: str
    name: str
    run_id: str
    parent_run_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    duration_ms: Optional[float] = None
    input: Any = None
    output: Any = None
    error: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "name": self.name,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "input": self.input,
            "output": self.output,
            "error": self.error,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class GraphTrajectory:
    """Structured trajectory for one graph invoke."""

    events: list[TrajectoryEvent] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "events": [event.to_dict() for event in self.events],
            "summary": self.summary(),
            "steps": self.to_steps(),
        }

    def to_api_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable trajectory payload for API responses."""
        return self.to_dict()

    def summary(self) -> dict[str, Any]:
        node_names: list[str] = []
        tool_names: list[str] = []
        model_names: list[str] = []
        errors: list[str] = []

        for event in self.events:
            if event.error:
                errors.append(f"{event.event_type}:{event.name}: {event.error}")
            if event.event_type == "chain_start":
                node_names.append(event.name)
            elif event.event_type == "tool_start":
                tool_names.append(event.name)
            elif event.event_type in {"chat_model_start", "llm_start"}:
                model_names.append(event.name)

        return {
            "event_count": len(self.events),
            "nodes": node_names,
            "tools": tool_names,
            "models": model_names,
            "errors": errors,
        }

    def to_steps(self) -> list[str]:
        """Return a human-readable step list suitable for API responses."""
        steps: list[str] = []

        for event in self.events:
            if event.event_type == "chain_start":
                steps.append(f"node:{event.name}")
            elif event.event_type == "tool_start":
                tool_input = event.input if isinstance(event.input, dict) else {"input": event.input}
                steps.append(f"tool:{event.name}({json.dumps(tool_input, default=str)})")
            elif event.event_type in {"chat_model_start", "llm_start"}:
                steps.append(f"llm:{event.name}")
            elif event.event_type.endswith("_error"):
                steps.append(f"error:{event.name}: {event.error}")

        return steps


class TrajectoryCallback(AsyncCallbackHandler):
    """Collect the full callback event stream for a LangGraph invoke.

    Attach to ``graph.ainvoke(..., config={"callbacks": [TrajectoryCallback()]})``
    or use ``TrajectoryMiddleware`` to wire it automatically.
  """

    def __init__(self, *, max_content_length: int = 500) -> None:
        self.max_content_length = max_content_length
        self.trajectory = GraphTrajectory()
        self._start_times: dict[str, float] = {}

    def _record_start(
        self,
        *,
        event_type: str,
        name: str,
        run_id: UUID,
        parent_run_id: UUID | None,
        input_data: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        run_id_str = str(run_id)
        self._start_times[run_id_str] = time.time()
        self.trajectory.events.append(
            TrajectoryEvent(
                event_type=event_type,
                name=name,
                run_id=run_id_str,
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                input=_truncate(input_data, self.max_content_length),
                tags=list(tags or []),
                metadata=dict(metadata or {}),
            )
        )

    def _record_end(
        self,
        *,
        event_type: str,
        name: str,
        run_id: UUID,
        parent_run_id: UUID | None,
        output_data: Any = None,
        error: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        run_id_str = str(run_id)
        started = self._start_times.pop(run_id_str, None)
        duration_ms = None
        if started is not None:
            duration_ms = (time.time() - started) * 1000

        self.trajectory.events.append(
            TrajectoryEvent(
                event_type=event_type,
                name=name,
                run_id=run_id_str,
                parent_run_id=str(parent_run_id) if parent_run_id else None,
                output=_truncate(output_data, self.max_content_length),
                error=error,
                duration_ms=duration_ms,
                tags=list(tags or []),
                metadata=dict(metadata or {}),
            )
        )

    def finalize(self) -> GraphTrajectory:
        """Mark the trajectory complete and return it."""
        self.trajectory.finished_at = time.time()
        return self.trajectory

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_start(
            event_type="chain_start",
            name=_extract_run_name(serialized, metadata),
            run_id=run_id,
            parent_run_id=parent_run_id,
            input_data=inputs,
            tags=tags,
            metadata=metadata,
        )

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_end(
            event_type="chain_end",
            name="chain",
            run_id=run_id,
            parent_run_id=parent_run_id,
            output_data=outputs,
            tags=tags,
            metadata=metadata,
        )

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_end(
            event_type="chain_error",
            name="chain",
            run_id=run_id,
            parent_run_id=parent_run_id,
            error=str(error),
            tags=tags,
            metadata=metadata,
        )

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_start(
            event_type="chat_model_start",
            name=_extract_run_name(serialized, metadata),
            run_id=run_id,
            parent_run_id=parent_run_id,
            input_data=_serialize_messages(messages),
            tags=tags,
            metadata=metadata,
        )

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_start(
            event_type="llm_start",
            name=_extract_run_name(serialized, metadata),
            run_id=run_id,
            parent_run_id=parent_run_id,
            input_data=prompts,
            tags=tags,
            metadata=metadata,
        )

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_end(
            event_type="llm_end",
            name="llm",
            run_id=run_id,
            parent_run_id=parent_run_id,
            output_data=_extract_llm_output(response),
            tags=tags,
            metadata=metadata,
        )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_end(
            event_type="llm_error",
            name="llm",
            run_id=run_id,
            parent_run_id=parent_run_id,
            error=str(error),
            tags=tags,
            metadata=metadata,
        )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_start(
            event_type="tool_start",
            name=_extract_run_name(serialized, metadata),
            run_id=run_id,
            parent_run_id=parent_run_id,
            input_data=inputs if inputs is not None else input_str,
            tags=tags,
            metadata=metadata,
        )

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_end(
            event_type="tool_end",
            name="tool",
            run_id=run_id,
            parent_run_id=parent_run_id,
            output_data=output,
            tags=tags,
            metadata=metadata,
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._record_end(
            event_type="tool_error",
            name="tool",
            run_id=run_id,
            parent_run_id=parent_run_id,
            error=str(error),
            tags=tags,
            metadata=metadata,
        )
