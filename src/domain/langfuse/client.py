import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

from langfuse import Langfuse, propagate_attributes

from src.config import Settings

logger = logging.getLogger(__name__)


def _stringify_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Convert metadata values to strings for propagate_attributes."""
    if not metadata:
        return None
    return {key: str(value) for key, value in metadata.items()}


class LangfuseTracer:
    """Wrapper for Langfuse v4 tracing client with CallbackHandler support."""

    def __init__(self, settings: Settings):
        self.settings = settings.langfuse
        self.client: Optional[Langfuse] = None

        if self.settings.enabled and self.settings.public_key and self.settings.secret_key:
            try:
                self.client = Langfuse(
                    public_key=self.settings.public_key,
                    secret_key=self.settings.secret_key,
                    host=self.settings.host,
                    flush_at=self.settings.flush_at,
                    flush_interval=self.settings.flush_interval,
                    debug=self.settings.debug,
                )
                logger.info("Langfuse v4 tracing initialized (host: %s)", self.settings.host)
            except Exception as e:
                logger.error("Failed to initialize Langfuse: %s", e)
                self.client = None
        else:
            logger.info("Langfuse tracing disabled or missing credentials")

    @contextmanager
    def _propagate_attributes(
        self,
        *,
        trace_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        environment: Optional[str] = None,
    ):
        """Apply v4 correlating attributes to the current and child observations."""
        if not self.client:
            yield
            return

        attr_kwargs: Dict[str, Any] = {}
        if trace_name:
            attr_kwargs["trace_name"] = trace_name
        if user_id:
            attr_kwargs["user_id"] = user_id
        if session_id:
            attr_kwargs["session_id"] = session_id
        if tags:
            attr_kwargs["tags"] = tags
        if environment:
            attr_kwargs["environment"] = environment

        string_metadata = _stringify_metadata(metadata)
        if string_metadata:
            attr_kwargs["metadata"] = string_metadata

        if not attr_kwargs:
            yield
            return

        with propagate_attributes(**attr_kwargs):
            yield

    @contextmanager
    def trace_rag_request(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for tracing a standard RAG request.

        Yields:
            Root Langfuse observation for the request, or None if tracing is disabled.
        """
        if not self.client:
            yield None
            return

        with self._propagate_attributes(
            trace_name="rag_request",
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        ):
            with self.client.start_as_current_observation(
                as_type="span",
                name="rag_request",
                input={"query": query},
            ) as observation:
                yield observation

    @contextmanager
    def trace_agent_request(
        self,
        name: str,
        *,
        input_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        environment: Optional[str] = None,
    ):
        """
        Context manager for tracing an agent workflow with v4 attribute propagation.

        Yields:
            Root Langfuse observation for the request, or None if tracing is disabled.
        """
        if not self.client:
            yield None
            return

        with self._propagate_attributes(
            trace_name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags,
            environment=environment,
        ):
            with self.client.start_as_current_observation(
                as_type="span",
                name=name,
                input=input_data,
            ) as observation:
                yield observation

    def create_span(
        self,
        trace,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a child span within an active observation.

        Args:
            trace: Parent Langfuse observation from trace_rag_request or trace_agent_request
            name: Name of the span
            input_data: Input data for the span
            metadata: Additional metadata

        Returns:
            LangfuseSpan if successful, None otherwise
        """
        if not trace or not self.client:
            return None

        try:
            if hasattr(trace, "start_observation"):
                return trace.start_observation(
                    as_type="span",
                    name=name,
                    input=input_data,
                    metadata=metadata or {},
                )
            return self.client.start_observation(
                as_type="span",
                name=name,
                input=input_data,
                metadata=metadata or {},
            )
        except Exception as e:
            logger.error("Error creating span %s: %s", name, e)
            return None

    def end_span(
        self,
        span,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """End a span with optional final output and metadata."""
        if not span:
            return

        try:
            if output is not None or metadata is not None:
                self.update_span(span, output=output, metadata=metadata)
            span.end()
        except Exception as e:
            logger.error("Error ending span: %s", e)

    def get_callback_handler(
        self,
        trace_name: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ):
        """
        Get a CallbackHandler for LangChain/LangGraph integration.

        Correlating attributes must be set via propagate_attributes() on an enclosing
        context manager (trace_agent_request / trace_rag_request). The handler only
        needs to be created inside that scope.

        Returns:
            CallbackHandler instance if Langfuse is enabled, None otherwise
        """
        if not self.client:
            return None

        try:
            from langfuse.langchain import CallbackHandler

            return CallbackHandler()
        except Exception as e:
            logger.error("Error creating CallbackHandler: %s", e)
            return None

    @contextmanager
    def trace_langgraph_agent(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        environment: Optional[str] = None,
    ):
        """
        Context manager to wrap LangGraph agent execution with a top-level observation.

        Usage:
            with tracer.trace_langgraph_agent(name="agentic_rag", ...) as (trace_ctx, handler):
                result = graph.invoke(input, config={"callbacks": [handler]})
                trace_ctx.update(output=result)

        Yields:
            Tuple of (observation_context, callback_handler) for graph execution
        """
        if not self.client:
            yield (None, None)
            return

        with self._propagate_attributes(
            trace_name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags,
            environment=environment,
        ):
            handler = self.get_callback_handler()
            yield (None, handler)

    def get_trace_id(self, trace=None) -> Optional[str]:
        """
        Get the current trace ID from Langfuse context.

        Args:
            trace: Optional observation; falls back to the active trace context

        Returns:
            Trace ID string or None if trace is disabled
        """
        if not self.client:
            return None

        try:
            if trace is not None:
                trace_id = getattr(trace, "trace_id", None)
                if trace_id:
                    return trace_id
            return self.client.get_current_trace_id()
        except Exception as e:
            logger.error("Error getting trace ID: %s", e)
            return None

    def submit_feedback(
        self,
        trace_id: str,
        score: float,
        name: str = "user-feedback",
        comment: Optional[str] = None,
    ) -> bool:
        """
        Submit user feedback for a trace.

        Args:
            trace_id: Trace ID from get_trace_id()
            score: Feedback score (0-1 or -1 to 1)
            name: Name of the score (default: "user-feedback")
            comment: Optional feedback comment

        Returns:
            True if feedback was submitted successfully, False otherwise
        """
        if not self.client:
            logger.warning("Cannot submit feedback: Langfuse is disabled")
            return False

        try:
            self.client.create_score(
                trace_id=trace_id,
                name=name,
                value=score,
                comment=comment,
            )
            logger.info("Submitted feedback for trace %s: score=%s", trace_id, score)
            return True
        except Exception as e:
            logger.error("Error submitting feedback: %s", e)
            return False

    def flush(self):
        """Flush any pending traces."""
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error("Error flushing Langfuse: %s", e)

    def shutdown(self):
        """Shutdown the Langfuse client."""
        if self.client:
            try:
                self.client.flush()
                self.client.shutdown()
            except Exception as e:
                logger.error("Error shutting down Langfuse: %s", e)

    @contextmanager
    def start_generation(
        self,
        name: str,
        model: str,
        input_data: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Start a generation observation for LLM calls.

        Yields:
            Generation context object for updates
        """
        if not self.client:
            yield None
            return

        with self.client.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=input_data,
            metadata=metadata or {},
        ) as generation:
            yield generation

    @contextmanager
    def start_span(
        self,
        name: str,
        input_data: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Start a generic span for non-LLM operations.

        Yields:
            Span context object for updates
        """
        if not self.client:
            yield None
            return

        with self.client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_data,
            metadata=metadata or {},
        ) as span:
            yield span

    def update_generation(
        self,
        generation,
        output: Any,
        usage_metadata: Optional[Dict[str, Any]] = None,
        completion_start_time: Optional[float] = None,
    ):
        """Update a generation observation with output and usage metrics."""
        if not generation:
            return

        try:
            update_data: Dict[str, Any] = {"output": output}

            if usage_metadata:
                update_data["usage_details"] = {
                    "input": usage_metadata.get("prompt_tokens", 0),
                    "output": usage_metadata.get("completion_tokens", 0),
                    "total": usage_metadata.get("total_tokens", 0),
                }

                if "latency_ms" in usage_metadata:
                    update_data["metadata"] = {"latency_ms": usage_metadata["latency_ms"]}

            generation.update(**update_data)
            generation.end()
        except Exception as e:
            logger.error("Error updating generation: %s", e)

    def update_span(
        self,
        span,
        output: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: Optional[str] = None,
        status_message: Optional[str] = None,
    ):
        """Update a span with output and metadata."""
        if not span:
            return

        try:
            update_data: Dict[str, Any] = {}
            if output is not None:
                update_data["output"] = output
            if metadata:
                update_data["metadata"] = metadata
            if level:
                update_data["level"] = level
            if status_message:
                update_data["status_message"] = status_message

            if update_data:
                span.update(**update_data)
        except Exception as e:
            logger.error("Error updating span: %s", e)
