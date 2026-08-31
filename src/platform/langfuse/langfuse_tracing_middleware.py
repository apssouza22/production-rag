"""Middleware that wraps agent invocations with Langfuse tracing."""

import logging
import time
from contextlib import ExitStack
from typing import Any, Callable, Optional

from src.platform.langfuse.client import LangfuseTracer

from src.platform.middleware.types import AgentContext, AgentMiddleware, InvokeResult

logger = logging.getLogger(__name__)

TraceInputBuilder = Callable[[AgentContext], dict[str, Any]]
TraceMetadataBuilder = Callable[[AgentContext], dict[str, Any]]
TraceOutputBuilder = Callable[[AgentContext, InvokeResult], dict[str, Any]]


class LangfuseTracingMiddleware(AgentMiddleware):
    """Opens a Langfuse trace before invoke and finalizes it after invoke or on error.

    Register this middleware first so ``propagate_attributes`` wraps downstream
    middleware (e.g. guardrail spans) and LangGraph callback observations.
    """

    def __init__(
        self,
        langfuse_tracer: Optional[LangfuseTracer],
        trace_name: str,
        *,
        environment: Optional[str] = None,
        build_trace_input: Optional[TraceInputBuilder] = None,
        build_trace_metadata: Optional[TraceMetadataBuilder] = None,
        build_trace_output: Optional[TraceOutputBuilder] = None,
    ) -> None:
        self.langfuse_tracer = langfuse_tracer
        self.trace_name = trace_name
        self.environment = environment
        self.build_trace_input = build_trace_input or self._default_trace_input
        self.build_trace_metadata = build_trace_metadata or self._default_trace_metadata
        self.build_trace_output = build_trace_output

    @staticmethod
    def _default_trace_input(ctx: AgentContext) -> dict[str, Any]:
        query = ctx.metadata.get("query")
        if query:
            return {"query": query}
        if ctx.messages:
            last = ctx.messages[-1]
            content = getattr(last, "content", str(last))
            return {"query": content}
        return {}

    @staticmethod
    def _default_trace_metadata(ctx: AgentContext) -> dict[str, Any]:
        return dict(ctx.metadata.get("trace_metadata", {}))

    def _tracing_enabled(self) -> bool:
        return self.langfuse_tracer is not None and self.langfuse_tracer.client is not None

    async def before_invoke(self, ctx: AgentContext) -> Optional[InvokeResult]:
        if not self._tracing_enabled():
            return None

        ctx.metadata["_trace_start_time"] = time.time()
        stack = ExitStack()
        ctx.metadata["_langfuse_exit_stack"] = stack

        user_id = str(ctx.metadata.get("user_id", "unknown"))
        trace_metadata = self.build_trace_metadata(ctx)

        try:
            cm = self.langfuse_tracer.trace_agent_request(
                name=self.trace_name,
                input_data=self.build_trace_input(ctx),
                user_id=user_id,
                session_id=ctx.session_id,
                environment=self.environment,
                metadata=trace_metadata,
            )
            trace = stack.enter_context(cm)
            ctx.metadata["trace"] = trace

            handler = self.langfuse_tracer.get_callback_handler()
            if handler:
                graph_config = ctx.config.setdefault("graph_config", {})
                callbacks = graph_config.setdefault("callbacks", [])
                if handler not in callbacks:
                    callbacks.append(handler)
                logger.debug("Langfuse CallbackHandler attached for %s", ctx.agent_name)
        except Exception as exc:
            logger.warning("Failed to start Langfuse trace for %s: %s", ctx.agent_name, exc)
            stack.close()
            ctx.metadata.pop("_langfuse_exit_stack", None)

        return None

    async def after_invoke(self, ctx: AgentContext, result: InvokeResult) -> InvokeResult:
        trace = ctx.metadata.get("trace")
        if trace and self._tracing_enabled():
            try:
                output = self._build_output(ctx, result)
                trace.update(output=output)
                ctx.metadata["trace_id"] = (
                    getattr(trace, "trace_id", None) or self.langfuse_tracer.get_trace_id()
                )
                self.langfuse_tracer.flush()
            except Exception as exc:
                logger.warning("Failed to finalize Langfuse trace for %s: %s", ctx.agent_name, exc)

        self._close_trace_stack(ctx)
        return result

    async def on_error(self, ctx: AgentContext, error: Exception) -> Optional[InvokeResult]:
        trace = ctx.metadata.get("trace")
        if trace and self._tracing_enabled():
            try:
                trace.update(output={"error": str(error)}, level="ERROR")
                ctx.metadata["trace_id"] = (
                    getattr(trace, "trace_id", None) or self.langfuse_tracer.get_trace_id()
                )
                self.langfuse_tracer.flush()
            except Exception as exc:
                logger.warning("Failed to record Langfuse error for %s: %s", ctx.agent_name, exc)

        self._close_trace_stack(ctx)
        return None

    def _build_output(self, ctx: AgentContext, result: InvokeResult) -> dict[str, Any]:
        if self.build_trace_output:
            return self.build_trace_output(ctx, result)

        output: dict[str, Any] = {"execution_time": time.time() - ctx.metadata.get("_trace_start_time", time.time())}
        if result:
            last = result[-1]
            content = getattr(last, "content", str(last))
            output["answer"] = content
        return output

    def _close_trace_stack(self, ctx: AgentContext) -> None:
        stack: Optional[ExitStack] = ctx.metadata.pop("_langfuse_exit_stack", None)
        if stack:
            stack.close()
