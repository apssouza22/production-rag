"""Tests for LangfuseTracingMiddleware."""

from contextlib import contextmanager
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.domain.langfuse.langfuse_tracing_middleware import LangfuseTracingMiddleware
from src.domain.middleware.pipeline import AgentPipeline
from src.domain.middleware.types import AgentContext


@pytest.fixture
def mock_tracer():
    tracer = Mock()
    tracer.client = Mock()
    trace_observation = Mock()
    trace_observation.trace_id = "trace-123"

    @contextmanager
    def fake_trace_agent_request(**_kwargs):
        yield trace_observation

    tracer.trace_agent_request = fake_trace_agent_request
    tracer.get_callback_handler = Mock(return_value=Mock())
    tracer.get_trace_id = Mock(return_value="trace-123")
    tracer.flush = Mock()
    return tracer


def _make_ctx(query: str = "What are transformers?") -> AgentContext:
    return AgentContext(
        messages=[HumanMessage(content=query)],
        session_id="session_user_1",
        user_id=None,
        config={"model": "gpt-4o-mini", "graph_config": {"thread_id": "session_user_1"}},
        agent_name="fusionsearch",
        metadata={"query": query, "user_id": "user_1", "trace_metadata": {"service": "test"}},
    )


class TestLangfuseTracingMiddleware:
    @pytest.mark.asyncio
    async def test_before_invoke_attaches_trace_and_callbacks(self, mock_tracer):
        middleware = LangfuseTracingMiddleware(
            langfuse_tracer=mock_tracer,
            trace_name="test_request",
            environment="test",
        )
        ctx = _make_ctx()

        result = await middleware.before_invoke(ctx)

        assert result is None
        assert "trace" in ctx.metadata
        assert "callbacks" in ctx.config["graph_config"]
        assert ctx.metadata["_trace_start_time"] is not None

    @pytest.mark.asyncio
    async def test_after_invoke_updates_trace_and_sets_trace_id(self, mock_tracer):
        middleware = LangfuseTracingMiddleware(
            langfuse_tracer=mock_tracer,
            trace_name="test_request",
            build_trace_output=lambda ctx, result: {"answer": result[-1].content},
        )
        ctx = _make_ctx()
        await middleware.before_invoke(ctx)

        trace = ctx.metadata["trace"]
        pipeline_result = [AIMessage(content="Test answer")]
        result = await middleware.after_invoke(ctx, pipeline_result)

        assert result == pipeline_result
        trace.update.assert_called_once_with(output={"answer": "Test answer"})
        mock_tracer.flush.assert_called_once()
        assert ctx.metadata["trace_id"] == "trace-123"
        assert "_langfuse_exit_stack" not in ctx.metadata

    @pytest.mark.asyncio
    async def test_on_error_records_error_and_closes_stack(self, mock_tracer):
        middleware = LangfuseTracingMiddleware(
            langfuse_tracer=mock_tracer,
            trace_name="test_request",
        )
        ctx = _make_ctx()
        await middleware.before_invoke(ctx)

        trace = ctx.metadata["trace"]
        result = await middleware.on_error(ctx, RuntimeError("boom"))

        assert result is None
        trace.update.assert_called_once_with(output={"error": "boom"}, level="ERROR")
        mock_tracer.flush.assert_called_once()
        assert "_langfuse_exit_stack" not in ctx.metadata

    @pytest.mark.asyncio
    async def test_disabled_when_tracer_is_none(self):
        middleware = LangfuseTracingMiddleware(langfuse_tracer=None, trace_name="test_request")
        ctx = _make_ctx()

        await middleware.before_invoke(ctx)

        assert "trace" not in ctx.metadata
        assert "callbacks" not in ctx.config.get("graph_config", {})

    @pytest.mark.asyncio
    async def test_pipeline_integration(self, mock_tracer):
        async def invoke_fn(ctx: AgentContext):
            ctx.metadata["graph_result"] = {"messages": [AIMessage(content="from graph")]}
            return [AIMessage(content="from graph")]

        pipeline = AgentPipeline(
            middlewares=[
                LangfuseTracingMiddleware(
                    langfuse_tracer=mock_tracer,
                    trace_name="pipeline_test",
                    build_trace_output=lambda ctx, result: {"answer": result[-1].content},
                )
            ],
            invoke_fn=invoke_fn,
        )
        ctx = _make_ctx()

        result = await pipeline.run(ctx)

        assert result[-1].content == "from graph"
        assert ctx.metadata["trace_id"] == "trace-123"
        mock_tracer.flush.assert_called_once()
