"""Tests for TrajectoryCallback and TrajectoryMiddleware."""

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import Generation, LLMResult

from src.platform.graph.callback_utils import extend_graph_callbacks
from src.platform.tracing import TrajectoryCallback
from src.platform.middleware.pipeline import AgentPipeline
from src.platform.middleware.trajectory_middleware import TrajectoryMiddleware
from src.platform.middleware.types import AgentContext


def _make_ctx() -> AgentContext:
    return AgentContext(
        messages=[HumanMessage(content="What are transformers?")],
        session_id="session-1",
        user_id=None,
        config={"graph_config": {"thread_id": "session-1"}},
        agent_name="fusionsearch",
        metadata={"query": "What are transformers?"},
    )


class TestTrajectoryCallback:
    @pytest.mark.asyncio
    async def test_records_chain_tool_and_llm_events(self):
        callback = TrajectoryCallback(max_content_length=100)
        chain_run_id = uuid4()
        tool_run_id = uuid4()
        llm_run_id = uuid4()

        await callback.on_chain_start(
            {"name": "retrieve"},
            {"messages": []},
            run_id=chain_run_id,
        )
        await callback.on_tool_start(
            {"name": "retrieve_papers"},
            '{"query": "transformers"}',
            run_id=tool_run_id,
            parent_run_id=chain_run_id,
            inputs={"query": "transformers"},
        )
        await callback.on_tool_end(
            "Found 3 papers",
            run_id=tool_run_id,
            parent_run_id=chain_run_id,
        )
        await callback.on_chat_model_start(
            {"name": "ChatOpenAI"},
            [[HumanMessage(content="grade this")]],
            run_id=llm_run_id,
            parent_run_id=chain_run_id,
        )
        await callback.on_llm_end(
            LLMResult(generations=[[Generation(text="yes")]]),
            run_id=llm_run_id,
            parent_run_id=chain_run_id,
        )
        await callback.on_chain_end(
            {"messages": []},
            run_id=chain_run_id,
        )

        trajectory = callback.finalize()

        assert trajectory.finished_at is not None
        assert [event.event_type for event in trajectory.events] == [
            "chain_start",
            "tool_start",
            "tool_end",
            "chat_model_start",
            "llm_end",
            "chain_end",
        ]
        assert trajectory.summary()["nodes"] == ["retrieve"]
        assert trajectory.summary()["tools"] == ["retrieve_papers"]
        assert trajectory.summary()["models"] == ["ChatOpenAI"]
        assert "tool:retrieve_papers" in trajectory.to_steps()[1]
        assert "llm:ChatOpenAI" in trajectory.to_steps()[2]

    @pytest.mark.asyncio
    async def test_records_errors(self):
        callback = TrajectoryCallback()
        run_id = uuid4()

        await callback.on_tool_start(
            {"name": "retrieve_papers"},
            "{}",
            run_id=run_id,
        )
        await callback.on_tool_error(RuntimeError("boom"), run_id=run_id)

        trajectory = callback.finalize()
        assert trajectory.summary()["errors"] == ["tool_error:tool: boom"]
        assert any("error:tool: boom" in step for step in trajectory.to_steps())

    @pytest.mark.asyncio
    async def test_handles_none_serialized_chain_start(self):
        callback = TrajectoryCallback()
        run_id = uuid4()

        await callback.on_chain_start(
            None,
            {"messages": []},
            run_id=run_id,
            metadata={"langgraph_node": "retrieve"},
        )

        trajectory = callback.finalize()
        assert trajectory.events[0].name == "retrieve"
        assert trajectory.to_steps() == ["node:retrieve"]

    @pytest.mark.asyncio
    async def test_truncates_large_payloads(self):
        callback = TrajectoryCallback(max_content_length=20)
        run_id = uuid4()

        await callback.on_tool_start(
            {"name": "retrieve_papers"},
            "x" * 100,
            run_id=run_id,
        )

        event = callback.trajectory.events[0]
        assert isinstance(event.input, str)
        assert event.input.endswith("...")
        assert len(event.input) <= 23

    @pytest.mark.asyncio
    async def test_sanitizes_langgraph_send_objects(self):
        from langgraph.types import Send

        from pydantic import TypeAdapter

        from src.platform.tracing.schemas import GraphTrajectoryResponse

        callback = TrajectoryCallback()
        run_id = uuid4()
        sends = [
            Send("documents", {"query": "What is a transformer?"}),
            Send("database", {"query": "How many transformer papers exist?"}),
        ]

        await callback.on_chain_end(sends, run_id=run_id)

        trajectory = callback.finalize()
        output = trajectory.events[0].output
        assert output == [
            {
                "type": "Send",
                "node": "documents",
                "arg": {"query": "What is a transformer?"},
            },
            {
                "type": "Send",
                "node": "database",
                "arg": {"query": "How many transformer papers exist?"},
            },
        ]

        TypeAdapter(GraphTrajectoryResponse).validate_python(trajectory.to_api_dict())


class TestTrajectoryMiddleware:
    @pytest.mark.asyncio
    async def test_attaches_callback_without_overwriting_existing_callbacks(self):
        middleware = TrajectoryMiddleware()
        ctx = _make_ctx()
        existing_handler = object()
        ctx.config["graph_config"]["callbacks"] = [existing_handler]

        await middleware.before_invoke(ctx)

        callbacks = ctx.config["graph_config"]["callbacks"]
        assert callbacks[0] is existing_handler
        assert isinstance(callbacks[1], TrajectoryCallback)

    @pytest.mark.asyncio
    async def test_stores_trajectory_after_invoke(self):
        async def invoke_fn(ctx: AgentContext):
            callback = ctx.config["graph_config"]["callbacks"][-1]
            run_id = uuid4()
            await callback.on_chain_start({"name": "generate_answer"}, {}, run_id=run_id)
            await callback.on_chain_end({"messages": []}, run_id=run_id)
            return [AIMessage(content="answer")]

        pipeline = AgentPipeline(
            middlewares=[TrajectoryMiddleware()],
            invoke_fn=invoke_fn,
        )
        ctx = _make_ctx()

        result = await pipeline.run(ctx)

        assert result[-1].content == "answer"
        trajectory = ctx.metadata["trajectory"]
        assert trajectory.summary()["nodes"] == ["generate_answer"]
        assert trajectory.to_steps() == ["node:generate_answer"]

    @pytest.mark.asyncio
    async def test_stores_trajectory_on_error(self):
        async def invoke_fn(_ctx: AgentContext):
            raise RuntimeError("graph failed")

        pipeline = AgentPipeline(
            middlewares=[TrajectoryMiddleware()],
            invoke_fn=invoke_fn,
        )
        ctx = _make_ctx()

        with pytest.raises(RuntimeError, match="graph failed"):
            await pipeline.run(ctx)

        trajectory = ctx.metadata["trajectory"]
        assert trajectory.summary()["errors"][-1] == "invoke_error:fusionsearch: graph failed"


class TestExtendGraphCallbacks:
    def test_appends_without_duplicates(self):
        ctx = _make_ctx()
        first = TrajectoryCallback()
        second = TrajectoryCallback()

        extend_graph_callbacks(ctx, [first])
        extend_graph_callbacks(ctx, [first, second])

        callbacks = ctx.config["graph_config"]["callbacks"]
        assert callbacks == [first, second]
