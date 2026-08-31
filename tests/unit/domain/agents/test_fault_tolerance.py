"""Tests for LangGraph fault-tolerance policies and error handlers."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.platform.graph import END, GraphBuilder, NodeError, START
from src.platform.graph.config import FaultToleranceConfig
from src.platform.graph.policies import (
    build_llm_timeout,
    build_retry_policy,
    is_transient_error,
)
from src.agents.fusionsearch.handlers import route_agentic_rag_failure
from src.agents.knowledgerouter.handlers import knowledge_router_error_handler
from src.agents.texttosql.handlers import text_to_sql_error_handler
from src.agents.texttosql.config import TextToSQLConfig
from src.agents.fusionsearch.context import Context
from src.agents.fusionsearch.config import GraphConfig
from src.agents.fusionsearch.graph import AgenticRAGGraph
from src.agents.fusionsearch.state import AgentState
from src.agents.knowledgerouter.state import RouterState
from src.platform.llm.exceptions import LLMConnectionError, LLMTimeoutError


class TestRetryClassification:
    def test_transient_errors_are_retryable(self):
        assert is_transient_error(ConnectionError("down")) is True
        assert is_transient_error(LLMConnectionError("down")) is True
        assert is_transient_error(LLMTimeoutError("slow")) is True

    def test_programming_errors_are_not_retryable(self):
        assert is_transient_error(ValueError("bad input")) is False
        assert is_transient_error(TypeError("wrong type")) is False


class TestPolicyBuilders:
    def test_build_retry_policy_uses_config(self):
        config = FaultToleranceConfig(max_attempts=5, initial_interval=1.0)
        policy = build_retry_policy(config)

        assert policy.max_attempts == 5
        assert policy.initial_interval == 1.0

    def test_build_llm_timeout_uses_config(self):
        config = FaultToleranceConfig(llm_run_timeout=90.0, llm_idle_timeout=10.0)
        timeout = build_llm_timeout(config)

        assert timeout.run_timeout == 90.0
        assert timeout.idle_timeout == 10.0


class TestAgenticRagFailureFlow:
    @pytest.mark.asyncio
    async def test_retry_then_route_to_handle_failure(self):
        graph = AgenticRAGGraph(
            llm_client=MagicMock(),
            opensearch_client=MagicMock(),
            embeddings_client=MagicMock(),
            retrieval_settings=MagicMock(),
            config=GraphConfig(),
        )

        async def failing_retrieve(state, runtime=None):
            raise ConnectionError("llm unavailable")

        workflow = GraphBuilder(AgentState, context_schema=Context)
        ft = FaultToleranceConfig(max_attempts=2, initial_interval=0.01)

        (
            workflow
            .set_node_defaults(
                retry_policy=build_retry_policy(ft),
                error_handler=route_agentic_rag_failure,
            )
            .add_node("retrieve", failing_retrieve)
            .add_node(
                "handle_failure",
                graph.handle_failure,
                retry_policy=None,
                error_handler=None,
            )
            .add_edge(START, "retrieve")
            .add_edge("handle_failure", END)
        )

        result = await workflow.compile().ainvoke(
            {
                "messages": [HumanMessage(content="test")],
                "retrieval_attempts": 0,
                "guardrail_result": None,
                "routing_decision": None,
                "sources": None,
                "relevant_sources": [],
                "relevant_tool_artefacts": None,
                "grading_results": [],
                "metadata": {},
                "original_query": None,
                "rewritten_query": None,
            },
            context=Context(trace_id=None),
        )

        assert result["metadata"]["fault_tolerance"]["failed_node"] == "retrieve"
        assert "temporary issue" in result["messages"][-1].content.lower()


class TestKnowledgeRouterHandlers:
    @pytest.mark.asyncio
    async def test_classify_failure_defaults_to_documents(self):
        state: RouterState = {
            "query": "papers on transformers",
            "classifications": [],
            "results": [],
            "final_answer": "",
        }
        command = await knowledge_router_error_handler(
            state, NodeError(node="classify", error=ConnectionError("down"))
        )

        assert command.goto == "documents"
        assert command.update["classifications"][0]["source"] == "documents"

    @pytest.mark.asyncio
    async def test_agent_failure_returns_partial_result(self):
        state: RouterState = {
            "query": "count papers",
            "classifications": [{"source": "database", "query": "count papers"}],
            "results": [],
            "final_answer": "",
        }
        command = await knowledge_router_error_handler(
            state, NodeError(node="database", error=ConnectionError("down"))
        )

        assert command.goto == "synthesize"
        assert command.update["results"][0]["source"] == "database"
        assert "temporarily unavailable" in command.update["results"][0]["result"]


class TestTextToSqlHandler:
    @pytest.mark.asyncio
    async def test_returns_user_facing_message(self):
        command = await text_to_sql_error_handler(
            {"messages": [HumanMessage(content="how many papers?")]},
            NodeError(node="run_query", error=ConnectionError("db down")),
        )

        assert command.goto == END
        assert "unable to query the database" in command.update["messages"][0].content.lower()


class TestTextToSqlGraphCompilation:
    def test_graph_compiles_with_fault_tolerance_enabled(self):
        from unittest.mock import AsyncMock, MagicMock

        from langchain.tools import tool

        from src.agents.texttosql.graph import build_text_to_sql_graph

        @tool("sql_db_list_tables")
        async def sql_db_list_tables() -> str:
            """List available SQL tables."""
            return "papers"

        @tool("sql_db_schema")
        async def sql_db_schema(table_names: str) -> str:
            """Return schema for the requested tables."""
            return "schema"

        @tool("sql_db_query")
        async def sql_db_query(query: str) -> str:
            """Execute a read-only SQL query."""
            return "result"

        model = MagicMock()
        model.bind_tools.return_value.ainvoke = AsyncMock(
            return_value=AIMessage(content="ok"),
        )

        graph = build_text_to_sql_graph(
            model=model,
            tools=[sql_db_list_tables, sql_db_schema, sql_db_query],
            config=TextToSQLConfig(),
        )

        assert graph is not None


class TestAgenticRagGraphCompilation:
    def test_service_compiles_with_fault_tolerance_enabled(self):
        from unittest.mock import MagicMock

        from src.agents.fusionsearch.agentic_rag import AgenticRAGService
        from src.agents.fusionsearch.config import GraphConfig
        from src.agents.fusionsearch.factory import make_agentic_rag_graph

        config = GraphConfig()
        agentic_rag_graph, retrieval_settings = make_agentic_rag_graph(
            llm_client=MagicMock(),
            opensearch_client=MagicMock(),
            embeddings_client=MagicMock(),
            graph_config=config,
        )
        service = AgenticRAGService(
            llm_client=MagicMock(),
            graph_builder=agentic_rag_graph,
            retrieval_settings=retrieval_settings,
            graph_config=config,
        )

        assert service.graph is not None
