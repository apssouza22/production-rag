"""Tests for TextToSQLGraph routing and structured SQL generation."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.texttosql.config import TextToSQLConfig
from src.agents.texttosql.graph import TextToSQLGraph
from src.agents.texttosql.models import SqlCheckOutput, SqlGenerationOutput
from src.platform.graph import END


def _make_graph(max_query_loops: int = 5) -> TextToSQLGraph:
    tools = [
        MagicMock(name="sql_db_list_tables"),
        MagicMock(name="sql_db_schema"),
        MagicMock(name="sql_db_query"),
    ]
    tools[0].name = "sql_db_list_tables"
    tools[1].name = "sql_db_schema"
    tools[2].name = "sql_db_query"
    return TextToSQLGraph(
        model=MagicMock(),
        tools=tools,
        config=TextToSQLConfig(max_query_loops=max_query_loops, ontology_prompt=""),
    )


class TestShouldContinue:
    def test_ends_when_no_tool_calls(self):
        graph = _make_graph()
        state = {
            "messages": [HumanMessage(content="q"), AIMessage(content="done")],
            "query_loop_count": 0,
        }
        assert graph.should_continue(state) == END

    def test_continues_when_tool_calls_and_under_limit(self):
        graph = _make_graph(max_query_loops=5)
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "sql_db_query",
                            "args": {"query": "SELECT 1"},
                            "id": "1",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            "query_loop_count": 4,
        }
        assert graph.should_continue(state) == "check_query"

    def test_ends_when_loop_count_reaches_max(self):
        graph = _make_graph(max_query_loops=5)
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "sql_db_query",
                            "args": {"query": "SELECT 1"},
                            "id": "1",
                            "type": "tool_call",
                        }
                    ],
                )
            ],
            "query_loop_count": 5,
        }
        assert graph.should_continue(state) == END


class TestStructuredQueryGeneration:
    @pytest.mark.asyncio
    async def test_generate_query_emits_tool_call_with_thought(self):
        graph = _make_graph()
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=SqlGenerationOutput(thought="Count films", query="SELECT COUNT(*) FROM film", final_answer=None)
        )
        graph._generate_query_llm = llm
        state: dict = {"messages": [HumanMessage(content="How many films?")], "query_loop_count": 0}
        update = await graph.generate_query(state)
        message = update["messages"][0]
        assert message.tool_calls[0]["args"]["query"] == "SELECT COUNT(*) FROM film"
        assert "thought" not in message.tool_calls[0]["args"]
        assert message.content == "Count films"

    @pytest.mark.asyncio
    async def test_generate_query_finishes_with_final_answer(self):
        graph = _make_graph()
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=SqlGenerationOutput(thought="Result is in context", query=None, final_answer="There are 10 films.")
        )
        graph._generate_query_llm = llm
        state: dict = {"messages": [HumanMessage(content="How many?")], "query_loop_count": 0}
        update = await graph.generate_query(state)
        message = update["messages"][0]
        assert message.content == "There are 10 films."
        assert not message.tool_calls

    @pytest.mark.asyncio
    async def test_check_query_preserves_thought_on_tool_call(self):
        graph = _make_graph()
        llm = MagicMock()
        llm.ainvoke = AsyncMock(
            return_value=SqlCheckOutput(thought="Added LIMIT", query="SELECT * FROM film LIMIT 5")
        )
        graph._check_query_llm = llm
        prior = AIMessage(
            content="Get films",
            tool_calls=[
                {
                    "name": "sql_db_query",
                    "args": {"query": "SELECT * FROM film"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )
        state: dict = {"messages": [prior], "query_loop_count": 0}
        update = await graph.check_query(state)
        message = update["messages"][0]
        assert update["query_loop_count"] == 1
        assert message.tool_calls[0]["args"]["query"] == "SELECT * FROM film LIMIT 5"
        assert "thought" not in message.tool_calls[0]["args"]
        assert message.content == "Added LIMIT"