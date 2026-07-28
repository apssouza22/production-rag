"""Tests for the text-to-SQL agent."""

from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from src.domain.agents.texttosql.config import TextToSQLConfig
from src.domain.agents.texttosql.service import TextToSQLService
from src.domain.agents.texttosql.tools import _validate_read_only_query


@pytest.fixture
def mock_ollama_client():
    client = Mock()
    client.get_langchain_model.return_value = Mock()
    return client


@pytest.fixture
def test_service(mock_ollama_client, monkeypatch):
    monkeypatch.setattr(
        "src.domain.agents.texttosql.service.create_sql_tools",
        lambda **kwargs: ([Mock(name="sql_db_list_tables"), Mock(name="sql_db_schema"), Mock(name="sql_db_query")], Mock()),
    )
    monkeypatch.setattr(
        "src.domain.agents.texttosql.service.build_text_to_sql_graph",
        lambda **kwargs: AsyncMock(),
    )

    return TextToSQLService(
        llm_client=mock_ollama_client,
        langfuse_tracer=None,
        agent_config=TextToSQLConfig(model="gpt-4o-mini", top_k=5),
    )


class TestQueryValidation:
    def test_allows_select_query(self):
        assert _validate_read_only_query("SELECT COUNT(*) FROM papers") is None

    def test_allows_with_query(self):
        assert _validate_read_only_query("WITH recent AS (SELECT * FROM papers) SELECT * FROM recent") is None

    def test_rejects_insert(self):
        assert "read-only" in _validate_read_only_query("INSERT INTO papers VALUES (1)")

    def test_rejects_delete(self):
        assert "read-only" in _validate_read_only_query("DELETE FROM papers")

    def test_rejects_empty_query(self):
        assert "empty" in _validate_read_only_query("   ")


class TestTextToSQLService:
    def test_service_initialization(self, test_service):
        assert test_service.llm is not None
        assert test_service.graph is not None
        assert test_service.agent_config.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_ask_empty_query_validation(self, test_service):
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await test_service.ask(query="")

    @pytest.mark.asyncio
    async def test_ask_returns_structured_response(self, test_service):
        test_service.graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="How many papers?"),
                    AIMessage(
                        content="There are 10 papers.",
                        tool_calls=[
                            {
                                "name": "sql_db_query",
                                "args": {"query": "SELECT COUNT(*) FROM papers"},
                                "id": "call_1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                ]
            }
        )

        result = await test_service.ask(query="How many papers?")

        assert result["query"] == "How many papers?"
        assert result["answer"] == "There are 10 papers."
        assert result["sql_queries"] == ["SELECT COUNT(*) FROM papers"]
        assert len(result["reasoning_steps"]) >= 3
