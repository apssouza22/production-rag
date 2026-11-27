import logging
import uuid
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.platform.graph import END, START, GraphBuilder
from src.platform.middleware import MiddlewareManager

from .config import TextToSQLConfig
from .models import SqlCheckOutput, SqlGenerationOutput
from .prompts import CHECK_QUERY_SYSTEM_PROMPT, GENERATE_QUERY_SYSTEM_PROMPT
from .state import TextToSQLState

logger = logging.getLogger(__name__)


class TextToSQLGraph:
    """Builds and compiles the text-to-SQL agent workflow."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: list,
        config: TextToSQLConfig,
    ):
        self.model = model
        self.config = config

        tools_by_name = {tool.name: tool for tool in tools}
        self._list_tables_tool = tools_by_name["sql_db_list_tables"]
        self._get_schema_tool = tools_by_name["sql_db_schema"]
        self._run_query_tool = tools_by_name["sql_db_query"]

        self._generate_query_system_prompt = GENERATE_QUERY_SYSTEM_PROMPT.format(
            dialect=config.dialect,
            top_k=config.top_k,
            ontology=config.ontology_prompt,
        )
        self._check_query_system_prompt = CHECK_QUERY_SYSTEM_PROMPT.format(
            dialect=config.dialect,
        )
        self._generate_query_llm = model.with_structured_output(SqlGenerationOutput)
        self._check_query_llm = model.with_structured_output(SqlCheckOutput)

    @staticmethod
    def _build_sql_tool_message(thought: str, query: str, *, message_id: str | None = None) -> AIMessage:
        tool_call = {
            "name": "sql_db_query",
            "args": {"query": query},
            "id": f"sql_query_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        }
        message = AIMessage(content=thought, tool_calls=[tool_call])
        if message_id is not None:
            message.id = message_id
        return message

    async def list_tables(self, state: TextToSQLState) -> dict:
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": f"list_tables_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        }
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])
        tool_message = await self._list_tables_tool.ainvoke(tool_call)
        response = AIMessage(content=f"Available tables: {tool_message.content}")

        return {"messages": [tool_call_message, tool_message, response]}

    async def call_get_schema(self, state: TextToSQLState) -> dict:
        llm_with_tools = self.model.bind_tools([self._get_schema_tool], tool_choice="any")
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    async def generate_query(self, state: TextToSQLState) -> dict:
        system_message = {
            "role": "system",
            "content": self._generate_query_system_prompt,
        }
        result: SqlGenerationOutput = await self._generate_query_llm.ainvoke([system_message] + state["messages"])
        if result.query:
            response = self._build_sql_tool_message(result.thought, result.query)
            logger.info("Thought: %s, Query: %s", result.thought, result.query)
        else:
            response = AIMessage(content=result.final_answer or result.thought)
            logger.info("Final answer: %s", result.final_answer)
        return {"messages": [response]}

    async def check_query(self, state: TextToSQLState) -> dict:
        system_message = {
            "role": "system",
            "content": self._check_query_system_prompt,
        }
        prior_message = state["messages"][-1]
        tool_call = prior_message.tool_calls[0]
        thought = prior_message.content if isinstance(prior_message.content, str) else ""
        user_message = {
            "role": "user",
            "content": f"Thought: {thought}\nQuery: {tool_call['args']['query']}",
        }
        result: SqlCheckOutput = await self._check_query_llm.ainvoke([system_message, user_message])
        response = self._build_sql_tool_message(
            result.thought,
            result.query,
            message_id=state["messages"][-1].id,
        )
        query_loop_count = state.get("query_loop_count", 0) + 1
        return {"messages": [response], "query_loop_count": query_loop_count}

    def should_continue(self, state: TextToSQLState) -> Literal["check_query", END]:
        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return END
        query_loop_count = state.get("query_loop_count", 0)
        if query_loop_count >= self.config.max_query_loops:
            logger.warning(
                "Max query loops (%s) reached; ending SQL retry cycle",
                self.config.max_query_loops,
            )
            return END
        return "check_query"

    def compile(self, middleware_manager: MiddlewareManager | None = None):
        builder = GraphBuilder(TextToSQLState)
        if middleware_manager is not None:
            builder.set_middleware_manager(middleware_manager)

        (
            builder.add_node("list_tables", self.list_tables)
            .add_node("call_get_schema", self.call_get_schema)
            .add_tool_node("get_schema", [self._get_schema_tool])
            .add_node("generate_query", self.generate_query)
            .add_node("check_query", self.check_query)
            .add_tool_node("run_query", [self._run_query_tool])
            .add_edge(START, "list_tables")
            .add_edge("list_tables", "call_get_schema")
            .add_edge("call_get_schema", "get_schema")
            .add_edge("get_schema", "generate_query")
            .add_conditional_edges("generate_query", self.should_continue)
            .add_edge("check_query", "run_query")
            .add_edge("run_query", "generate_query")
        )

        logger.info("Text-to-SQL graph compiled successfully")
        return builder.compile()


def build_text_to_sql_graph(
    model: BaseChatModel,
    tools: list,
    config: TextToSQLConfig,
    middleware_manager: MiddlewareManager | None = None,
):
    """Build the LangGraph SQL agent workflow from the LangChain reference."""
    return TextToSQLGraph(
        model=model,
        tools=tools,
        config=config,
    ).compile(middleware_manager=middleware_manager)
