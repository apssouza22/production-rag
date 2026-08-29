import logging
import uuid
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.domain.agent_fault_tolerance import (
    build_llm_timeout,
    build_retry_policy,
    build_tool_retry_policy,
    build_tool_timeout,
)
from src.agents.texttosql.handlers import text_to_sql_error_handler

from .config import TextToSQLConfig
from .prompts import CHECK_QUERY_SYSTEM_PROMPT, GENERATE_QUERY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class TextToSQLGraph:
    """Builds and compiles the LangGraph text-to-SQL agent workflow."""

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
        )
        self._check_query_system_prompt = CHECK_QUERY_SYSTEM_PROMPT.format(
            dialect=config.dialect,
        )

        self._get_schema_node = ToolNode([self._get_schema_tool], name="get_schema")
        self._run_query_node = ToolNode([self._run_query_tool], name="run_query")

    async def list_tables(self, state: MessagesState) -> dict:
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

    async def call_get_schema(self, state: MessagesState) -> dict:
        llm_with_tools = self.model.bind_tools([self._get_schema_tool], tool_choice="any")
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    async def generate_query(self, state: MessagesState) -> dict:
        system_message = {
            "role": "system",
            "content": self._generate_query_system_prompt,
        }
        llm_with_tools = self.model.bind_tools([self._run_query_tool])
        response = await llm_with_tools.ainvoke([system_message] + state["messages"])
        return {"messages": [response]}

    async def check_query(self, state: MessagesState) -> dict:
        system_message = {
            "role": "system",
            "content": self._check_query_system_prompt,
        }
        tool_call = state["messages"][-1].tool_calls[0]
        user_message = {"role": "user", "content": tool_call["args"]["query"]}
        llm_with_tools = self.model.bind_tools([self._run_query_tool], tool_choice="any")
        response = await llm_with_tools.ainvoke([system_message, user_message])
        response.id = state["messages"][-1].id
        return {"messages": [response]}

    def should_continue(self, state: MessagesState) -> Literal["check_query", END]:
        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return END
        return "check_query"

    def _configure_tool_fault_tolerance(self, builder: StateGraph) -> dict:
        ft = self.config.fault_tolerance
        tool_node_kwargs: dict = {}

        if ft.enabled:
            builder.set_node_defaults(
                retry_policy=build_retry_policy(ft),
                timeout=build_llm_timeout(ft),
                error_handler=text_to_sql_error_handler,
            )
            tool_node_kwargs = {
                "retry_policy": build_tool_retry_policy(ft),
                "timeout": build_tool_timeout(ft),
            }

        return tool_node_kwargs

    def compile(self):
        builder = StateGraph(MessagesState)
        tool_node_kwargs = self._configure_tool_fault_tolerance(builder)

        builder.add_node("list_tables", self.list_tables)
        builder.add_node("call_get_schema", self.call_get_schema)
        builder.add_node("get_schema", self._get_schema_node, **tool_node_kwargs)
        builder.add_node("generate_query", self.generate_query)
        builder.add_node("check_query", self.check_query)
        builder.add_node("run_query", self._run_query_node, **tool_node_kwargs)

        builder.add_edge(START, "list_tables")
        builder.add_edge("list_tables", "call_get_schema")
        builder.add_edge("call_get_schema", "get_schema")
        builder.add_edge("get_schema", "generate_query")
        builder.add_conditional_edges("generate_query", self.should_continue)
        builder.add_edge("check_query", "run_query")
        builder.add_edge("run_query", "generate_query")

        logger.info("Text-to-SQL graph compiled successfully")
        return builder.compile()


def build_text_to_sql_graph(
    model: BaseChatModel,
    tools: list,
    config: TextToSQLConfig,
):
    """Build the LangGraph SQL agent workflow from the LangChain reference."""
    return TextToSQLGraph(model=model, tools=tools, config=config).compile()
