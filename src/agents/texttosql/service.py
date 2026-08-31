import logging
import time
from typing import List, Optional

from langchain_core.messages import HumanMessage

from src.platform.langfuse.client import LangfuseTracer
from src.platform.llm.protocol import LlmProviderClient
from src.platform.middleware import (
    AgentContext,
    AgentPipeline,
    ErrorHandlingMiddleware,
    LoggingMiddleware,
)

from .config import TextToSQLConfig
from .graph import build_text_to_sql_graph
from .tools import create_sql_tools
from src.platform.langfuse.langfuse_tracing_middleware import LangfuseTracingMiddleware

logger = logging.getLogger(__name__)


class TextToSQLService:
    """LangGraph text-to-SQL agent for querying PostgreSQL."""

    def __init__(
        self,
        llm_client: LlmProviderClient,
        langfuse_tracer: Optional[LangfuseTracer] = None,
        agent_config: Optional[TextToSQLConfig] = None,
    ):
        self.llm = llm_client
        self.langfuse_tracer = langfuse_tracer
        self.agent_config = agent_config or TextToSQLConfig()

        self.tools, self.db = create_sql_tools(
            database_url=self.agent_config.settings.postgres_database_url,
            include_tables=self.agent_config.include_tables,
            sample_rows_in_table_info=self.agent_config.sample_rows_in_table_info,
        )
        self.model = self.llm.get_langchain_model(
            model=self.agent_config.model,
            temperature=self.agent_config.temperature,
        )

        self.middleware_pipeline = AgentPipeline(
            middlewares=[
                LangfuseTracingMiddleware(
                    langfuse_tracer=langfuse_tracer,
                    trace_name="text_to_sql_request",
                    environment=self.agent_config.settings.environment,
                    build_trace_metadata=self._build_trace_metadata,
                    build_trace_output=self._build_trace_output,
                ),
                LoggingMiddleware(),
                ErrorHandlingMiddleware(),
            ],
            invoke_fn=self._core_invoke,
        )
        self.graph = build_text_to_sql_graph(
            model=self.model,
            tools=self.tools,
            config=self.agent_config,
            middleware_manager=self.middleware_pipeline.manager,
        )

        logger.info(
            "TextToSQLService initialized (dialect=%s, tables=%s)",
            self.agent_config.dialect,
            self.agent_config.include_tables,
        )

    async def ask(
        self,
        query: str,
        user_id: str = "api_user",
        model: Optional[str] = None,
    ) -> dict:
        """Run a natural-language question against PostgreSQL."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        model_to_use = model or self.agent_config.model
        if model_to_use != self.agent_config.model:
            self.model = self.llm.get_langchain_model(
                model=model_to_use,
                temperature=self.agent_config.temperature,
            )
            self.graph = build_text_to_sql_graph(
                model=self.model,
                tools=self.tools,
                config=self.agent_config,
                middleware_manager=self.middleware_pipeline.manager,
            )

        try:
            return await self._run_workflow(query, model_to_use, user_id)
        except Exception:
            logger.exception("Text-to-SQL execution failed")
            raise

    async def _core_invoke(self, ctx: AgentContext) -> list:
        """Run the LangGraph workflow; graph result is stored on ctx.metadata."""
        config = dict(ctx.config.get("graph_config", {}))
        result = await self.graph.ainvoke({"messages": list(ctx.messages)}, config=config)
        ctx.metadata["graph_result"] = result
        return result.get("messages", [])

    def _build_trace_metadata(self, ctx: AgentContext) -> dict:
        return dict(ctx.metadata.get("trace_metadata", {}))

    def _build_trace_output(self, ctx: AgentContext, result: list) -> dict:
        execution_time = time.time() - ctx.metadata.get("_trace_start_time", time.time())
        graph_result = ctx.metadata.get("graph_result") or {"messages": result}
        answer = self._extract_answer(graph_result)
        sql_queries = self._extract_sql_queries(graph_result)
        reasoning_steps = self._extract_reasoning_steps(graph_result, sql_queries)
        return {
            "answer": answer,
            "sql_queries": sql_queries,
            "reasoning_steps": reasoning_steps,
            "execution_time": execution_time,
        }

    async def _run_workflow(self, query: str, model_to_use: str, user_id: str) -> dict:
        start_time = time.time()
        session_id = f"user_{user_id}_session_{int(time.time())}"

        ctx = AgentContext(
            messages=[HumanMessage(content=query)],
            session_id=session_id,
            user_id=None,
            config={"model": model_to_use, "graph_config": {"thread_id": session_id}},
            agent_name="texttosql",
            metadata={
                "query": query,
                "user_id": user_id,
                "trace_metadata": {
                    "service": "text_to_sql",
                    "model": model_to_use,
                    "dialect": self.agent_config.dialect,
                },
            },
        )

        await self.middleware_pipeline.run(ctx)

        execution_time = time.time() - start_time
        graph_result = ctx.metadata.get("graph_result") or {}
        answer = self._extract_answer(graph_result)
        sql_queries = self._extract_sql_queries(graph_result)
        reasoning_steps = self._extract_reasoning_steps(graph_result, sql_queries)

        return {
            "query": query,
            "answer": answer,
            "sql_queries": sql_queries,
            "reasoning_steps": reasoning_steps,
            "execution_time": execution_time,
            "trace_id": ctx.metadata.get("trace_id"),
        }

    def _extract_answer(self, result: dict) -> str:
        messages = result.get("messages", [])
        if not messages:
            return "No answer generated."

        final_message = messages[-1]
        return final_message.content if hasattr(final_message, "content") else str(final_message)

    def _extract_sql_queries(self, result: dict) -> List[str]:
        queries: List[str] = []
        for message in result.get("messages", []):
            tool_calls = getattr(message, "tool_calls", None) or []
            for tool_call in tool_calls:
                if tool_call.get("name") == "sql_db_query":
                    query = tool_call.get("args", {}).get("query")
                    if query and query not in queries:
                        queries.append(query)
        return queries

    def _extract_reasoning_steps(self, result: dict, sql_queries: List[str]) -> List[str]:
        steps = ["Listed available database tables", "Fetched relevant table schemas"]
        if sql_queries:
            steps.append(f"Executed {len(sql_queries)} SQL quer{'y' if len(sql_queries) == 1 else 'ies'}")
        steps.append("Generated natural-language answer from query results")
        return steps
