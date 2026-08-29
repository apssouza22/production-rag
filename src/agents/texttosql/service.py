import logging
import time
from typing import List, Optional

from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler

from src.domain.langfuse.client import LangfuseTracer
from src.domain.llm.protocol import LLMClient

from .config import TextToSQLConfig
from .graph import build_text_to_sql_graph
from .tools import create_sql_tools

logger = logging.getLogger(__name__)


class TextToSQLService:
    """LangGraph text-to-SQL agent for querying PostgreSQL."""

    def __init__(
        self,
        llm_client: LLMClient,
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
        self.graph = build_text_to_sql_graph(
            model=self.model,
            tools=self.tools,
            config=self.agent_config,
        )

        logger.info("TextToSQLService initialized (dialect=%s, tables=%s)", self.agent_config.dialect, self.agent_config.include_tables)

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
            )

        metadata = {
            "service": "text_to_sql",
            "model": model_to_use,
            "dialect": self.agent_config.dialect,
        }

        async def _execute_with_trace():
            if self.langfuse_tracer and self.langfuse_tracer.client:
                with self.langfuse_tracer.trace_agent_request(
                    name="text_to_sql_request",
                    input_data={"query": query},
                    user_id=user_id,
                    session_id=f"session_{user_id}",
                    environment=self.agent_config.settings.environment,
                    metadata=metadata,
                ) as trace_obj:
                    return await self._run_workflow(query, user_id, trace_obj)
            return await self._run_workflow(query, user_id, None)

        try:
            return await _execute_with_trace()
        except Exception:
            logger.exception("Text-to-SQL execution failed")
            raise

    async def _run_workflow(self, query: str, user_id: str, trace) -> dict:
        start_time = time.time()
        config = {"thread_id": f"user_{user_id}_session_{int(time.time())}"}

        if self.langfuse_tracer and trace:
            try:
                config["callbacks"] = [CallbackHandler()]
            except Exception as exc:
                logger.warning("Failed to create Langfuse callback handler: %s", exc)

        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )

        execution_time = time.time() - start_time
        answer = self._extract_answer(result)
        sql_queries = self._extract_sql_queries(result)
        reasoning_steps = self._extract_reasoning_steps(result, sql_queries)

        trace_id = None
        if trace:
            trace_id = getattr(trace, "trace_id", None) or self.langfuse_tracer.get_trace_id()
            trace.update(
                output={
                    "answer": answer,
                    "sql_queries": sql_queries,
                    "reasoning_steps": reasoning_steps,
                    "execution_time": execution_time,
                }
            )
            self.langfuse_tracer.flush()

        return {
            "query": query,
            "answer": answer,
            "sql_queries": sql_queries,
            "reasoning_steps": reasoning_steps,
            "execution_time": execution_time,
            "trace_id": trace_id,
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
