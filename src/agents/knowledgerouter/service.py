import logging
import time
from typing import List, Optional

from langchain_core.messages import HumanMessage

from src.agents.fusionsearch.agentic_rag import AgenticRAGService
from src.agents.texttosql.service import TextToSQLService
from src.platform.langfuse.client import LangfuseTracer
from src.platform.llm.protocol import LlmProviderClient
from src.platform.middleware import (
    AgentContext,
    AgentPipeline,
    ErrorHandlingMiddleware,
    LoggingMiddleware,
    TrajectoryMiddleware,
)

from .config import KnowledgeRouterConfig
from .graph import build_knowledge_router_graph
from .schemas import AgentResultItem, ClassificationItem, classification_to_schema
from src.platform.langfuse.langfuse_tracing_middleware import LangfuseTracingMiddleware

logger = logging.getLogger(__name__)


class KnowledgeRouterService:
    """Multi-source knowledge router that delegates to specialized retrieval agents."""

    def __init__(
        self,
        agentic_rag_service: AgenticRAGService,
        text_to_sql_service: TextToSQLService,
        llm_client: LlmProviderClient,
        langfuse_tracer: Optional[LangfuseTracer] = None,
        agent_config: Optional[KnowledgeRouterConfig] = None,
    ):
        self.agentic_rag = agentic_rag_service
        self.text_to_sql = text_to_sql_service
        self.llm = llm_client
        self.langfuse_tracer = langfuse_tracer
        self.agent_config = agent_config or KnowledgeRouterConfig()

        self.router_model = self.llm.get_langchain_model(
            model=self.agent_config.router_model,
            temperature=self.agent_config.temperature,
        )
        self.synthesis_model = self.llm.get_langchain_model(
            model=self.agent_config.model,
            temperature=self.agent_config.temperature,
        )
        self.graph = build_knowledge_router_graph(
            router_model=self.router_model,
            synthesis_model=self.synthesis_model,
            agentic_rag_service=self.agentic_rag,
            text_to_sql_service=self.text_to_sql,
            config=self.agent_config,
        )

        self.middleware_pipeline = AgentPipeline(
            middlewares=[
                LangfuseTracingMiddleware(
                    langfuse_tracer=langfuse_tracer,
                    trace_name="knowledge_router_request",
                    environment=self.agent_config.settings.environment,
                    build_trace_metadata=self._build_trace_metadata,
                    build_trace_output=self._build_trace_output,
                ),
                TrajectoryMiddleware(),
                LoggingMiddleware(),
                ErrorHandlingMiddleware(),
            ],
            invoke_fn=self._core_invoke,
        )

        logger.info("KnowledgeRouterService initialized")

    async def ask(
        self,
        query: str,
        user_id: str = "api_user",
    ) -> dict:
        """Route a question to the appropriate knowledge sources and synthesize the answer."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        try:
            return await self._run_workflow(query, user_id)
        except Exception:
            logger.exception("Knowledge router execution failed")
            raise

    async def _core_invoke(self, ctx: AgentContext) -> list:
        """Run the LangGraph workflow; graph result is stored on ctx.metadata."""
        query = ctx.metadata["query"]
        config = dict(ctx.config.get("graph_config", {}))
        result = await self.graph.ainvoke({"query": query}, config=config)
        ctx.metadata["graph_result"] = result
        return []

    def _build_trace_metadata(self, ctx: AgentContext) -> dict:
        return dict(ctx.metadata.get("trace_metadata", {}))

    def _build_trace_output(self, ctx: AgentContext, result: list) -> dict:
        execution_time = time.time() - ctx.metadata.get("_trace_start_time", time.time())
        graph_result = ctx.metadata.get("graph_result", {})
        classifications = [
            classification_to_schema(item) for item in graph_result.get("classifications", [])
        ]
        agent_results = self._build_agent_results(graph_result.get("results", []))
        reasoning_steps = self._extract_reasoning_steps(classifications, agent_results)
        return {
            "answer": graph_result.get("final_answer", ""),
            "classifications": [item.model_dump() for item in classifications],
            "agent_results": [item.model_dump() for item in agent_results],
            "reasoning_steps": reasoning_steps,
            "execution_time": execution_time,
        }

    async def _run_workflow(self, query: str, user_id: str) -> dict:
        start_time = time.time()
        session_id = f"user_{user_id}_session_{int(time.time())}"

        ctx = AgentContext(
            messages=[HumanMessage(content=query)],
            session_id=session_id,
            user_id=None,
            config={"graph_config": {"thread_id": session_id}},
            agent_name="knowledgerouter",
            metadata={
                "query": query,
                "user_id": user_id,
                "trace_metadata": {
                    "service": "knowledge_router",
                    "model": self.agent_config.model,
                },
            },
        )

        await self.middleware_pipeline.run(ctx)

        execution_time = time.time() - start_time
        graph_result = ctx.metadata.get("graph_result") or {}
        classifications = [
            classification_to_schema(item) for item in graph_result.get("classifications", [])
        ]
        agent_results = self._build_agent_results(graph_result.get("results", []))
        reasoning_steps = self._extract_reasoning_steps(classifications, agent_results)

        trajectory = ctx.metadata.get("trajectory")

        return {
            "query": query,
            "answer": graph_result.get("final_answer", ""),
            "classifications": classifications,
            "agent_results": agent_results,
            "reasoning_steps": reasoning_steps,
            "execution_time": execution_time,
            "trace_id": ctx.metadata.get("trace_id"),
            "trajectory": trajectory.summary() if trajectory else None,
        }

    def _build_agent_results(self, results: list) -> List[AgentResultItem]:
        return [
            AgentResultItem(
                source=item["source"],
                result=item["result"],
                metadata=item.get("metadata", {}),
            )
            for item in results
        ]

    def _extract_reasoning_steps(
        self,
        classifications: List[ClassificationItem],
        agent_results: List[AgentResultItem],
    ) -> List[str]:
        sources = [item.source for item in classifications]
        steps = [f"Classified query into {len(sources)} source(s): {', '.join(sources)}"]

        if len(agent_results) > 1:
            steps.append(f"Queried {len(agent_results)} agents in parallel")
        elif agent_results:
            steps.append(f"Queried {agent_results[0].source} agent")

        if len(agent_results) > 1:
            steps.append("Synthesized combined answer")
        else:
            steps.append("Returned answer from selected agent")

        return steps
