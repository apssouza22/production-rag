import logging
import time
from typing import List, Optional

from langfuse.langchain import CallbackHandler

from src.domain.agents.fusionsearch.agentic_rag import AgenticRAGService
from src.domain.agents.texttosql.service import TextToSQLService
from src.domain.langfuse.client import LangfuseTracer
from src.domain.llm.protocol import LLMClient

from .config import KnowledgeRouterConfig
from .graph import build_knowledge_router_graph
from .schemas import AgentResultItem, ClassificationItem, classification_to_schema

logger = logging.getLogger(__name__)


class KnowledgeRouterService:
    """Multi-source knowledge router that delegates to specialized retrieval agents."""

    def __init__(
        self,
        agentic_rag_service: AgenticRAGService,
        text_to_sql_service: TextToSQLService,
        llm_client: LLMClient,
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

        logger.info("KnowledgeRouterService initialized")

    async def ask(
        self,
        query: str,
        user_id: str = "api_user",
        model: Optional[str] = None,
    ) -> dict:
        """Route a question to the appropriate knowledge sources and synthesize the answer."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        model_to_use = model or self.agent_config.model
        if model_to_use != self.agent_config.model:
            self.agent_config = self.agent_config.model_copy(update={"model": model_to_use})
            self.synthesis_model = self.llm.get_langchain_model(
                model=model_to_use,
                temperature=self.agent_config.temperature,
            )
            self.graph = build_knowledge_router_graph(
                router_model=self.router_model,
                synthesis_model=self.synthesis_model,
                agentic_rag_service=self.agentic_rag,
                text_to_sql_service=self.text_to_sql,
                config=self.agent_config,
            )

        trace = None
        metadata = {
            "env": self.agent_config.settings.environment,
            "service": "knowledge_router",
            "model": model_to_use,
        }

        if self.langfuse_tracer and self.langfuse_tracer.client:
            trace = self.langfuse_tracer.client.start_as_current_span(name="knowledge_router_request")

        async def _execute_with_trace():
            if trace is not None:
                with trace as trace_obj:
                    trace_obj.update(
                        input={"query": query},
                        metadata=metadata,
                        user_id=user_id,
                        session_id=f"session_{user_id}",
                    )
                    return await self._run_workflow(query, user_id, trace_obj)
            return await self._run_workflow(query, user_id, None)

        try:
            return await _execute_with_trace()
        except Exception:
            logger.exception("Knowledge router execution failed")
            raise

    async def _run_workflow(self, query: str, user_id: str, trace) -> dict:
        start_time = time.time()
        config = {"thread_id": f"user_{user_id}_session_{int(time.time())}"}

        if self.langfuse_tracer and trace:
            try:
                config["callbacks"] = [CallbackHandler()]
            except Exception as exc:
                logger.warning("Failed to create Langfuse callback handler: %s", exc)

        result = await self.graph.ainvoke({"query": query}, config=config)

        execution_time = time.time() - start_time
        classifications = [
            classification_to_schema(item) for item in result.get("classifications", [])
        ]
        agent_results = self._build_agent_results(result.get("results", []))
        reasoning_steps = self._extract_reasoning_steps(classifications, agent_results)

        trace_id = None
        if trace:
            trace_id = getattr(trace, "trace_id", None) or self.langfuse_tracer.get_trace_id()
            trace.update(
                output={
                    "answer": result.get("final_answer", ""),
                    "classifications": [item.model_dump() for item in classifications],
                    "agent_results": [item.model_dump() for item in agent_results],
                    "reasoning_steps": reasoning_steps,
                    "execution_time": execution_time,
                }
            )
            trace.end()
            self.langfuse_tracer.flush()

        return {
            "query": query,
            "answer": result.get("final_answer", ""),
            "classifications": classifications,
            "agent_results": agent_results,
            "reasoning_steps": reasoning_steps,
            "execution_time": execution_time,
            "trace_id": trace_id,
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
