import logging
import time
from typing import List, Optional

from langchain_core.messages import HumanMessage
from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.domain.langfuse.client import LangfuseTracer
from src.domain.llm.protocol import LLMClient
from src.domain.middleware import (
    AgentContext,
    AgentPipeline,
    ErrorHandlingMiddleware,
    LoggingMiddleware,
)

from .config import GraphConfig
from .context import Context
from .graph import AgenticRAGGraph
from .middleware import GuardrailMiddleware
from .retrieval_settings import RetrievalSettings
from ...domain.langfuse.langfuse_tracing_middleware import LangfuseTracingMiddleware

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """Agentic RAG service backed by a compiled LangGraph workflow."""

    def __init__(
        self,
        llm_client: LLMClient,
        retrieval_settings: RetrievalSettings,
        graph_builder: AgenticRAGGraph,
        reranker_client: JinaRerankerClient | None = None,
        langfuse_tracer: Optional[LangfuseTracer] = None,
        graph_config: Optional[GraphConfig] = None,
    ):
        """Initialize agentic RAG service.

        :param llm_client: Client for LLM generation
        :param retrieval_settings: Mutable retrieval settings shared with the graph
        :param graph_builder: Graph builder used for per-request overrides
        :param reranker_client: Optional client for Jina reranking
        :param langfuse_tracer: Optional Langfuse tracer
        :param graph_config: Configuration for graph execution
        """
        self.llm = llm_client
        self.graph_builder = graph_builder
        self.retrieval_settings = retrieval_settings
        self.reranker = reranker_client
        self.langfuse_tracer = langfuse_tracer
        self.graph_config = graph_config or GraphConfig()

        self.middleware_pipeline = AgentPipeline(
            middlewares=[
                LangfuseTracingMiddleware(
                    langfuse_tracer=langfuse_tracer,
                    trace_name="agentic_rag_request",
                    environment=self.graph_config.settings.environment,
                    build_trace_metadata=self._build_trace_metadata,
                    build_trace_output=self._build_trace_output,
                ),
                GuardrailMiddleware(
                    llm_client=llm_client,
                    config=self.graph_config,
                    langfuse_tracer=langfuse_tracer,
                ),
                LoggingMiddleware(),
                ErrorHandlingMiddleware(),
            ],
            invoke_fn=self._core_invoke,
        )
        self.graph = self.graph_builder.compile(middleware_manager=self.middleware_pipeline.manager)

        logger.info("Initializing AgenticRAGService with configuration:")
        logger.info(f"  Model: {self.graph_config.model}")
        logger.info(f"  Top-k: {self.graph_config.top_k}")
        logger.info(f"  Hybrid search: {self.graph_config.use_hybrid}")
        logger.info(f"  Reranking: {self.graph_config.rerank_enabled}")
        logger.info(f"  Max retrieval attempts: {self.graph_config.max_retrieval_attempts}")
        logger.info(f"  Guardrail threshold: {self.graph_config.guardrail_threshold}")
        logger.info("✓ AgenticRAGService initialized successfully")

    async def ask(
        self,
        query: str,
        user_id: str = "api_user",
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        use_hybrid: Optional[bool] = None,
        rerank_enabled: Optional[bool] = None,
    ) -> dict:
        """Ask a question using agentic RAG.

        :param query: User question
        :param user_id: User identifier for tracing
        :param model: Optional model override
        :param top_k: Optional number of documents to retrieve
        :param use_hybrid: Optional hybrid search toggle
        :param rerank_enabled: Optional reranking toggle
        :returns: Dictionary with answer, sources, reasoning steps, and metadata
        :raises ValueError: If query is empty
        """
        model_to_use = model or self.graph_config.model

        self.retrieval_settings.top_k = top_k if top_k is not None else self.graph_config.top_k
        self.retrieval_settings.use_hybrid = (
            use_hybrid if use_hybrid is not None else self.graph_config.use_hybrid
        )
        self.retrieval_settings.rerank_enabled = (
            rerank_enabled if rerank_enabled is not None else self.graph_config.rerank_enabled
        )

        logger.info("=" * 80)
        logger.info("Starting Agentic RAG Request")
        logger.info(f"Query: {query}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Model: {model_to_use}")
        logger.info("=" * 80)

        # Validate input
        if not query or len(query.strip()) == 0:
            logger.error("Empty query received")
            raise ValueError("Query cannot be empty")

        try:
            return await self._run_workflow(query, model_to_use, user_id)
        except Exception as e:
            logger.error(f"Error in Agentic RAG execution: {str(e)}")
            logger.exception("Full traceback:")
            raise

    async def _core_invoke(self, ctx: AgentContext) -> list:
        """Run the LangGraph workflow; graph result is stored on ctx.metadata."""
        query = ctx.metadata["query"]
        model_to_use = ctx.config["model"]
        user_id = ctx.metadata["user_id"]
        trace = ctx.metadata.get("trace")

        state_input = {
            "messages": list(ctx.messages),
            "retrieval_attempts": 0,
            "guardrail_result": ctx.metadata.get("guardrail_result"),
            "routing_decision": None,
            "sources": None,
            "relevant_sources": [],
            "relevant_tool_artefacts": None,
            "grading_results": [],
            "metadata": {},
            "original_query": None,
            "rewritten_query": None,
        }

        trace_id = getattr(trace, "trace_id", None) if trace else None
        self.graph_builder.prepare_request(model=model_to_use, trace=trace)
        runtime_context = Context(trace_id=trace_id)

        config = dict(ctx.config.get("graph_config", {}))

        result = await self.graph.ainvoke(
            state_input,
            config=config,
            context=runtime_context,
        )
        ctx.metadata["graph_result"] = result
        return result.get("messages", [])

    def _build_trace_metadata(self, ctx: AgentContext) -> dict:
        """Build Langfuse metadata from the current request context."""
        return dict(ctx.metadata.get("trace_metadata", {}))

    def _build_trace_output(self, ctx: AgentContext, result: list) -> dict:
        """Build Langfuse trace output after pipeline execution."""
        execution_time = time.time() - ctx.metadata.get("_trace_start_time", time.time())
        guardrail_result = ctx.metadata.get("guardrail_result")
        graph_result = ctx.metadata.get("graph_result")

        if graph_result is not None:
            answer = self._extract_answer(graph_result)
            sources = self._extract_sources(graph_result)
            retrieval_attempts = graph_result.get("retrieval_attempts", 0)
            reasoning_steps = self._extract_reasoning_steps(graph_result, guardrail_result)
        else:
            answer = self._extract_answer({"messages": result})
            sources = []
            retrieval_attempts = 0
            reasoning_steps = self._extract_reasoning_steps({}, guardrail_result)

        return {
            "answer": answer,
            "sources_count": len(sources),
            "retrieval_attempts": retrieval_attempts,
            "reasoning_steps": reasoning_steps,
            "execution_time": execution_time,
        }

    async def _run_workflow(self, query: str, model_to_use: str, user_id: str) -> dict:
        """Execute the middleware pipeline and build the API response."""
        start_time = time.time()
        session_id = f"user_{user_id}_session_{int(time.time())}"

        logger.info("Invoking agent pipeline (tracing → guardrail → LangGraph)")

        ctx = AgentContext(
            messages=[HumanMessage(content=query)],
            session_id=session_id,
            user_id=None,
            config={"model": model_to_use, "graph_config": {"thread_id": session_id}},
            agent_name="fusionsearch",
            metadata={
                "query": query,
                "user_id": user_id,
                "trace_metadata": {
                    "service": "agentic_rag",
                    "top_k": self.graph_config.top_k,
                    "use_hybrid": self.graph_config.use_hybrid,
                    "model": model_to_use,
                },
            },
        )

        pipeline_result = await self.middleware_pipeline.run(ctx)

        execution_time = time.time() - start_time
        guardrail_result = ctx.metadata.get("guardrail_result")
        graph_result = ctx.metadata.get("graph_result")

        if graph_result is not None:
            answer = self._extract_answer(graph_result)
            sources = self._extract_sources(graph_result)
            retrieval_attempts = graph_result.get("retrieval_attempts", 0)
            reasoning_steps = self._extract_reasoning_steps(graph_result, guardrail_result)
            fault_tolerance = graph_result.get("metadata", {}).get("fault_tolerance")
        else:
            answer = self._extract_answer({"messages": pipeline_result})
            sources = []
            retrieval_attempts = 0
            reasoning_steps = self._extract_reasoning_steps({}, guardrail_result)
            fault_tolerance = None

        trace_id = ctx.metadata.get("trace_id")

        logger.info("=" * 80)
        logger.info("Agentic RAG Request Completed Successfully")
        logger.info(f"Answer length: {len(answer)} characters")
        logger.info(f"Sources found: {len(sources)}")
        logger.info(f"Retrieval attempts: {retrieval_attempts}")
        logger.info(f"Execution time: {execution_time:.2f}s")
        if trace_id:
            logger.info(f"Langfuse trace ID: {trace_id}")
        logger.info("=" * 80)

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "reasoning_steps": reasoning_steps,
            "retrieval_attempts": retrieval_attempts,
            "rewritten_query": graph_result.get("rewritten_query") if graph_result else None,
            "execution_time": execution_time,
            "guardrail_score": guardrail_result.score if guardrail_result else None,
            "trace_id": trace_id,
            "fault_tolerance": fault_tolerance,
        }

    def _extract_answer(self, result: dict) -> str:
        """Extract final answer from graph result."""
        messages = result.get("messages", [])
        if not messages:
            return "No answer generated."

        final_message = messages[-1]
        return final_message.content if hasattr(final_message, "content") else str(final_message)

    def _extract_sources(self, result: dict) -> List[dict]:
        """Extract sources from graph result."""
        sources = []
        relevant_sources = result.get("relevant_sources", [])

        for source in relevant_sources:
            if hasattr(source, "to_dict"):
                sources.append(source.to_dict())
            elif isinstance(source, dict):
                sources.append(source)

        return sources

    def _extract_reasoning_steps(self, result: dict, guardrail_result=None) -> List[str]:
        """Extract reasoning steps from graph result."""
        steps = []
        retrieval_attempts = result.get("retrieval_attempts", 0)
        grading_results = result.get("grading_results", [])

        if guardrail_result:
            steps.append(f"Validated query scope (score: {guardrail_result.score}/100)")
            if guardrail_result.score < self.graph_config.guardrail_threshold:
                steps.append("Query rejected as out of scope")
                return steps

        if retrieval_attempts > 0:
            if self.retrieval_settings.rerank_enabled and self.reranker and self.reranker.is_configured:
                steps.append(f"Retrieved and reranked documents ({retrieval_attempts} attempt(s))")
            else:
                steps.append(f"Retrieved documents ({retrieval_attempts} attempt(s))")

        if grading_results:
            relevant_count = sum(1 for g in grading_results if g.is_relevant)
            steps.append(f"Graded documents ({relevant_count} relevant)")

        if result.get("rewritten_query"):
            steps.append("Rewritten query for better results")

        if result.get("messages"):
            steps.append("Generated answer from context")

        return steps

    def get_graph_mermaid(self) -> str:
        """Return a Mermaid diagram of the compiled graph."""
        return self.graph.get_graph().draw_mermaid()
