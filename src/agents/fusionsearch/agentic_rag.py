import logging
import time
from typing import List, Optional

from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler

from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.domain.langfuse.client import LangfuseTracer
from src.domain.llm.protocol import LLMClient

from .config import GraphConfig
from .context import Context
from .retrieval_settings import RetrievalSettings

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """Agentic RAG service 

    This implementation uses:
    - context_schema for dependency injection
    - Runtime[Context] for type-safe access in nodes
    - Direct client invocation (no pre-built runnables)
    - Lightweight nodes as pure functions
    """

    def __init__(
        self,
        llm_client: LLMClient,
        graph,
        retrieval_settings: RetrievalSettings,
        reranker_client: JinaRerankerClient | None = None,
        langfuse_tracer: Optional[LangfuseTracer] = None,
        graph_config: Optional[GraphConfig] = None,
    ):
        """Initialize agentic RAG service.

        :param llm_client: Client for LLM generation
        :param graph: Compiled LangGraph workflow (from AgenticRAGGraph.compile())
        :param retrieval_settings: Mutable retrieval settings shared with the graph
        :param reranker_client: Optional client for Jina reranking
        :param langfuse_tracer: Optional Langfuse tracer
        :param graph_config: Configuration for graph execution
        """
        self.llm = llm_client
        self.graph = graph
        self.retrieval_settings = retrieval_settings
        self.reranker = reranker_client
        self.langfuse_tracer = langfuse_tracer
        self.graph_config = graph_config or GraphConfig()

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

        metadata = {
            "service": "agentic_rag",
            "top_k": self.graph_config.top_k,
            "use_hybrid": self.graph_config.use_hybrid,
            "model": model_to_use,
        }

        async def _execute_with_trace():
            """Execute the workflow with or without tracing context."""
            if self.langfuse_tracer and self.langfuse_tracer.client:
                with self.langfuse_tracer.trace_agent_request(
                    name="agentic_rag_request",
                    input_data={"query": query},
                    user_id=user_id,
                    session_id=f"session_{user_id}",
                    environment=self.graph_config.settings.environment,
                    metadata=metadata,
                ) as trace_obj:
                    return await self._run_workflow(query, model_to_use, user_id, trace_obj)
            return await self._run_workflow(query, model_to_use, user_id, None)

        try:
            return await _execute_with_trace()
        except Exception as e:
            logger.error(f"Error in Agentic RAG execution: {str(e)}")
            logger.exception("Full traceback:")
            raise

    async def _run_workflow(self, query: str, model_to_use: str, user_id: str, trace) -> dict:
        """Execute the workflow with the given trace context."""
        try:
            start_time = time.time()

            logger.info("Invoking LangGraph workflow")

            # State initialization
            state_input = {
                "messages": [HumanMessage(content=query)],
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
            }

            # Runtime context (dependencies)
            runtime_context = Context(
                llm_client=self.llm,
                langfuse_tracer=self.langfuse_tracer,
                trace=trace,
                model_name=model_to_use,
                temperature=self.graph_config.temperature,
                max_retrieval_attempts=self.graph_config.max_retrieval_attempts,
                guardrail_threshold=self.graph_config.guardrail_threshold,
            )

            config = {"thread_id": f"user_{user_id}_session_{int(time.time())}"}

            if self.langfuse_tracer and trace:
                try:
                    callback_handler = CallbackHandler()
                    config["callbacks"] = [callback_handler]
                    logger.info("CallbackHandler added for LangGraph LLM tracing")
                except Exception as e:
                    logger.warning(f"Failed to create CallbackHandler: {e}")

            result = await self.graph.ainvoke(
                state_input,
                config=config,
                context=runtime_context,
            )

            execution_time = time.time() - start_time
            logger.info(f"✓ Graph execution completed in {execution_time:.2f}s")

            # Extract results
            answer = self._extract_answer(result)
            sources = self._extract_sources(result)
            retrieval_attempts = result.get("retrieval_attempts", 0)
            reasoning_steps = self._extract_reasoning_steps(result)

            trace_id = None
            if trace:
                trace_id = getattr(trace, "trace_id", None) or self.langfuse_tracer.get_trace_id()
                trace.update(
                    output={
                        "answer": answer,
                        "sources_count": len(sources),
                        "retrieval_attempts": retrieval_attempts,
                        "reasoning_steps": reasoning_steps,
                        "execution_time": execution_time,
                    }
                )
                self.langfuse_tracer.flush()

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
                "rewritten_query": result.get("rewritten_query"),
                "execution_time": execution_time,
                "guardrail_score": result.get("guardrail_result").score if result.get("guardrail_result") else None,
                "trace_id": trace_id,
                "fault_tolerance": result.get("metadata", {}).get("fault_tolerance"),
            }

        except Exception as e:
            logger.error(f"Error in workflow execution: {str(e)}")
            logger.exception("Full traceback:")

            if trace:
                trace.update(output={"error": str(e)}, level="ERROR")
                self.langfuse_tracer.flush()

            raise

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

    def _extract_reasoning_steps(self, result: dict) -> List[str]:
        """Extract reasoning steps from graph result."""
        steps = []
        retrieval_attempts = result.get("retrieval_attempts", 0)
        guardrail_result = result.get("guardrail_result")
        grading_results = result.get("grading_results", [])

        if guardrail_result:
            steps.append(f"Validated query scope (score: {guardrail_result.score}/100)")

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

        steps.append("Generated answer from context")

        return steps
