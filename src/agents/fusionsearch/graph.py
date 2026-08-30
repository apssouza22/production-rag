import logging
import time
from typing import Dict, List, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.domain.graph import END, GraphBuilder, START, ToolNode, tools_condition

from src.domain.agent_fault_tolerance import (
    build_llm_timeout,
    build_retry_policy,
    build_tool_retry_policy,
    build_tool_timeout,
)
from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.domain.langfuse.client import LangfuseTracer
from src.domain.llm.protocol import LLMClient
from src.domain.middleware import MiddlewareManager, middleware_tool_wrappers
from src.domain.opensearch.client import OpenSearchClient
from src.agents.fusionsearch.handlers import route_agentic_rag_failure
from src.agents.fusionsearch.models import GradeDocuments, GradingResult
from src.agents.fusionsearch.utils import get_latest_context, get_latest_query
from src.agents.fusionsearch.prompts import (
    GENERATE_ANSWER_PROMPT,
    GRADE_DOCUMENTS_PROMPT,
    REWRITE_PROMPT,
)

from .config import GraphConfig
from .context import Context
from .retrieval_settings import RetrievalSettings
from .state import AgentState
from .tools import create_retriever_tool

logger = logging.getLogger(__name__)


class QueryRewriteOutput(BaseModel):
    """Structured output for query rewriting."""

    rewritten_query: str = Field(
        description="The improved query optimized for document retrieval"
    )
    reasoning: str = Field(
        description="Brief explanation of how the query was improved"
    )


class AgenticRAGGraph:
    """Builds and compiles the agentic RAG workflow."""

    def __init__(
        self,
        llm_client: LLMClient,
        opensearch_client: OpenSearchClient,
        embeddings_client: JinaEmbeddingsClient,
        retrieval_settings: RetrievalSettings,
        config: GraphConfig,
        reranker_client: JinaRerankerClient | None = None,
        langfuse_tracer: Optional[LangfuseTracer] = None,
    ):
        self.llm_client = llm_client
        self.opensearch_client = opensearch_client
        self.embeddings_client = embeddings_client
        self.reranker_client = reranker_client
        self.retrieval_settings = retrieval_settings
        self.config = config
        self.langfuse_tracer = langfuse_tracer
        self._trace = None
        self._model_name: Optional[str] = None

    def prepare_request(self, *, model: Optional[str] = None, trace=None) -> None:
        """Set per-request overrides before graph invocation."""
        self._trace = trace
        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name or self.config.model

    @property
    def tracing_enabled(self) -> bool:
        return (
            self._trace is not None
            and self.langfuse_tracer is not None
            and self.langfuse_tracer.client is not None
        )

    def _create_node_span(
        self,
        name: str,
        input_data: dict,
        metadata: Optional[dict] = None,
    ):
        if not self.tracing_enabled:
            return None
        try:
            return self.langfuse_tracer.create_span(
                trace=self._trace,
                name=name,
                input_data=input_data,
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning("Failed to create span for %s: %s", name, exc)
            return None

    def _end_node_span(
        self,
        span,
        output: dict,
        metadata: Optional[dict] = None,
        level: Optional[str] = None,
    ) -> None:
        if not span:
            return
        if level:
            self.langfuse_tracer.update_span(span, output=output, metadata=metadata, level=level)
            self.langfuse_tracer.end_span(span)
            return
        self.langfuse_tracer.end_span(span, output=output, metadata=metadata)

    async def retrieve(self, state: AgentState) -> Dict[str, Union[int, str, list]]:
        """Initiate retrieval or return fallback if max attempts reached."""
        logger.info("NODE: retrieve")
        start_time = time.time()
        messages = state["messages"]
        question = get_latest_query(messages)
        current_attempts = state.get("retrieval_attempts", 0)
        max_attempts = self.config.max_retrieval_attempts

        updates: dict = {}
        if state.get("original_query") is None:
            updates["original_query"] = question

        span = self._create_node_span(
            "document_retrieval_initiation",
            {
                "query": question,
                "attempt": current_attempts + 1,
                "max_attempts": max_attempts,
            },
            {"node": "retrieve"},
        )

        if current_attempts >= max_attempts:
            logger.warning("Max retrieval attempts (%s) reached", max_attempts)
            fallback_msg = (
                f"I apologize, but I couldn't find relevant research papers after {max_attempts} attempts.\n"
                "This may be because:\n"
                "1. No papers in the database contain relevant information\n"
                "2. The query terms don't match the indexed content\n\n"
                "Please try rephrasing your question with more specific technical terms."
            )
            if span:
                self._end_node_span(
                    span,
                    {"status": "max_attempts_reached", "fallback": True},
                    {"execution_time_ms": (time.time() - start_time) * 1000},
                )
            return {**updates, "messages": [AIMessage(content=fallback_msg)]}

        new_attempt_count = current_attempts + 1
        updates["retrieval_attempts"] = new_attempt_count
        updates["messages"] = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"retrieve_{new_attempt_count}",
                        "name": "retrieve_papers",
                        "args": {"query": question},
                    }
                ],
            )
        ]

        if span:
            self._end_node_span(
                span,
                {
                    "status": "tool_call_created",
                    "query": question,
                    "attempt": new_attempt_count,
                },
                {"execution_time_ms": (time.time() - start_time) * 1000},
            )

        return updates

    async def grade_documents(self, state: AgentState) -> Dict[str, str | list]:
        """Grade retrieved documents for relevance."""
        logger.info("NODE: grade_documents")
        start_time = time.time()
        question = get_latest_query(state["messages"])
        context = get_latest_context(state["messages"])

        span = self._create_node_span(
            "document_grading",
            {
                "query": question,
                "context_length": len(context) if context else 0,
                "has_context": context is not None,
            },
            {"node": "grade_documents", "model": self.model_name},
        )

        if not context:
            logger.warning("No context found, routing to rewrite_query")
            if span:
                self._end_node_span(
                    span,
                    {"routing_decision": "rewrite_query", "reason": "no_context"},
                    {"execution_time_ms": (time.time() - start_time) * 1000},
                )
            return {"routing_decision": "rewrite_query", "grading_results": []}

        try:
            grading_prompt = GRADE_DOCUMENTS_PROMPT.format(context=context, question=question)
            llm = self.llm_client.get_langchain_model(model=self.model_name, temperature=0.0)
            structured_llm = llm.with_structured_output(GradeDocuments)
            grading_response = await structured_llm.ainvoke(grading_prompt)

            is_relevant = grading_response.binary_score == "yes"
            score = 1.0 if is_relevant else 0.0
            grading_result = GradingResult(
                document_id="retrieved_docs",
                is_relevant=is_relevant,
                score=score,
                reasoning=grading_response.reasoning,
            )
        except Exception as exc:
            logger.error("LLM grading failed: %s, falling back to heuristic", exc)
            is_relevant = len(context.strip()) > 50
            grading_result = GradingResult(
                document_id="retrieved_docs",
                is_relevant=is_relevant,
                score=1.0 if is_relevant else 0.0,
                reasoning=(
                    f"Fallback heuristic (LLM failed): "
                    f"{'sufficient content' if is_relevant else 'insufficient content'}"
                ),
            )

        route = "generate_answer" if is_relevant else "rewrite_query"
        if span:
            self._end_node_span(
                span,
                {
                    "routing_decision": route,
                    "is_relevant": is_relevant,
                    "score": grading_result.score,
                    "reasoning": grading_result.reasoning,
                },
                {
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "context_length": len(context),
                },
            )

        return {"routing_decision": route, "grading_results": [grading_result]}

    async def rewrite_query(self, state: AgentState) -> Dict[str, str | List]:
        """Rewrite the original query for better document retrieval."""
        logger.info("NODE: rewrite_query")
        start_time = time.time()
        original_question = state.get("original_query") or state["messages"][0].content
        current_attempt = state.get("retrieval_attempts", 0)
        llm_duration = None

        span = self._create_node_span(
            "query_rewriting",
            {"original_query": original_question, "attempt": current_attempt},
            {"node": "rewrite_query", "strategy": "llm_based_expansion", "model": self.model_name},
        )

        try:
            llm = self.llm_client.get_langchain_model(model=self.model_name, temperature=0.3)
            structured_llm = llm.with_structured_output(QueryRewriteOutput)
            prompt = REWRITE_PROMPT.format(question=original_question)
            llm_start = time.time()
            result: QueryRewriteOutput = await structured_llm.ainvoke(prompt)

            if not result or not result.rewritten_query:
                raise ValueError("LLM failed to return valid structured output for query rewriting")

            rewritten_query = result.rewritten_query.strip()
            if not rewritten_query:
                raise ValueError("LLM returned empty rewritten query")

            reasoning = result.reasoning
            llm_duration = time.time() - llm_start
        except Exception as exc:
            logger.error("Failed to rewrite query using LLM: %s", exc)
            rewritten_query = f"{original_question} research paper arxiv machine learning"
            reasoning = "Fallback: Simple keyword expansion due to LLM error"

        if span:
            self._end_node_span(
                span,
                {
                    "rewritten_query": rewritten_query,
                    "reasoning": reasoning,
                    "original_query": original_question,
                },
                {
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "original_length": len(original_question),
                    "rewritten_length": len(rewritten_query),
                    "llm_duration_seconds": llm_duration,
                },
            )

        return {
            "messages": [HumanMessage(content=rewritten_query)],
            "rewritten_query": rewritten_query,
        }

    async def generate_answer(self, state: AgentState) -> Dict[str, List[AIMessage]]:
        """Generate the final answer from retrieved context."""
        logger.info("NODE: generate_answer")
        start_time = time.time()
        question = get_latest_query(state["messages"])
        context = get_latest_context(state["messages"])
        sources_count = len(state.get("relevant_sources", []))

        if not context:
            context = "No relevant documents found."
            logger.warning("No context available for answer generation")

        span = self._create_node_span(
            "answer_generation",
            {
                "query": question,
                "context_length": len(context),
                "sources_count": sources_count,
            },
            {
                "node": "generate_answer",
                "model": self.model_name,
                "temperature": self.config.temperature,
            },
        )

        try:
            answer_prompt = GENERATE_ANSWER_PROMPT.format(context=context, question=question)
            llm = self.llm_client.get_langchain_model(
                model=self.model_name,
                temperature=self.config.temperature,
            )
            response = await llm.ainvoke(answer_prompt)
            answer = response.content if hasattr(response, "content") else str(response)

            if span:
                self._end_node_span(
                    span,
                    {"answer_length": len(answer), "sources_used": sources_count},
                    {
                        "execution_time_ms": (time.time() - start_time) * 1000,
                        "context_length": len(context),
                    },
                )
        except Exception as exc:
            logger.error("LLM answer generation failed: %s", exc)
            answer = (
                "I apologize, but I encountered an error while generating the answer: "
                f"{exc}\n\nPlease try again or rephrase your question."
            )
            if span:
                self._end_node_span(
                    span,
                    {"error": str(exc), "fallback": True},
                    {"execution_time_ms": (time.time() - start_time) * 1000},
                    level="ERROR",
                )

        return {"messages": [AIMessage(content=answer)]}

    async def handle_failure(self, state: AgentState) -> Dict[str, List[AIMessage]]:
        """Return a graceful fallback message after a node exhausts its retries."""
        logger.info("NODE: handle_failure")
        fault = (state.get("metadata") or {}).get("fault_tolerance", {})
        failed_node = fault.get("failed_node", "unknown")
        response_text = (
            "I apologize, but I encountered a temporary issue while processing your request "
            f"at the '{failed_node}' step.\n\n"
            "This is usually caused by a brief network or service interruption. "
            "Please try your question again in a few moments.\n\n"
            "If the problem persists, the upstream LLM or search service may be unavailable."
        )
        return {"messages": [AIMessage(content=response_text)]}

    def _configure_fault_tolerance(self, workflow: GraphBuilder) -> tuple[dict, dict]:
        ft = self.config.fault_tolerance
        no_fault_tolerance: dict = {}
        fault_tolerance: dict = {}

        if ft.enabled:
            workflow.set_node_defaults(
                retry_policy=build_retry_policy(ft),
                timeout=build_llm_timeout(ft),
                error_handler=route_agentic_rag_failure,
            )
            no_fault_tolerance = {
                "retry_policy": None,
                "error_handler": None,
                "timeout": None,
            }
            fault_tolerance = {
                "retry_policy": build_tool_retry_policy(ft),
                "timeout": build_tool_timeout(ft),
            }

        return no_fault_tolerance, fault_tolerance

    def compile(self, middleware_manager: Optional[MiddlewareManager] = None):
        """Build and compile the agentic RAG workflow."""
        logger.info("Building agentic RAG workflow with context_schema")

        workflow = GraphBuilder(AgentState, context_schema=Context)

        retriever_tool = create_retriever_tool(
            opensearch_client=self.opensearch_client,
            embeddings_client=self.embeddings_client,
            retrieval_settings=self.retrieval_settings,
            reranker_client=self.reranker_client,
        )
        no_fault_tolerance, fault_tolerance = self._configure_fault_tolerance(workflow)

        (
            workflow
            .add_node("retrieve", self.retrieve)
            .add_node(
                "tool_retrieve",
                ToolNode(
                    [retriever_tool],
                    **middleware_tool_wrappers(middleware_manager),
                ),
                **fault_tolerance,
            )
            .add_node("grade_documents", self.grade_documents)
            .add_node("rewrite_query", self.rewrite_query)
            .add_node("generate_answer", self.generate_answer)
            .add_node("handle_failure", self.handle_failure, **no_fault_tolerance)
            .add_edge(START, "retrieve")
            .add_conditional_edges(
                "retrieve",
                tools_condition,
                {
                    "tools": "tool_retrieve",
                    END: END,
                },
            )
            .add_edge("tool_retrieve", "grade_documents")
            .add_conditional_edges(
                "grade_documents",
                lambda state: state.get("routing_decision", "generate_answer"),
                {
                    "generate_answer": "generate_answer",
                    "rewrite_query": "rewrite_query",
                },
            )
            .add_edge("rewrite_query", "retrieve")
            .add_edge("generate_answer", END)
            .add_edge("handle_failure", END)
        )

        compiled_graph = workflow.compile()
        logger.info("Graph compilation successful")
        return compiled_graph


def build_agentic_rag_graph(
    llm_client: LLMClient,
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    retrieval_settings: RetrievalSettings,
    config: GraphConfig,
    reranker_client: JinaRerankerClient | None = None,
    langfuse_tracer: Optional[LangfuseTracer] = None,
    middleware_manager: Optional[MiddlewareManager] = None,
):
    """Build the LangGraph agentic RAG workflow."""
    return AgenticRAGGraph(
        llm_client=llm_client,
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        retrieval_settings=retrieval_settings,
        config=config,
        reranker_client=reranker_client,
        langfuse_tracer=langfuse_tracer,
    ).compile(middleware_manager=middleware_manager)
