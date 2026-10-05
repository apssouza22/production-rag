import logging
import time
from typing import Dict, List, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.agents.fusionsearch.citations import answer_indicates_insufficient_information, validate_citations
from src.agents.fusionsearch.handlers import route_agentic_rag_failure
from src.agents.fusionsearch.models import AnswerJudgement, GradeDocuments, GradingResult, SourceItem
from src.agents.fusionsearch.prompts import (
    ABSTENTION_CITATION_MESSAGE,
    ABSTENTION_JUDGE_MESSAGE,
    ABSTENTION_JUDGE_PROMPT,
    ABSTENTION_MAX_ATTEMPTS_MESSAGE,
    ABSTENTION_RETRIEVAL_MESSAGE,
    GENERATE_ANSWER_PROMPT,
    GRADE_DOCUMENTS_PROMPT,
    PARTIAL_ANSWER_NOTE,
    REWRITE_PROMPT,
)
from src.agents.fusionsearch.utils import get_latest_context, get_latest_query, get_latest_retrieval_payload
from src.domain.rerank.service import RerankSearchService
from src.platform.graph import END, START, GraphBuilder, tools_condition
from src.platform.graph.policies import (
    build_llm_timeout,
    build_retry_policy,
    build_tool_retry_policy,
    build_tool_timeout,
)
from src.platform.langfuse.client import LangfuseTracer
from src.platform.llm.chat_model import make_chat_model
from src.platform.middleware import MiddlewareManager

from .config import GraphConfig
from .context import Context
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
        rerank_search_service: RerankSearchService,
        config: GraphConfig,
        langfuse_tracer: Optional[LangfuseTracer] = None,
    ):
        self.rerank_search_service = rerank_search_service
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
            if span:
                self._end_node_span(
                    span,
                    {"status": "max_attempts_reached", "abstained": True},
                    {"execution_time_ms": (time.time() - start_time) * 1000},
                )
            return {
                **updates,
                "messages": [AIMessage(content=ABSTENTION_MAX_ATTEMPTS_MESSAGE)],
                "abstained": True,
                "abstention_reason": "max_retrieval_attempts_exhausted",
                "response_status": "abstained_retrieval",
                "retrieval_confidence": 0.0,
            }

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
            llm = make_chat_model(self.config.settings, self.model_name, temperature=0.0)
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

    async def check_retrieval_score(self, state: AgentState) -> Dict[str, Union[str, float, list, bool]]:
        """Score retrieval confidence and gate low-quality results before grading."""
        logger.info("NODE: assess_retrieval")
        start_time = time.time()
        payload = get_latest_retrieval_payload(state["messages"])
        retrieval_confidence = float(payload.get("retrieval_confidence", 0.0))
        document_count = int(payload.get("document_count", 0))
        raw_sources = payload.get("sources", [])
        relevant_sources = [
            SourceItem(**source) if isinstance(source, dict) else source for source in raw_sources
        ]

        span = self._create_node_span(
            "retrieval_confidence_assessment",
            {
                "retrieval_confidence": retrieval_confidence,
                "document_count": document_count,
                "threshold": self.config.retrieval_confidence_threshold,
            },
            {"node": "assess_retrieval"},
        )

        below_threshold = (
            document_count == 0
            or retrieval_confidence < self.config.retrieval_confidence_threshold
        )
        route = "abstain" if below_threshold else "grade_documents"

        if span:
            self._end_node_span(
                span,
                {
                    "routing_decision": route,
                    "retrieval_confidence": retrieval_confidence,
                    "document_count": document_count,
                    "below_threshold": below_threshold,
                },
                {"execution_time_ms": (time.time() - start_time) * 1000},
            )

        updates: Dict[str, Union[str, float, list, bool]] = {
            "retrieval_confidence": retrieval_confidence,
            "relevant_sources": relevant_sources,
            "routing_decision": route,
        }
        if below_threshold:
            updates.update(
                {
                    "abstained": True,
                    "abstention_reason": "abstained_retrieval",
                    "response_status": "abstained_retrieval",
                }
            )

        return updates

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
            llm = make_chat_model(self.config.settings, self.model_name, temperature=0.3)
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
            llm = make_chat_model(
                self.config.settings,
                self.model_name,
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

    async def validate_answer(self, state: AgentState) -> Dict[str, Union[List[AIMessage], str, bool, dict]]:
        """Validate citations and judge answer grounding before returning to the user."""
        logger.info("NODE: validate_answer")
        start_time = time.time()
        question = get_latest_query(state["messages"])
        context = get_latest_context(state["messages"])
        relevant_sources = state.get("relevant_sources", [])
        answer = self._get_latest_ai_answer(state["messages"])

        span = self._create_node_span(
            "answer_validation",
            {
                "query": question,
                "context_length": len(context),
                "sources_count": len(relevant_sources),
                "enable_citation_validation": self.config.enable_citation_validation,
                "enable_abstention_judge": self.config.enable_abstention_judge,
            },
            {"node": "validate_answer", "model": self.model_name},
        )

        citation_result = None
        if self.config.enable_citation_validation and not answer_indicates_insufficient_information(answer):
            citation_result = validate_citations(answer, relevant_sources)
            if not citation_result.is_trustworthy:
                if span:
                    self._end_node_span(
                        span,
                        {
                            "status": "abstained_citation",
                            "citation_validation": citation_result.model_dump(),
                        },
                        {"execution_time_ms": (time.time() - start_time) * 1000},
                    )
                return {
                    "messages": [AIMessage(content=ABSTENTION_CITATION_MESSAGE)],
                    "citation_validation": citation_result,
                    "abstained": True,
                    "abstention_reason": "abstained_citation",
                    "response_status": "abstained_citation",
                }

        judgement = None
        if self.config.enable_abstention_judge:
            judgement = await self._judge_answer(question, context, answer)
            unsupported = judgement.verdict == "unsupported"
            low_confidence = judgement.confidence < self.config.abstention_judge_threshold
            if unsupported or low_confidence:
                if span:
                    self._end_node_span(
                        span,
                        {
                            "status": "abstained_judge",
                            "answer_judgement": judgement.model_dump(),
                            "citation_validation": citation_result.model_dump() if citation_result else None,
                        },
                        {"execution_time_ms": (time.time() - start_time) * 1000},
                    )
                return {
                    "messages": [AIMessage(content=ABSTENTION_JUDGE_MESSAGE)],
                    "citation_validation": citation_result,
                    "answer_judgement": judgement,
                    "abstained": True,
                    "abstention_reason": "abstained_judge",
                    "response_status": "abstained_judge",
                }

            if judgement.verdict == "partial":
                answer = f"{answer}{PARTIAL_ANSWER_NOTE}"

        if span:
            self._end_node_span(
                span,
                {
                    "status": "answered",
                    "citation_validation": citation_result.model_dump() if citation_result else None,
                    "answer_judgement": judgement.model_dump() if judgement else None,
                },
                {"execution_time_ms": (time.time() - start_time) * 1000},
            )

        updates: Dict[str, Union[List[AIMessage], str, bool, dict]] = {
            "response_status": "answered",
            "abstained": False,
        }
        if citation_result is not None:
            updates["citation_validation"] = citation_result
        if judgement is not None:
            updates["answer_judgement"] = judgement
        if judgement is not None and judgement.verdict == "partial":
            updates["messages"] = [AIMessage(content=answer)]

        return updates

    async def abstain(self, state: AgentState) -> Dict[str, List[AIMessage]]:
        """Return a structured abstention message when confidence checks fail."""
        logger.info("NODE: abstain")
        reason = state.get("abstention_reason") or "abstained_retrieval"
        message_by_reason = {
            "abstained_retrieval": ABSTENTION_RETRIEVAL_MESSAGE,
            "abstained_citation": ABSTENTION_CITATION_MESSAGE,
            "abstained_judge": ABSTENTION_JUDGE_MESSAGE,
            "max_retrieval_attempts_exhausted": ABSTENTION_MAX_ATTEMPTS_MESSAGE,
        }
        message = message_by_reason.get(reason, ABSTENTION_RETRIEVAL_MESSAGE)
        return {
            "messages": [AIMessage(content=message)],
            "abstained": True,
            "abstention_reason": reason,
            "response_status": reason if reason.startswith("abstained") else "abstained_retrieval",
        }

    async def _judge_answer(self, question: str, context: str, answer: str) -> AnswerJudgement:
        """Run the LLM abstention judge on a generated answer."""
        try:
            judge_prompt = ABSTENTION_JUDGE_PROMPT.format(
                question=question,
                context=context,
                answer=answer,
            )
            llm = make_chat_model(self.config.settings, self.model_name, temperature=0.0)
            structured_llm = llm.with_structured_output(AnswerJudgement)
            return await structured_llm.ainvoke(judge_prompt)
        except Exception as exc:
            logger.error("Answer judge failed: %s, defaulting to partial/low-confidence", exc)
            return AnswerJudgement(
                verdict="partial",
                confidence=0.5,
                reasoning=f"Judge fallback after error: {exc}",
            )

    @staticmethod
    def _get_latest_ai_answer(messages: list) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                return message.content if isinstance(message.content, str) else str(message.content)
        return ""

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
        if middleware_manager is not None:
            workflow.set_middleware_manager(middleware_manager)

        retriever_tool = create_retriever_tool(
            rerank_search_service=self.rerank_search_service,
            graph_config=self.config,
        )
        no_fault_tolerance, fault_tolerance = self._configure_fault_tolerance(workflow)

        (
            workflow
            .add_node("retrieve", self.retrieve)
            .add_tool_node("tool_retrieve", [retriever_tool])
            .add_node("check_retrieval_score", self.check_retrieval_score)
            .add_node("grade_documents", self.grade_documents)
            .add_node("rewrite_query", self.rewrite_query)
            .add_node("generate_answer", self.generate_answer)
            .add_node("validate_answer", self.validate_answer)
            .add_node("abstain", self.abstain)
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
            .add_edge("tool_retrieve", "check_retrieval_score")
            .add_conditional_edges(
                "check_retrieval_score",
                lambda state: state.get("routing_decision", "grade_documents"),
                {
                    "grade_documents": "grade_documents",
                    "abstain": "abstain",
                },
            )
            .add_conditional_edges(
                "grade_documents",
                lambda state: state.get("routing_decision", "generate_answer"),
                {
                    "generate_answer": "generate_answer",
                    "rewrite_query": "rewrite_query",
                },
            )
            .add_edge("rewrite_query", "retrieve")
            .add_edge("generate_answer", "validate_answer")
            .add_edge("validate_answer", END)
            .add_edge("abstain", END)
            .add_edge("handle_failure", END)
        )

        compiled_graph = workflow.compile()
        logger.info("Graph compilation successful")
        return compiled_graph

