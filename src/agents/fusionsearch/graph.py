import logging

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.domain.agent_fault_tolerance import (
    build_llm_timeout,
    build_retry_policy,
    build_tool_retry_policy,
    build_tool_timeout,
)
from src.domain.jinaai.jina_client import JinaEmbeddingsClient
from src.domain.jinaai.jina_reranker_client import JinaRerankerClient
from src.domain.opensearch.client import OpenSearchClient
from src.agents.fusionsearch.handlers import route_agentic_rag_failure
from src.agents.fusionsearch.nodes import (
    ainvoke_generate_answer_step,
    ainvoke_grade_documents_step,
    ainvoke_guardrail_step,
    ainvoke_handle_failure_step,
    ainvoke_out_of_scope_step,
    ainvoke_retrieve_step,
    ainvoke_rewrite_query_step,
    continue_after_guardrail,
)

from .config import GraphConfig
from .context import Context
from .retrieval_settings import RetrievalSettings
from .state import AgentState
from .tools import create_retriever_tool

logger = logging.getLogger(__name__)


class AgenticRAGGraph:
    """Builds and compiles the LangGraph agentic RAG workflow."""

    def __init__(
        self,
        opensearch_client: OpenSearchClient,
        embeddings_client: JinaEmbeddingsClient,
        retrieval_settings: RetrievalSettings,
        config: GraphConfig,
        reranker_client: JinaRerankerClient | None = None,
    ):
        self.opensearch_client = opensearch_client
        self.embeddings_client = embeddings_client
        self.reranker_client = reranker_client
        self.retrieval_settings = retrieval_settings
        self.config = config

    def _configure_fault_tolerance(self, workflow: StateGraph) -> tuple[dict, dict]:
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

    def compile(self):
        """Build and compile the LangGraph workflow."""
        logger.info("Building LangGraph workflow with context_schema")

        workflow = StateGraph(AgentState, context_schema=Context)

        retriever_tool = create_retriever_tool(
            opensearch_client=self.opensearch_client,
            embeddings_client=self.embeddings_client,
            retrieval_settings=self.retrieval_settings,
            reranker_client=self.reranker_client,
        )
        tool_retrieve = [retriever_tool]
        no_fault_tolerance, fault_tolerance = self._configure_fault_tolerance(workflow)

        logger.info("Adding nodes to workflow graph")
        workflow.add_node("guardrail", ainvoke_guardrail_step)
        workflow.add_node("out_of_scope", ainvoke_out_of_scope_step, **no_fault_tolerance)
        workflow.add_node("retrieve", ainvoke_retrieve_step)
        workflow.add_node("tool_retrieve", ToolNode(tool_retrieve), **fault_tolerance)
        workflow.add_node("grade_documents", ainvoke_grade_documents_step)
        workflow.add_node("rewrite_query", ainvoke_rewrite_query_step)
        workflow.add_node("generate_answer", ainvoke_generate_answer_step)
        workflow.add_node("handle_failure", ainvoke_handle_failure_step, **no_fault_tolerance)

        logger.info("Configuring graph edges and routing logic")
        workflow.add_edge(START, "guardrail")
        workflow.add_conditional_edges(
            "guardrail",
            continue_after_guardrail,
            {
                "continue": "retrieve",
                "out_of_scope": "out_of_scope",
            },
        )
        workflow.add_edge("out_of_scope", END)
        workflow.add_conditional_edges(
            "retrieve",
            tools_condition,
            {
                "tools": "tool_retrieve",
                END: END,
            },
        )
        workflow.add_edge("tool_retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            lambda state: state.get("routing_decision", "generate_answer"),
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
            },
        )
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("handle_failure", END)

        logger.info("Compiling LangGraph workflow")
        compiled_graph = workflow.compile()
        logger.info("✓ Graph compilation successful")
        return compiled_graph


def build_agentic_rag_graph(
    opensearch_client: OpenSearchClient,
    embeddings_client: JinaEmbeddingsClient,
    retrieval_settings: RetrievalSettings,
    config: GraphConfig,
    reranker_client: JinaRerankerClient | None = None,
):
    """Build the LangGraph agentic RAG workflow."""
    return AgenticRAGGraph(
        opensearch_client=opensearch_client,
        embeddings_client=embeddings_client,
        retrieval_settings=retrieval_settings,
        config=config,
        reranker_client=reranker_client,
    ).compile()
