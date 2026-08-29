import logging

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from src.domain.agent_fault_tolerance import (
    build_llm_timeout,
    build_retry_policy,
)
from src.agents.knowledgerouter.handlers import knowledge_router_error_handler
from src.agents.fusionsearch.agentic_rag import AgenticRAGService
from src.agents.texttosql.service import TextToSQLService

from .config import KnowledgeRouterConfig
from .prompts import CLASSIFY_SYSTEM_PROMPT, SYNTHESIZE_SYSTEM_PROMPT
from .schemas import ClassificationItem
from .state import AgentInput, RouterState

logger = logging.getLogger(__name__)


class ClassificationResult(BaseModel):
    """Structured output schema for the classifier."""

    classifications: list[ClassificationItem] = Field(
        description="List of agents to invoke with their targeted sub-questions",
    )


class KnowledgeRouterGraph:
    """Builds and compiles the LangGraph knowledge router workflow."""

    def __init__(
        self,
        router_model: BaseChatModel,
        synthesis_model: BaseChatModel,
        agentic_rag_service: AgenticRAGService,
        text_to_sql_service: TextToSQLService,
        config: KnowledgeRouterConfig,
    ):
        self.router_model = router_model
        self.synthesis_model = synthesis_model
        self.agentic_rag_service = agentic_rag_service
        self.text_to_sql_service = text_to_sql_service
        self.config = config

    async def classify_query(self, state: RouterState) -> dict:
        """Classify query and determine which agents to invoke."""
        structured_llm = self.router_model.with_structured_output(ClassificationResult)

        result = await structured_llm.ainvoke(
            [
                {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": state["query"]},
            ]
        )

        classifications = [
            {"source": item.source, "query": item.query}
            for item in result.classifications
        ]
        if not classifications:
            logger.info("No classifications returned, defaulting to documents source")
            classifications = [{"source": "documents", "query": state["query"]}]

        logger.info(
            "Classified query into %d source(s): %s",
            len(classifications),
            [item["source"] for item in classifications],
        )
        return {"classifications": classifications}

    def route_to_agents(self, state: RouterState) -> list[Send]:
        """Fan out to agents based on classifications."""
        return [
            Send(item["source"], {"query": item["query"]})
            for item in state["classifications"]
        ]

    async def query_documents(self, state: AgentInput) -> dict:
        """Query the agentic RAG service for paper content."""
        logger.info("Querying documents agent: %s", state["query"][:100])
        result = await self.agentic_rag_service.ask(query=state["query"])
        sources = result.get("sources", [])
        source_urls = [
            source.get("url", source) if isinstance(source, dict) else str(source)
            for source in sources
        ]
        return {
            "results": [
                {
                    "source": "documents",
                    "result": result.get("answer", "No answer generated."),
                    "metadata": {
                        "sources": source_urls,
                        "retrieval_attempts": result.get("retrieval_attempts", 0),
                        "reasoning_steps": result.get("reasoning_steps", []),
                    },
                }
            ]
        }

    async def query_database(self, state: AgentInput) -> dict:
        """Query the text-to-SQL service for structured metadata."""
        logger.info("Querying database agent: %s", state["query"][:100])
        result = await self.text_to_sql_service.ask(query=state["query"])
        return {
            "results": [
                {
                    "source": "database",
                    "result": result.get("answer", "No answer generated."),
                    "metadata": {
                        "sql_queries": result.get("sql_queries", []),
                        "reasoning_steps": result.get("reasoning_steps", []),
                    },
                }
            ]
        }

    async def synthesize_results(self, state: RouterState) -> dict:
        """Combine results from all agents into a coherent answer."""
        if not state["results"]:
            return {"final_answer": "No results found from any knowledge source."}

        if len(state["results"]) == 1:
            return {"final_answer": state["results"][0]["result"]}

        formatted = [
            f"**From {item['source'].title()}:**\n{item['result']}"
            for item in state["results"]
        ]

        synthesis_response = await self.synthesis_model.ainvoke(
            [
                {
                    "role": "system",
                    "content": SYNTHESIZE_SYSTEM_PROMPT.format(query=state["query"]),
                },
                {"role": "user", "content": "\n\n".join(formatted)},
            ]
        )

        content = (
            synthesis_response.content
            if hasattr(synthesis_response, "content")
            else str(synthesis_response)
        )
        return {"final_answer": content}

    def _configure_fault_tolerance(self, workflow: StateGraph) -> None:
        ft = self.config.fault_tolerance
        if ft.enabled:
            workflow.set_node_defaults(
                retry_policy=build_retry_policy(ft),
                timeout=build_llm_timeout(ft),
                error_handler=knowledge_router_error_handler,
            )

    def compile(self):
        workflow = StateGraph(RouterState)
        self._configure_fault_tolerance(workflow)

        (
            workflow
            .add_node("classify", self.classify_query)
            .add_node("documents", self.query_documents)
            .add_node("database", self.query_database)
            .add_node("synthesize", self.synthesize_results)
            .add_edge(START, "classify")
            .add_conditional_edges("classify", self.route_to_agents, ["documents", "database"])
            .add_edge("documents", "synthesize")
            .add_edge("database", "synthesize")
            .add_edge("synthesize", END)
        )

        logger.info(
            "Knowledge router graph compiled successfully (model=%s)",
            self.config.model,
        )
        return workflow.compile()


def build_knowledge_router_graph(
    router_model: BaseChatModel,
    synthesis_model: BaseChatModel,
    agentic_rag_service: AgenticRAGService,
    text_to_sql_service: TextToSQLService,
    config: KnowledgeRouterConfig,
):
    """Build the LangGraph knowledge router workflow."""
    return KnowledgeRouterGraph(
        router_model=router_model,
        synthesis_model=synthesis_model,
        agentic_rag_service=agentic_rag_service,
        text_to_sql_service=text_to_sql_service,
        config=config,
    ).compile()
