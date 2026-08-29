"""Guardrail middleware that validates query scope before graph execution."""

import logging
from typing import Optional

from langchain_core.messages import AIMessage

from src.domain.langfuse.client import LangfuseTracer
from src.domain.llm.protocol import LLMClient
from src.domain.middleware.types import AgentContext, AgentMiddleware, InvokeResult

from ..config import GraphConfig
from ..guardrail import build_out_of_scope_message, evaluate_guardrail
from ..utils import get_latest_query

logger = logging.getLogger(__name__)


class GuardrailMiddleware(AgentMiddleware):
    """Runs guardrail validation in ``before_invoke``.

    When the query scores below the configured threshold the middleware
    short-circuits the pipeline and returns an out-of-scope response,
    skipping the LangGraph workflow entirely.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: GraphConfig,
        langfuse_tracer: Optional[LangfuseTracer] = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config
        self.langfuse_tracer = langfuse_tracer

    async def before_invoke(self, ctx: AgentContext) -> Optional[InvokeResult]:
        query = get_latest_query(ctx.messages)
        model_name = ctx.config.get("model", self.config.model)
        trace = ctx.metadata.get("trace")

        result = await evaluate_guardrail(
            query=query,
            llm_client=self.llm_client,
            model_name=model_name,
            threshold=self.config.guardrail_threshold,
            langfuse_tracer=self.langfuse_tracer,
            trace=trace,
        )
        ctx.metadata["guardrail_result"] = result

        if result.score < self.config.guardrail_threshold:
            logger.info(
                "Guardrail rejected query (score=%s, threshold=%s)",
                result.score,
                self.config.guardrail_threshold,
            )
            return [AIMessage(content=build_out_of_scope_message(query))]

        logger.info(
            "Guardrail passed (score=%s, threshold=%s)",
            result.score,
            self.config.guardrail_threshold,
        )
        return None
