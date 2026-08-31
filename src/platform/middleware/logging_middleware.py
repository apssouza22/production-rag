"""Middleware that adds structured logging around agent invocations."""
import logging
from typing import Any, Optional

from .types import AgentContext, AgentMiddleware, InvokeResult

logger = logging.getLogger(__name__)


class LoggingMiddleware(AgentMiddleware):
    """Logs agent lifecycle events: invoke, model calls, and tool calls."""

    async def before_invoke(self, ctx: AgentContext) -> Optional[InvokeResult]:
        logger.info("agent_invoke_started %s", ctx.agent_name)
        return None

    async def after_invoke(self, ctx: AgentContext, result: InvokeResult) -> InvokeResult:
        logger.info("agent_invoke_completed %s", ctx.agent_name)
        return result

    async def before_model_call(
        self,
        ctx: AgentContext,
        *,
        messages: list,
        model_name: str,
    ) -> list:
        logger.debug("model_call_started %s %s %s", ctx.agent_name, ctx.session_id, model_name)
        return messages

    async def after_model_call(
        self,
        ctx: AgentContext,
        *,
        response: Any,
        model_name: str,
    ) -> Any:
        logger.debug("model_call_finished %s %s %s", ctx.agent_name, ctx.session_id, model_name)
        return response

    async def before_tool_call(
        self,
        ctx: AgentContext,
        *,
        tool_name: str,
        tool_args: dict,
    ) -> dict:
        logger.info( "tool_call_started %s %s %s", ctx.agent_name, ctx.session_id, tool_name)
        return tool_args

    async def after_tool_call(
        self,
        ctx: AgentContext,
        *,
        tool_name: str,
        tool_result: Any,
    ) -> Any:
        logger.info("tool_call_finished %s %s %s", ctx.agent_name, ctx.session_id, tool_name)
        return tool_result
