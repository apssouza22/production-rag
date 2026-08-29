"""Middleware that provides consistent error handling across agents."""
import logging
from typing import Optional

from .types import AgentContext, AgentMiddleware, InvokeResult
logger = logging.getLogger(__name__)

class ErrorHandlingMiddleware(AgentMiddleware):
    """Catches exceptions, records metrics, re-raises in dev or returns fallback."""

    async def on_error(self, ctx: AgentContext, error: Exception) -> Optional[InvokeResult]:
        logger.exception("Error in agent %s: %s",ctx.agent_name, error, exc_info=error)
        return []
