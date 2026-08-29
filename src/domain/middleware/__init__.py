"""Composable middleware for agent invocations.

Usage::

    from src.domain.middleware import (
        AgentContext,
        AgentPipeline,
        ErrorHandlingMiddleware,
        LoggingMiddleware,
    )

    pipeline = AgentPipeline(
        middlewares=[LoggingMiddleware(), ErrorHandlingMiddleware()],
        invoke_fn=agent.core_invoke,
    )
    result = await pipeline.run(ctx)
"""

from .error_handling_middleware import ErrorHandlingMiddleware
from .langfuse_tracing_middleware import LangfuseTracingMiddleware
from .logging_middleware import LoggingMiddleware
from .pipeline import AgentPipeline, MiddlewareManager
from .types import AgentContext, AgentMiddleware, InvokeResult, NextFn, build_invoke_config

__all__ = [
    "AgentContext",
    "AgentMiddleware",
    "AgentPipeline",
    "ErrorHandlingMiddleware",
    "InvokeResult",
    "LangfuseTracingMiddleware",
    "LoggingMiddleware",
    "MiddlewareManager",
    "NextFn",
    "build_invoke_config",
]
