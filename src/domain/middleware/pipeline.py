"""Agent pipeline that composes class-based middlewares with lifecycle hooks.

The ``MiddlewareManager`` dispatches calls to every registered middleware
for each lifecycle hook.  ``AgentPipeline`` drives the top-level
``before_invoke → invoke → after_invoke`` flow and exposes the manager
so that graph nodes can trigger ``before/after_model_call`` and
``before/after_tool_call`` hooks.
"""

from typing import Any, Optional, Sequence

from .types import AgentContext, AgentMiddleware, InvokeResult, NextFn


class MiddlewareManager:
    """Dispatches lifecycle hooks to all registered middlewares.

    The manager is stored on the pipeline and made available to graph
    nodes via ``pipeline.manager`` so they can call model/tool hooks.
    """

    def __init__(self, middlewares: Sequence[AgentMiddleware]) -> None:
        self._middlewares = list(middlewares)
        self._active_ctx: Optional[AgentContext] = None

    @property
    def active_ctx(self) -> Optional[AgentContext]:
        """The AgentContext for the currently running invocation."""
        return self._active_ctx

    def set_active_ctx(self, ctx: Optional[AgentContext]) -> None:
        self._active_ctx = ctx

    # -- invoke-level hooks ---------------------------------------------------

    async def run_before_invoke(self, ctx: AgentContext) -> Optional[InvokeResult]:
        """Run ``before_invoke`` on each middleware; short-circuit on first non-None."""
        for mw in self._middlewares:
            result = await mw.before_invoke(ctx)
            if result is not None:
                return result
        return None

    async def run_after_invoke(self, ctx: AgentContext, result: InvokeResult) -> InvokeResult:
        """Run ``after_invoke`` on each middleware in reverse (stack-unwind) order."""
        for mw in reversed(self._middlewares):
            result = await mw.after_invoke(ctx, result)
        return result

    async def run_on_error(self, ctx: AgentContext, error: Exception) -> Optional[InvokeResult]:
        """Run ``on_error`` on each middleware; first non-None result wins."""
        for mw in self._middlewares:
            result = await mw.on_error(ctx, error)
            if result is not None:
                return result
        return None

    # -- model-call hooks -----------------------------------------------------

    async def run_before_model_call(
        self,
        ctx: AgentContext,
        *,
        messages: list,
        model_name: str,
    ) -> list:
        for mw in self._middlewares:
            messages = await mw.before_model_call(ctx, messages=messages, model_name=model_name)
        return messages

    async def run_after_model_call(
        self,
        ctx: AgentContext,
        *,
        response: Any,
        model_name: str,
    ) -> Any:
        for mw in reversed(self._middlewares):
            response = await mw.after_model_call(ctx, response=response, model_name=model_name)
        return response

    # -- tool-call hooks ------------------------------------------------------

    async def run_before_tool_call(
        self,
        ctx: AgentContext,
        *,
        tool_name: str,
        tool_args: dict,
    ) -> dict:
        for mw in self._middlewares:
            tool_args = await mw.before_tool_call(ctx, tool_name=tool_name, tool_args=tool_args)
        return tool_args

    async def run_after_tool_call(
        self,
        ctx: AgentContext,
        *,
        tool_name: str,
        tool_result: Any,
    ) -> Any:
        for mw in reversed(self._middlewares):
            tool_result = await mw.after_tool_call(ctx, tool_name=tool_name, tool_result=tool_result)
        return tool_result


class AgentPipeline:
    """Composable middleware pipeline for agent invocations.

    Usage::

        pipeline = AgentPipeline(
            middlewares=[LoggingMiddleware(), ErrorHandlingMiddleware(), MemoryMiddleware()],
            invoke_fn=agent.core_invoke,
        )
        result = await pipeline.run(ctx)
    """

    def __init__(
        self,
        middlewares: Sequence[AgentMiddleware],
        invoke_fn: NextFn,
    ) -> None:
        self.manager = MiddlewareManager(middlewares)
        self._invoke_fn = invoke_fn

    async def run(self, ctx: AgentContext) -> InvokeResult:
        """Execute the full middleware lifecycle around the core invoke function."""
        self.manager.set_active_ctx(ctx)
        try:
            short_circuit = await self.manager.run_before_invoke(ctx)
            if short_circuit is not None:
                return short_circuit

            try:
                result = await self._invoke_fn(ctx)
            except Exception as e:
                error_result = await self.manager.run_on_error(ctx, e)
                if error_result is not None:
                    return error_result
                raise

            return await self.manager.run_after_invoke(ctx, result)
        finally:
            self.manager.set_active_ctx(None)
