"""Middleware that captures graph execution trajectories via LangChain callbacks."""

from __future__ import annotations

from typing import Optional

from src.platform.graph.callback_utils import extend_graph_callbacks
from src.platform.tracing.trajectory import TrajectoryCallback

from .types import AgentContext, AgentMiddleware, InvokeResult


class TrajectoryMiddleware(AgentMiddleware):
    """Attach a ``TrajectoryCallback`` before graph invoke and store the result."""

    def __init__(self, *, max_content_length: int = 500) -> None:
        self.max_content_length = max_content_length
        self._callback: Optional[TrajectoryCallback] = None

    @property
    def callback(self) -> Optional[TrajectoryCallback]:
        return self._callback

    async def before_invoke(self, ctx: AgentContext) -> Optional[InvokeResult]:
        self._callback = TrajectoryCallback(max_content_length=self.max_content_length)
        extend_graph_callbacks(ctx, [self._callback])
        return None

    async def after_invoke(self, ctx: AgentContext, result: InvokeResult) -> InvokeResult:
        if self._callback is not None:
            ctx.metadata["trajectory"] = self._callback.finalize()
        return result

    async def on_error(self, ctx: AgentContext, error: Exception) -> Optional[InvokeResult]:
        if self._callback is not None:
            from src.platform.tracing.trajectory import TrajectoryEvent

            trajectory = self._callback.finalize()
            trajectory.events.append(
                TrajectoryEvent(
                    event_type="invoke_error",
                    name=ctx.agent_name,
                    run_id="invoke",
                    error=str(error),
                )
            )
            ctx.metadata["trajectory"] = trajectory
        return None
