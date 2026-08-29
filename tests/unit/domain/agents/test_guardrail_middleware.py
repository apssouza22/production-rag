"""Tests for GuardrailMiddleware."""

import pytest
from unittest.mock import AsyncMock, Mock
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.fusionsearch.config import GraphConfig
from src.agents.fusionsearch.middleware.guardrail_middleware import GuardrailMiddleware
from src.agents.fusionsearch.models import GuardrailScoring
from src.domain.middleware.types import AgentContext


@pytest.fixture
def middleware(mock_ollama_client):
    return GuardrailMiddleware(
        llm_client=mock_ollama_client,
        config=GraphConfig(guardrail_threshold=60),
    )


def _make_ctx(query: str = "What are transformers?") -> AgentContext:
    return AgentContext(
        messages=[HumanMessage(content=query)],
        session_id="test-session",
        user_id=None,
        config={"model": "gpt-4o-mini"},
        agent_name="fusionsearch",
        metadata={},
    )


class TestGuardrailMiddleware:
    @pytest.mark.asyncio
    async def test_before_invoke_passes_when_above_threshold(self, middleware, monkeypatch):
        monkeypatch.setattr(
            "src.agents.fusionsearch.middleware.guardrail_middleware.evaluate_guardrail",
            AsyncMock(return_value=GuardrailScoring(score=85, reason="Relevant")),
        )

        ctx = _make_ctx()
        result = await middleware.before_invoke(ctx)

        assert result is None
        assert ctx.metadata["guardrail_result"].score == 85

    @pytest.mark.asyncio
    async def test_before_invoke_short_circuits_when_below_threshold(self, middleware, monkeypatch):
        monkeypatch.setattr(
            "src.agents.fusionsearch.middleware.guardrail_middleware.evaluate_guardrail",
            AsyncMock(return_value=GuardrailScoring(score=30, reason="Off topic")),
        )

        ctx = _make_ctx("What is a dog?")
        result = await middleware.before_invoke(ctx)

        assert result is not None
        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert "outside my domain" in result[0].content.lower()
        assert ctx.metadata["guardrail_result"].score == 30

    @pytest.mark.asyncio
    async def test_before_invoke_short_circuits_on_llm_failure_default(self, middleware, monkeypatch):
        monkeypatch.setattr(
            "src.agents.fusionsearch.middleware.guardrail_middleware.evaluate_guardrail",
            AsyncMock(return_value=GuardrailScoring(score=50, reason="LLM validation failed")),
        )

        ctx = _make_ctx()
        result = await middleware.before_invoke(ctx)

        assert result is not None
        assert ctx.metadata["guardrail_result"].score == 50
