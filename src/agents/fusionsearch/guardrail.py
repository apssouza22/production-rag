"""Guardrail evaluation for fusionsearch agent requests."""

import logging
import time
from typing import Optional

from src.domain.langfuse.client import LangfuseTracer
from src.domain.llm.protocol import LlmProviderClient

from .models import GuardrailScoring
from .prompts import GUARDRAIL_PROMPT

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_MESSAGE = (
    "I apologize, but I can only help with questions about academic research papers "
    "in Computer Science, Artificial Intelligence, and Machine Learning from arXiv.\n\n"
    "Your question: '{question}'\n\n"
    "This appears to be outside my domain of expertise. For questions like this, you might want to try:\n"
    "- General-purpose AI assistants for broad knowledge questions\n"
    "- Domain-specific resources for topics outside CS/AI/ML\n"
    "- Technical documentation if asking about specific software/tools\n\n"
    "If you have a question about AI/ML research papers, I'd be happy to help!"
)


def build_out_of_scope_message(question: str) -> str:
    """Return the user-facing message for out-of-scope queries."""
    return OUT_OF_SCOPE_MESSAGE.format(question=question)


async def evaluate_guardrail(
    query: str,
    llm_client: LlmProviderClient,
    model_name: str,
    threshold: int,
    *,
    langfuse_tracer: Optional[LangfuseTracer] = None,
    trace=None,
) -> GuardrailScoring:
    """Score whether a user query is within the CS/AI/ML research domain."""
    logger.info("guardrail_validation")
    start_time = time.time()

    span = None
    if langfuse_tracer and langfuse_tracer.client and trace is not None:
        try:
            span = langfuse_tracer.create_span(
                trace=trace,
                name="guardrail_validation",
                input_data={"query": query, "threshold": threshold},
                metadata={"middleware": "guardrail", "model": model_name},
            )
        except Exception as exc:
            logger.warning("Failed to create guardrail span: %s", exc)

    try:
        guardrail_prompt = GUARDRAIL_PROMPT.format(question=query)
        llm = llm_client.get_langchain_model(model=model_name, temperature=0.0)
        structured_llm = llm.with_structured_output(GuardrailScoring)
        response = await structured_llm.ainvoke(guardrail_prompt)
        logger.info("Guardrail result - Score: %s, Reason: %s", response.score, response.reason)

        if span:
            langfuse_tracer.end_span(
                span,
                output={
                    "score": response.score,
                    "reason": response.reason,
                    "decision": "continue" if response.score >= threshold else "out_of_scope",
                },
                metadata={
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "threshold": threshold,
                },
            )
    except Exception as exc:
        logger.error("LLM guardrail validation failed: %s, falling back to default", exc)
        response = GuardrailScoring(
            score=50,
            reason=f"LLM validation failed, using conservative default: {exc}",
        )
        if span and langfuse_tracer:
            langfuse_tracer.update_span(
                span,
                output={"score": response.score, "reason": response.reason, "error": str(exc)},
                metadata={"execution_time_ms": (time.time() - start_time) * 1000, "fallback": True},
                level="WARNING",
            )
            langfuse_tracer.end_span(span)

    return response
