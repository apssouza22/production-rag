from dataclasses import dataclass
from typing import Optional

from langfuse._client.span import LangfuseSpan

from src.domain.langfuse.client import LangfuseTracer
from src.domain.llm.protocol import LLMClient


@dataclass
class Context:
    """Runtime context for per-request graph execution.

    Holds dependencies that nodes read at runtime. Retrieval clients
    (OpenSearch, embeddings) are bound into the retriever tool at graph
    compile time and do not belong here.
    """

    llm_client: LLMClient
    langfuse_tracer: Optional[LangfuseTracer]
    trace: Optional["LangfuseSpan"] = None
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 60

    @property
    def tracing_enabled(self) -> bool:
        return (
            self.trace is not None
            and self.langfuse_tracer is not None
            and self.langfuse_tracer.client is not None
        )
