from dataclasses import dataclass
from typing import Optional


@dataclass
class Context:
    """Per-request runtime context passed through LangGraph."""

    trace_id: Optional[str] = None
