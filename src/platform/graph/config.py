from pydantic import BaseModel, Field

from src.config import Settings, get_settings


class FaultToleranceConfig(BaseModel):
    """LangGraph fault-tolerance settings (retries, timeouts, error handlers).

  Based on LangGraph RetryPolicy, TimeoutPolicy, and error_handler primitives.
  See: https://www.langchain.com/blog/fault-tolerance-in-langgraph
  """

    enabled: bool = True
    max_attempts: int = 3
    initial_interval: float = 0.5
    backoff_factor: float = 2.0
    max_interval: float = 128.0
    jitter: bool = True # jitter adds randomness to retry intervals to avoid thundering herd effect
    llm_run_timeout: float = 120.0
    llm_idle_timeout: float = 30.0
    tool_run_timeout: float = 60.0
    tool_idle_timeout: float = 15.0
    settings: Settings = Field(default_factory=get_settings)
