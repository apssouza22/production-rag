from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.domain.agents.fault_tolerance import FaultToleranceConfig


class KnowledgeRouterConfig(BaseModel):
    """Configuration for the knowledge router LangGraph agent."""

    model: str = "gpt-4o-mini"
    router_model: str = "gpt-4o-mini"
    temperature: float = 0.0
    fault_tolerance: FaultToleranceConfig = Field(default_factory=FaultToleranceConfig)
    settings: Settings = Field(default_factory=get_settings)
