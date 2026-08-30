from typing import List, Optional

from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.domain.graph.config import FaultToleranceConfig


class TextToSQLConfig(BaseModel):
    """Configuration for the text-to-SQL LangGraph agent."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    top_k: int = 5
    dialect: str = "postgresql"
    include_tables: Optional[List[str]] = Field(default_factory=lambda: ["papers"])
    sample_rows_in_table_info: int = 3
    fault_tolerance: FaultToleranceConfig = Field(default_factory=FaultToleranceConfig)
    settings: Settings = Field(default_factory=get_settings)
