from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.platform.graph.config import FaultToleranceConfig

from .ontology import default_ontology_path


class TextToSQLConfig(BaseModel):
    """Configuration for the text-to-SQL LangGraph agent."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    top_k: int = 5
    dialect: str = "postgresql"
    ontology_path: Path = Field(default_factory=default_ontology_path)
    ontology_prompt: str = ""
    include_tables: Optional[List[str]] = None
    sample_rows_in_table_info: int = 3
    max_query_loops: int = 5
    fault_tolerance: FaultToleranceConfig = Field(default_factory=FaultToleranceConfig)
    settings: Settings = Field(default_factory=get_settings)
