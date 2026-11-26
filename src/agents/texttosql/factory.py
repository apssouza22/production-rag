from pathlib import Path
from typing import Optional

from src.config import get_settings
from src.platform.langfuse.client import LangfuseTracer
from src.platform.llm.chat_model import make_chat_model

from .config import TextToSQLConfig
from .ontology import default_ontology_path, load_schema_ontology
from .service import TextToSQLService


def _resolve_ontology_path(settings_ontology_path: str, override_path: Path | None) -> Path:
    if override_path is not None:
        return override_path
    if settings_ontology_path.strip():
        return Path(settings_ontology_path)
    return default_ontology_path()


def make_text_to_sql_service(
    langfuse_tracer: Optional[LangfuseTracer] = None,
    model: Optional[str] = None,
    top_k: int = 5,
    include_tables: Optional[list[str]] = None,
    ontology_path: Path | None = None,
) -> TextToSQLService:
    """Create a configured TextToSQLService instance."""
    settings = get_settings()
    resolved_ontology_path = _resolve_ontology_path(settings.text_to_sql.ontology_path, ontology_path)
    ontology = load_schema_ontology(resolved_ontology_path)
    tables = include_tables or ontology.table_names()

    agent_config = TextToSQLConfig(
        model=model or settings.agent_model,
        top_k=top_k,
        ontology_path=resolved_ontology_path,
        ontology_prompt=ontology.format_for_prompt(),
        include_tables=tables,
        settings=settings,
    )

    return TextToSQLService(
        chat_model=make_chat_model(settings, agent_config.model, temperature=agent_config.temperature),
        langfuse_tracer=langfuse_tracer,
        agent_config=agent_config,
    )