"""Build evaluation samples by calling TextToSQLService and executing SQL."""

import json
import logging
from pathlib import Path
from typing import Protocol

from evals.core.schemas import EvalSample
from evals.texttosql_evals.config import TextToSQLEvalSettings
from evals.texttosql_evals.schemas import EvalQuery
from evals.texttosql_evals.sql_executor import SqlResultExecutor

logger = logging.getLogger(__name__)


class TextToSQLServiceProtocol(Protocol):
    async def ask(self, query: str, user_id: str = "api_user", model: str | None = None) -> dict: ...


def load_queries(dataset_path: Path) -> list[EvalQuery]:
    """Load evaluation queries from a JSON file."""
    with dataset_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("Dataset file must contain a JSON list of query objects")

    return [EvalQuery.model_validate(item) for item in payload]


class DatasetBuilder:
    """Populate EvalSample rows from the agent and SQL execution results."""

    def __init__(
        self,
        settings: TextToSQLEvalSettings,
        service: TextToSQLServiceProtocol,
        sql_executor: SqlResultExecutor,
    ) -> None:
        self.settings = settings
        self.service = service
        self.sql_executor = sql_executor

    async def build_sample_from_query(self, query: EvalQuery) -> EvalSample | None:
        """Call the agent, execute SQL, and prepare one normalized sample."""
        reference, reference_error = self.sql_executor.execute(query.reference_sql)
        if reference_error:
            logger.warning("Skipping query with invalid reference SQL (%s): %s", query.id or query.user_input, reference_error)
            return None

        model_override = self.settings.text_to_sql_eval_model or None
        agent_response = await self.service.ask(
            query=query.user_input,
            user_id=self.settings.text_to_sql_eval_user_id,
            model=model_override,
        )

        sql_queries: list[str] = list(agent_response.get("sql_queries") or [])
        agent_sql = sql_queries[-1] if sql_queries else None
        agent_output = ""
        agent_sql_error: str | None = None
        if agent_sql:
            agent_output, agent_sql_error = self.sql_executor.execute(agent_sql)

        return EvalSample(
            input=query.user_input,
            reference=reference,
            output=agent_output,
            source_id=query.id,
            metadata={
                "reference_sql": query.reference_sql,
                "agent_sql": agent_sql,
                "sql_queries": sql_queries,
                "natural_language_answer": agent_response.get("answer", ""),
                "agent_sql_error": agent_sql_error,
                "reference_sql_error": reference_error,
                "trace_id": agent_response.get("trace_id"),
            },
        )

    async def build_samples(self, queries: list[EvalQuery]) -> list[EvalSample]:
        """Build samples for each query, skipping those with invalid reference SQL."""
        samples: list[EvalSample] = []
        for query in queries:
            sample = await self.build_sample_from_query(query)
            if sample is not None:
                samples.append(sample)
        return samples


class TextToSQLDataset:
    """Load a local query file and enrich it with agent outputs and SQL results."""

    source_name = "json_texttosql"

    def __init__(self, dataset_path: Path, builder: DatasetBuilder) -> None:
        self.dataset_path = dataset_path
        self.builder = builder
        self.total_items = 0

    async def load(self) -> list[EvalSample]:
        queries = load_queries(self.dataset_path)
        self.total_items = len(queries)
        return await self.builder.build_samples(queries)