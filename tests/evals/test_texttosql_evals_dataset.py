"""Tests for text-to-SQL dataset loading and sample building."""

import json
from unittest.mock import AsyncMock, Mock

import pytest
from evals.texttosql_evals.config import TextToSQLEvalSettings
from evals.texttosql_evals.dataset import DatasetBuilder, load_queries
from evals.texttosql_evals.schemas import EvalQuery
from pydantic import ValidationError


@pytest.mark.asyncio
async def test_dataset_builder_maps_service_response(tmp_path) -> None:
    settings = TextToSQLEvalSettings()
    service = AsyncMock()
    service.ask = AsyncMock(
        return_value={
            "answer": "There are 1000 films.",
            "sql_queries": ["SELECT COUNT(*) FROM film", "SELECT COUNT(*)::int FROM film"],
            "trace_id": "trace-1",
        }
    )
    sql_executor = Mock()
    sql_executor.execute = Mock(
        side_effect=[
            ("[(1000,)]", None),
            ("[(1000,)]", None),
        ]
    )

    builder = DatasetBuilder(settings=settings, service=service, sql_executor=sql_executor)
    sample = await builder.build_sample_from_query(
        EvalQuery(user_input="How many films?", reference_sql="SELECT COUNT(*) FROM film")
    )

    assert sample is not None
    assert sample.input == "How many films?"
    assert sample.reference == "[(1000,)]"
    assert sample.output == "[(1000,)]"
    assert sample.metadata["agent_sql"] == "SELECT COUNT(*)::int FROM film"
    assert sample.metadata["sql_queries"] == [
        "SELECT COUNT(*) FROM film",
        "SELECT COUNT(*)::int FROM film",
    ]
    service.ask.assert_awaited_once_with(
        query="How many films?",
        user_id=settings.text_to_sql_eval_user_id,
        model=None,
    )
    assert sql_executor.execute.call_args_list[-1].args[0] == "SELECT COUNT(*)::int FROM film"


def test_load_queries_rejects_non_list(tmp_path) -> None:
    dataset_path = tmp_path / "bad.json"
    dataset_path.write_text(json.dumps({"user_input": "x", "reference_sql": "SELECT 1"}))
    with pytest.raises(ValueError, match="JSON list"):
        load_queries(dataset_path)


def test_load_queries_rejects_missing_reference_sql(tmp_path) -> None:
    dataset_path = tmp_path / "missing_sql.json"
    dataset_path.write_text(json.dumps([{"user_input": "How many films?"}]))
    with pytest.raises(ValidationError):
        load_queries(dataset_path)