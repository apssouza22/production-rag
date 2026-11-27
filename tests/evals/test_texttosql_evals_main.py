"""Tests for text-to-SQL eval CLI exit codes."""

from unittest.mock import AsyncMock, Mock

import pytest
from evals.texttosql_evals.main import run_evaluation


@pytest.mark.asyncio
async def test_run_evaluation_exits_one_on_failure(monkeypatch) -> None:
    runner = Mock()
    runner.run_eval = AsyncMock(side_effect=RuntimeError("No evaluation samples"))
    monkeypatch.setattr(
        "evals.texttosql_evals.main.TextToSQLEvaluationRunner",
        Mock(return_value=runner),
    )
    monkeypatch.setattr(
        "evals.texttosql_evals.main.make_report_publisher",
        Mock(return_value=Mock()),
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluation()

    assert exc_info.value.code == 1