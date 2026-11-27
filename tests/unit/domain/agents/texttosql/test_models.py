"""Tests for text-to-SQL structured output models."""

import pytest
from pydantic import ValidationError

from src.agents.texttosql.models import SqlGenerationOutput


class TestSqlGenerationOutput:
    def test_accepts_query_only(self):
        out = SqlGenerationOutput(thought="Count rows", query="SELECT 1", final_answer=None)
        assert out.query == "SELECT 1"

    def test_accepts_final_answer_only(self):
        out = SqlGenerationOutput(thought="Already know", query=None, final_answer="42 films")
        assert out.final_answer == "42 films"

    def test_rejects_both_query_and_answer(self):
        with pytest.raises(ValidationError):
            SqlGenerationOutput(thought="x", query="SELECT 1", final_answer="done")

    def test_rejects_neither_query_nor_answer(self):
        with pytest.raises(ValidationError):
            SqlGenerationOutput(thought="x", query=None, final_answer=None)