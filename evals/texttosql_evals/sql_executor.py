"""Read-only SQL execution for eval gold and agent queries."""

import re

from langchain_community.utilities import SQLDatabase
from src.agents.texttosql.tools import _validate_read_only_query


def normalize_sql_result(raw: str | None) -> str:
    """Collapse whitespace so LangChain result strings compare deterministically."""
    if not raw:
        return ""
    return " ".join(raw.strip().split())


class SqlResultExecutor:
    """Execute validated read-only SQL against the text-to-SQL database."""

    def __init__(
        self,
        database_url: str,
        *,
        include_tables: list[str] | None = None,
        sample_rows_in_table_info: int = 0,
    ) -> None:
        self.db = SQLDatabase.from_uri(
            database_url,
            include_tables=include_tables,
            sample_rows_in_table_info=sample_rows_in_table_info,
        )

    def execute(self, query: str) -> tuple[str, str | None]:
        """Return normalized result text and an optional error message."""
        validation_error = _validate_read_only_query(query)
        if validation_error:
            return "", validation_error

        raw = self.db.run_no_throw(query)
        if raw is None:
            return "", "Error: query returned no result."

        text = str(raw).strip()
        if not text:
            return "", "Error: query returned an empty result."

        if re.match(r"^Error\b", text, re.IGNORECASE):
            return "", text

        return normalize_sql_result(text), None