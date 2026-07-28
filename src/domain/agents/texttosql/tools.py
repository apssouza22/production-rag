import re
from typing import List, Tuple

from langchain.tools import tool
from langchain_community.tools.sql_database.tool import InfoSQLDatabaseTool, ListSQLDatabaseTool
from langchain_community.utilities import SQLDatabase

_DML_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def _validate_read_only_query(query: str) -> str | None:
    """Return an error message if the query is not read-only."""
    stripped = query.strip().rstrip(";")
    if not stripped:
        return "Error: query is empty."

    if _DML_PATTERN.search(stripped):
        return "Error: only read-only SELECT queries are allowed."

    if not stripped.upper().startswith("SELECT") and not stripped.upper().startswith("WITH"):
        return "Error: only SELECT (or WITH ... SELECT) queries are allowed."

    return None


def create_sql_tools(
    database_url: str,
    include_tables: List[str] | None = None,
    sample_rows_in_table_info: int = 3,
) -> Tuple[List, SQLDatabase]:
    """Create LangChain SQL tools backed by PostgreSQL."""
    db = SQLDatabase.from_uri(
        database_url,
        include_tables=include_tables,
        sample_rows_in_table_info=sample_rows_in_table_info,
    )

    list_tables_tool = ListSQLDatabaseTool(db=db)
    get_schema_tool = InfoSQLDatabaseTool(db=db)

    @tool(
        "sql_db_query",
        description=(
            "Input to this tool is a detailed and correct SQL query, output is a result from the database. "
            "If the query is not correct, an error message will be returned. "
            "If an error is returned, rewrite the query, check the query, and try again. "
            "If you encounter an issue with Unknown column in field list, use sql_db_schema "
            "to query the correct table fields."
        ),
    )
    def sql_db_query(query: str) -> str:
        validation_error = _validate_read_only_query(query)
        if validation_error:
            return validation_error
        return db.run_no_throw(query)

    return [list_tables_tool, get_schema_tool, sql_db_query], db
