"""Core evaluation metrics: valid SQL (sqlglot), execute_query, and helpers."""

import sqlite3
from pathlib import Path

import sqlglot


def _sqlite_connect(db_path):
    """Open SQLite connection with UTF-8-safe text decoding for DBs with non-UTF-8 data."""
    conn = sqlite3.connect(str(db_path))
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b
    return conn


def is_valid_sql(sql: str) -> bool:
    """Return True if the string parses as valid SQL (sqlglot), False otherwise."""
    if not sql or not sql.strip():
        return False
    try:
        sqlglot.parse_one(sql, dialect="sqlite")
        return True
    except Exception:
        return False


def execute_query(db_path: Path, sql: str) -> tuple[bool, list | None]:
    """Execute SQL against the SQLite database at db_path.

    Returns (True, rows) on success, (False, None) on any exception.
    rows is the list of result tuples from fetchall().
    """
    if not db_path.exists():
        return False, None
    try:
        conn = _sqlite_connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return True, rows
    except Exception:
        return False, None
