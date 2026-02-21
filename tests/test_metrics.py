"""Unit tests for evaluation metrics: is_valid_sql, execute_query."""

import pytest
from pathlib import Path

from text2sql.evaluation.metrics import execute_query, is_valid_sql


def test_is_valid_sql() -> None:
    assert is_valid_sql("SELECT 1") is True
    assert is_valid_sql("SELECT * FROM t") is True
    assert is_valid_sql("SELECT FROM") is False
    assert is_valid_sql("") is False


def test_execute_query_no_file() -> None:
    ok, rows = execute_query(Path("/nonexistent/db.sqlite"), "SELECT 1")
    assert ok is False
    assert rows is None


def test_execute_query_spider_db() -> None:
    """Run against Spider dev DB if available."""
    root = Path(__file__).resolve().parent.parent
    for base in [root / "data" / "raw", root / "spider_data"]:
        dev_path = base / "dev.json"
        db_dir = base / "database"
        if not dev_path.exists() or not db_dir.exists():
            continue
        import json
        with open(dev_path) as f:
            examples = json.load(f)
        if not examples:
            pytest.skip("No dev examples")
        db_id = examples[0]["db_id"]
        db_path = db_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            pytest.skip("Spider database not found")
        ok, rows = execute_query(db_path, "SELECT 1")
        assert ok is True
        assert rows == [(1,)]
        return
    pytest.skip("Spider data not available")
