"""End-to-end tests for the evaluation pipeline with trivial baseline."""

import json
import pytest
from pathlib import Path

from text2sql.evaluation import run_eval, is_valid_sql, classify_error
from text2sql.evaluation.metrics import execute_query
from text2sql.data import get_schema_for_db


def _spider_data_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if (root / "data" / "raw" / "tables.json").exists():
        return root / "data" / "raw"
    return root / "spider_data"


def _dev_path() -> Path:
    return _spider_data_root() / "dev.json"


def _tables_path() -> Path:
    return _spider_data_root() / "tables.json"


def _db_dir() -> Path:
    return _spider_data_root() / "database"


@pytest.fixture
def spider_available() -> bool:
    """True if Spider dev data and DBs are present."""
    root = _spider_data_root()
    if not (_dev_path().exists() and _tables_path().exists() and _db_dir().exists()):
        return False
    with open(_dev_path()) as f:
        examples = json.load(f)
    if not examples:
        return False
    db_id = examples[0]["db_id"]
    db_path = _db_dir() / db_id / f"{db_id}.sqlite"
    return db_path.exists()


def trivial_baseline(examples: list[dict], tables_list: list[dict]) -> list[str]:
    """SELECT * FROM first_table per example."""
    preds = []
    for ex in examples:
        schema = get_schema_for_db(ex["db_id"], tables_list)
        if schema and schema.get("table_names_original"):
            first_table = schema["table_names_original"][0]
            preds.append(f"SELECT * FROM {first_table}")
        else:
            preds.append("SELECT 1")
    return preds


def test_is_valid_sql() -> None:
    assert is_valid_sql("SELECT 1") is True
    assert is_valid_sql("SELECT * FROM t") is True
    assert is_valid_sql("SELECT FROM") is False
    assert is_valid_sql("") is False


def test_execute_query(spider_available: bool) -> None:
    if not spider_available:
        pytest.skip("Spider data not available")
    db_dir = _db_dir()
    with open(_dev_path()) as f:
        examples = json.load(f)
    db_id = examples[0]["db_id"]
    db_path = db_dir / db_id / f"{db_id}.sqlite"
    ok, rows = execute_query(db_path, "SELECT 1")
    assert ok is True
    assert rows == [(1,)]


def test_eval_pipeline_trivial_baseline(spider_available: bool, tmp_path: Path) -> None:
    if not spider_available:
        pytest.skip("Spider data not available")
    with open(_dev_path()) as f:
        examples = json.load(f)
    with open(_tables_path()) as f:
        tables_list = json.load(f)
    subset = examples[:5]
    predictions = trivial_baseline(subset, tables_list)

    result = run_eval(
        predictions=predictions,
        examples=subset,
        db_dir=_db_dir(),
        tables_path=_tables_path(),
        output_dir=tmp_path,
    )

    assert "valid_sql_rate" in result
    assert "execution_accuracy" in result
    assert "exact_match_accuracy" in result
    assert "valid_sql_by_difficulty" in result
    assert "execution_by_difficulty" in result
    assert "exact_match_by_difficulty" in result
    assert "failures" in result
    assert "failure_analysis_path" in result

    assert result["valid_sql_rate"] == 1.0

    for h in ["easy", "medium", "hard", "extra"]:
        assert h in result["valid_sql_by_difficulty"]
        assert h in result["execution_by_difficulty"]
        assert h in result["exact_match_by_difficulty"]

    assert (tmp_path / "failure_analysis.json").exists()
    with open(tmp_path / "failure_analysis.json") as f:
        failures = json.load(f)
    for item in failures:
        assert "question" in item
        assert "gold_sql" in item
        assert "pred_sql" in item
        assert "db_id" in item
        assert "difficulty" in item
        assert "is_valid_sql" in item
        assert "error_type" in item


def test_classify_error_invalid_syntax(tmp_path: Path) -> None:
    if not _tables_path().exists():
        pytest.skip("Spider tables.json not available")
    with open(_tables_path()) as f:
        tables_list = json.load(f)
    with open(_dev_path()) as f:
        ex = json.load(f)[0]
    dummy_db = tmp_path / "dummy.sqlite"
    err = classify_error("SELECT FROM", ex["query"], ex["db_id"], tables_list, dummy_db)
    assert err == "invalid_syntax"
