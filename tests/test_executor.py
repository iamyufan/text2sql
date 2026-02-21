"""Unit tests for SQL execution and execution-equivalence (executor)."""

import json
import pytest
from pathlib import Path

from text2sql.evaluation.executor import exec_match_one
from text2sql.evaluation.spider import build_foreign_key_map_from_json
from text2sql.evaluation.difficulty import get_hardness_and_parsed


def _data_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if (root / "data" / "raw" / "tables.json").exists():
        return root / "data" / "raw"
    if (root / "spider_data" / "tables.json").exists():
        return root / "spider_data"
    return None


@pytest.fixture
def spider_available() -> bool:
    root = _data_root()
    if root is None:
        return False
    dev_path = root / "dev.json"
    tables_path = root / "tables.json"
    db_dir = root / "database"
    if not (dev_path.exists() and tables_path.exists() and db_dir.exists()):
        return False
    with open(dev_path) as f:
        examples = json.load(f)
    if not examples:
        return False
    db_id = examples[0]["db_id"]
    return (db_dir / db_id / f"{db_id}.sqlite").exists()


def test_exec_match_one_trivial(spider_available: bool) -> None:
    """exec_match_one: same SQL should be execution-equivalent."""
    if not spider_available:
        pytest.skip("Spider data not available")
    root = _data_root()
    with open(root / "dev.json") as f:
        examples = json.load(f)
    with open(root / "tables.json") as f:
        tables_list = json.load(f)
    kmaps = build_foreign_key_map_from_json(str(root / "tables.json"))
    ex = examples[0]
    db_path = root / "database" / ex["db_id"] / f"{ex['db_id']}.sqlite"
    gold_str = ex["query"]
    _, g_sql = get_hardness_and_parsed(db_path, gold_str)
    kmap = kmaps.get(ex["db_id"], {})
    assert exec_match_one(db_path, gold_str, gold_str, g_sql, kmap) is True
