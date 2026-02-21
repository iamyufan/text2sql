"""Tests for schema serialization (CREATE TABLE format)."""

import json
import pytest
from pathlib import Path

from text2sql.data import (
    serialize_schema,
    get_schema_for_db,
    serialize_schema_for_db,
    SPIDER_TYPE_TO_SQL,
)


def _tables_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    data_raw = root / "data" / "raw" / "tables.json"
    if data_raw.exists():
        return data_raw
    return root / "spider_data" / "tables.json"


@pytest.fixture
def tables_list() -> list:
    with open(_tables_path()) as f:
        return json.load(f)


@pytest.fixture
def perpetrator_schema(tables_list: list) -> dict:
    s = get_schema_for_db("perpetrator", tables_list)
    assert s is not None
    return s


def test_serialize_schema_output_contains_create_table(perpetrator_schema: dict) -> None:
    out = serialize_schema(perpetrator_schema)
    assert "CREATE TABLE" in out
    assert out.strip()


def test_serialize_schema_all_table_names_original_appear(perpetrator_schema: dict) -> None:
    out = serialize_schema(perpetrator_schema)
    for t in perpetrator_schema["table_names_original"]:
        assert t in out, f"table {t!r} not in output"


def test_serialize_schema_no_star_column_definition(perpetrator_schema: dict) -> None:
    out = serialize_schema(perpetrator_schema)
    # The (-1, "*") entry should not appear as a column definition (e.g. "* INT")
    assert " * " not in out or "(*)" in out
    assert "*, " not in out


def test_serialize_schema_primary_keys_present(perpetrator_schema: dict) -> None:
    out = serialize_schema(perpetrator_schema)
    assert "PRIMARY KEY" in out


def test_serialize_schema_foreign_keys_present(perpetrator_schema: dict) -> None:
    out = serialize_schema(perpetrator_schema)
    assert "FOREIGN KEY" in out
    assert "REFERENCES" in out


def test_get_schema_for_db_found(tables_list: list) -> None:
    s = get_schema_for_db("perpetrator", tables_list)
    assert s is not None
    assert s["db_id"] == "perpetrator"


def test_get_schema_for_db_not_found(tables_list: list) -> None:
    s = get_schema_for_db("nonexistent_db_xyz", tables_list)
    assert s is None


def test_serialize_schema_for_db_success(tables_list: list) -> None:
    out = serialize_schema_for_db("perpetrator", tables_list)
    assert "CREATE TABLE perpetrator" in out
    assert "CREATE TABLE people" in out


def test_serialize_schema_for_db_raises_when_missing(tables_list: list) -> None:
    with pytest.raises(KeyError, match="nonexistent_db_xyz"):
        serialize_schema_for_db("nonexistent_db_xyz", tables_list)


def test_concert_singer_format(tables_list: list) -> None:
    """Spot-check concert_singer: expected tables and CREATE TABLE format."""
    out = serialize_schema_for_db("concert_singer", tables_list)
    assert "CREATE TABLE singer" in out
    assert "CREATE TABLE concert" in out
    assert "Singer_ID" in out
    assert "INT" in out
    assert "TEXT" in out
    assert "PRIMARY KEY" in out


def test_type_mapping_covered() -> None:
    for k in ("text", "number", "time", "boolean", "others"):
        assert k in SPIDER_TYPE_TO_SQL
        assert SPIDER_TYPE_TO_SQL[k] in ("TEXT", "INT")
