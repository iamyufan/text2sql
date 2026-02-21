"""Failure logging and error classification."""

from pathlib import Path

import sqlglot
from sqlglot import exp

from text2sql.data import get_schema_for_db
from text2sql.evaluation.metrics import is_valid_sql

ERROR_TYPES = (
    "invalid_syntax",
    "wrong_table",
    "wrong_column",
    "wrong_aggregation",
    "missing_join",
    "wrong_condition",
    "wrong_value",
    "other",
)

AGG_NAMES = {"count", "sum", "avg", "min", "max"}


def _parse_sql(sql: str):
    """Return sqlglot Select or None if invalid."""
    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
        if isinstance(parsed, exp.Select):
            return parsed
        return None
    except Exception:
        return None


def _tables_from_select(sel: exp.Select) -> set[str]:
    """Extract table names from FROM/JOIN."""
    tables = set()
    for name in sel.find_all(exp.Table):
        tables.add(name.name.lower())
    return tables


def _columns_from_select(sel: exp.Select) -> set[tuple[str, str]]:
    """Extract (table, column) pairs from SELECT and WHERE/ON expressions."""
    cols = set()
    for col in sel.find_all(exp.Column):
        table = (col.table or "").lower()
        col_name = (col.name or "").lower()
        cols.add((table, col_name))
    return cols


def _aggregations_from_select(sel: exp.Select) -> set[tuple[str, str]]:
    """Extract (agg_name, column_key) from SELECT."""
    aggs = set()
    for agg in sel.find_all(exp.AggFunc):
        agg_name = (agg.sql_name() or "").lower()
        if agg_name in AGG_NAMES:
            for c in agg.find_all(exp.Column):
                aggs.add((agg_name, (c.table or "").lower() + "." + (c.name or "").lower()))
            if not list(agg.find_all(exp.Column)):
                aggs.add((agg_name, "*"))
    return aggs


def _join_count(sel: exp.Select) -> int:
    """Number of JOINs."""
    return len(list(sel.find_all(exp.Join)))


def _literals_from_select(sel: exp.Select) -> set[str]:
    """Extract string/number literals."""
    return {str(lit.this).lower() for lit in sel.find_all(exp.Literal)}


def _condition_count(sel: exp.Select) -> int:
    """Rough count of WHERE/ON conditions."""
    n = 0
    for _ in sel.find_all(exp.Where):
        n += 1
    for join in sel.find_all(exp.Join):
        if join.on:
            n += 1
    return n


def classify_error(
    pred_sql: str,
    gold_sql: str,
    db_id: str,
    tables_list: list[dict],
    db_path: Path,
) -> str:
    """Classify the failure mode. Returns one of ERROR_TYPES."""
    if not is_valid_sql(pred_sql):
        return "invalid_syntax"

    pred_ast = _parse_sql(pred_sql)
    gold_ast = _parse_sql(gold_sql)
    if pred_ast is None or gold_ast is None:
        return "other"

    schema_dict = get_schema_for_db(db_id, tables_list)
    if schema_dict is None:
        valid_tables = set()
    else:
        table_names = schema_dict.get("table_names_original", [])
        valid_tables = {t.lower() for t in table_names}

    pred_tables = _tables_from_select(pred_ast)
    gold_tables = _tables_from_select(gold_ast)

    for t in pred_tables:
        if valid_tables and t not in valid_tables:
            return "wrong_table"
        if gold_tables and t not in gold_tables:
            return "wrong_table"

    if gold_tables and len(gold_tables) > len(pred_tables):
        return "missing_join"
    if _join_count(gold_ast) > _join_count(pred_ast):
        return "missing_join"

    pred_cols = _columns_from_select(pred_ast)
    gold_cols = _columns_from_select(gold_ast)
    if pred_tables == gold_tables and pred_cols != gold_cols:
        return "wrong_column"

    pred_aggs = _aggregations_from_select(pred_ast)
    gold_aggs = _aggregations_from_select(gold_ast)
    if pred_aggs != gold_aggs:
        return "wrong_aggregation"

    if _condition_count(pred_ast) != _condition_count(gold_ast):
        return "wrong_condition"

    if _literals_from_select(pred_ast) != _literals_from_select(gold_ast):
        return "wrong_value"

    return "other"
