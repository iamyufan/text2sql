"""SQL execution against SQLite and execution-equivalence checking (Spider)."""

from pathlib import Path

from text2sql.evaluation.spider import (
    Schema,
    build_valid_col_units,
    eval_exec_match,
    get_schema,
    get_sql,
    rebuild_sql_col,
    rebuild_sql_val,
)

_EMPTY_SQL = {
    "except": None,
    "from": {"conds": [], "table_units": []},
    "groupBy": [],
    "having": [],
    "intersect": None,
    "limit": None,
    "orderBy": [],
    "select": [False, []],
    "union": None,
    "where": [],
}


def exec_match_one(
    db_path: Path,
    pred_str: str,
    gold_str: str,
    g_sql: dict,
    kmap: dict,
) -> bool:
    """Return True if prediction is execution-equivalent to gold (Spider definition)."""
    schema = Schema(get_schema(str(db_path)))
    try:
        p_sql = get_sql(schema, pred_str)
    except Exception:
        p_sql = dict(_EMPTY_SQL)
        p_sql["from"] = {"conds": [], "table_units": []}

    g_valid = build_valid_col_units(g_sql["from"]["table_units"], schema)
    g_sql = rebuild_sql_val(g_sql)
    g_sql = rebuild_sql_col(g_valid, g_sql, kmap)
    p_valid = build_valid_col_units(p_sql["from"]["table_units"], schema)
    p_sql = rebuild_sql_val(p_sql)
    p_sql = rebuild_sql_col(p_valid, p_sql, kmap)
    return eval_exec_match(str(db_path), pred_str, gold_str, p_sql, g_sql)
