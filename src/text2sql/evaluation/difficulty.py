"""Per-tier breakdown using Spider hardness labels."""

from pathlib import Path

from text2sql.evaluation.spider import Evaluator, Schema, get_schema, get_sql


def get_hardness(db_path: Path, gold_sql_str: str) -> str:
    """Return Spider hardness label for the gold query: easy, medium, hard, extra."""
    schema = Schema(get_schema(str(db_path)))
    g_sql = get_sql(schema, gold_sql_str)
    return Evaluator().eval_hardness(g_sql)


def get_hardness_and_parsed(db_path: Path, gold_sql_str: str) -> tuple[str, dict]:
    """Return (hardness, parsed_gold_sql) for the gold query."""
    schema = Schema(get_schema(str(db_path)))
    g_sql = get_sql(schema, gold_sql_str)
    hardness = Evaluator().eval_hardness(g_sql)
    return hardness, g_sql
