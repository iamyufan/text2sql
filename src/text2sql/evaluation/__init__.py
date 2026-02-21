"""Evaluation: valid SQL, execution accuracy, exact match, difficulty, error analysis."""

from text2sql.evaluation.error_analysis import ERROR_TYPES, classify_error
from text2sql.evaluation.metrics import execute_query, is_valid_sql
from text2sql.evaluation.pipeline import run_eval

__all__ = [
    "run_eval",
    "is_valid_sql",
    "execute_query",
    "classify_error",
    "ERROR_TYPES",
]
