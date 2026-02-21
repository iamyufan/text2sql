"""Run full evaluation pipeline: valid SQL, execution, exact match, failure analysis."""

import json
import tempfile
from pathlib import Path

import nltk

from text2sql.evaluation.difficulty import get_hardness_and_parsed
from text2sql.evaluation.error_analysis import classify_error
from text2sql.evaluation.executor import exec_match_one
from text2sql.evaluation.metrics import is_valid_sql
from text2sql.evaluation.spider import (
    build_foreign_key_map_from_json,
    evaluate,
)


def run_eval(
    predictions: list[str],
    examples: list[dict],
    db_dir: Path,
    tables_path: Path,
    output_dir: Path | None = None,
) -> dict:
    """Run the three-level evaluation and failure analysis."""
    n = len(examples)
    if n != len(predictions):
        raise ValueError(f"len(predictions)={len(predictions)} != len(examples)={n}")

    nltk_data = tables_path.parent / ".nltk_data"
    nltk_data.mkdir(parents=True, exist_ok=True)
    if str(nltk_data) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_data))
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True, download_dir=str(nltk_data))

    with open(tables_path) as f:
        tables_list = json.load(f)
    kmaps = build_foreign_key_map_from_json(str(tables_path))
    out_dir = output_dir or Path.cwd()

    valid_count = 0
    valid_by_hardness = {"easy": 0, "medium": 0, "hard": 0, "extra": 0}
    count_by_hardness = {"easy": 0, "medium": 0, "hard": 0, "extra": 0}
    hardness_per_example = []

    for pred, ex in zip(predictions, examples):
        db_id = ex["db_id"]
        db_path = db_dir / db_id / f"{db_id}.sqlite"
        gold_str = ex["query"]
        try:
            hardness, _ = get_hardness_and_parsed(db_path, gold_str)
        except Exception:
            hardness = "easy"
        hardness_per_example.append(hardness)
        count_by_hardness[hardness] = count_by_hardness.get(hardness, 0) + 1
        if is_valid_sql(pred):
            valid_count += 1
            valid_by_hardness[hardness] = valid_by_hardness.get(hardness, 0) + 1

    valid_sql_rate = valid_count / n if n else 0.0
    valid_sql_by_difficulty = {
        h: (
            valid_by_hardness[h] / count_by_hardness[h] if count_by_hardness[h] else 0.0
        )
        for h in ["easy", "medium", "hard", "extra"]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".gold", delete=False) as fg:
        gold_path = Path(fg.name)
        for ex in examples:
            fg.write(
                ex["query"].replace("\t", " ").replace("\n", " ")
                + "\t"
                + ex["db_id"]
                + "\n"
            )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pred", delete=False) as fp:
        pred_path = Path(fp.name)
        for p, ex in zip(predictions, examples):
            fp.write(
                p.replace("\t", " ").replace("\n", " ") + "\t" + ex["db_id"] + "\n"
            )

    try:
        spider_scores = evaluate(
            str(gold_path),
            str(pred_path),
            str(db_dir),
            "all",
            kmaps,
        )
    finally:
        gold_path.unlink(missing_ok=True)
        pred_path.unlink(missing_ok=True)

    execution_accuracy = (
        spider_scores["all"]["exec"] if spider_scores["all"]["count"] else 0.0
    )
    execution_by_difficulty = {
        h: (spider_scores[h]["exec"] if spider_scores[h]["count"] else 0.0)
        for h in ["easy", "medium", "hard", "extra"]
    }
    exact_match_accuracy = (
        spider_scores["all"]["exact"] if spider_scores["all"]["count"] else 0.0
    )
    exact_match_by_difficulty = {
        h: (spider_scores[h]["exact"] if spider_scores[h]["count"] else 0.0)
        for h in ["easy", "medium", "hard", "extra"]
    }

    failures = []
    for i, (pred, ex) in enumerate(zip(predictions, examples)):
        db_id = ex["db_id"]
        db_path = db_dir / db_id / f"{db_id}.sqlite"
        gold_str = ex["query"]
        hardness = hardness_per_example[i]
        kmap = kmaps.get(db_id, {})
        try:
            _, g_sql = get_hardness_and_parsed(db_path, gold_str)
            exec_ok = exec_match_one(db_path, pred, gold_str, g_sql, kmap)
        except Exception:
            exec_ok = False
        if not exec_ok:
            error_type = classify_error(pred, gold_str, db_id, tables_list, db_path)
            failures.append(
                {
                    "question": ex["question"],
                    "gold_sql": gold_str,
                    "pred_sql": pred,
                    "db_id": db_id,
                    "difficulty": hardness,
                    "is_valid_sql": is_valid_sql(pred),
                    "error_type": error_type,
                }
            )

    failure_path = out_dir / "failure_analysis.json"
    with open(failure_path, "w") as f:
        json.dump(failures, f, indent=2)

    return {
        "valid_sql_rate": valid_sql_rate,
        "valid_sql_by_difficulty": valid_sql_by_difficulty,
        "execution_accuracy": execution_accuracy,
        "execution_by_difficulty": execution_by_difficulty,
        "exact_match_accuracy": exact_match_accuracy,
        "exact_match_by_difficulty": exact_match_by_difficulty,
        "failures": failures,
        "failure_analysis_path": str(failure_path),
        "spider_scores": spider_scores,
    }
