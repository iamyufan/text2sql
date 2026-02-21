"""Run evaluation on Spider dev set. Write predictions and report to outputs/.

Usage:
  # Trivial baseline
  uv run python scripts/evaluate.py --run-name baseline

  # With predictions file (one SQL per line, same order as dev.json)
  uv run python scripts/evaluate.py --run-name my_run --predictions preds.txt

  # Limit to first N examples (for testing)
  uv run python scripts/evaluate.py --run-name smoke --limit 5
"""

import argparse
import json
from pathlib import Path

from text2sql.config import PROJECT_ROOT, database_dir, dev_path, tables_json_path
from text2sql.data import get_schema_for_db
from text2sql.evaluation import run_eval


def trivial_baseline(examples: list[dict], tables_list: list[dict]) -> list[str]:
    """Predict 'SELECT * FROM <first_table>' for each example."""
    predictions = []
    for ex in examples:
        schema = get_schema_for_db(ex["db_id"], tables_list)
        if schema and schema.get("table_names_original"):
            first_table = schema["table_names_original"][0]
            predictions.append(f"SELECT * FROM {first_table}")
        else:
            predictions.append("SELECT 1")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Spider evaluation pipeline")
    parser.add_argument(
        "--run-name",
        type=str,
        default="eval",
        help="Run name for outputs (predictions and reports)",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to predictions file (one SQL per line, same order as dev.json)",
    )
    parser.add_argument(
        "--dev",
        type=Path,
        default=None,
        help="Path to dev.json (default: from config)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N dev examples (for testing)",
    )
    args = parser.parse_args()

    db_dir = database_dir()
    tables_path = tables_json_path()
    dev_path_ = args.dev or dev_path()

    if not dev_path_.exists():
        raise SystemExit(f"Dev file not found: {dev_path_}")
    if not tables_path.exists():
        raise SystemExit(f"Tables file not found: {tables_path}")
    if not db_dir.exists():
        raise SystemExit(f"Database dir not found: {db_dir}")

    with open(dev_path_) as f:
        examples = json.load(f)
    if args.limit is not None:
        examples = examples[: args.limit]
        print(f"Limited to first {args.limit} examples")
    with open(tables_path) as f:
        tables_list = json.load(f)

    if args.predictions is not None:
        with open(args.predictions) as f:
            predictions = [line.strip() for line in f if line.strip()]
        if len(predictions) != len(examples):
            raise SystemExit(
                f"Predictions file has {len(predictions)} lines, dev has {len(examples)} examples"
            )
        print("Using predictions from", args.predictions)
    else:
        predictions = trivial_baseline(examples, tables_list)
        print("Using trivial baseline: SELECT * FROM <first_table>")

    outputs_dir = PROJECT_ROOT / "outputs"
    predictions_dir = outputs_dir / "predictions"
    reports_dir = outputs_dir / "reports"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_dir = reports_dir  # run_eval writes failure_analysis.json here; we'll rename

    result = run_eval(
        predictions=predictions,
        examples=examples,
        db_dir=db_dir,
        tables_path=tables_path,
        output_dir=output_dir,
    )

    # Write predictions JSONL and rename report by run_name
    preds_path = predictions_dir / f"{args.run_name}_dev_preds.jsonl"
    with open(preds_path, "w") as f:
        for ex, pred in zip(examples, predictions):
            f.write(json.dumps({"db_id": ex["db_id"], "question": ex["question"], "pred_sql": pred}) + "\n")
    print(f"Predictions written to {preds_path}")

    report_path = reports_dir / f"{args.run_name}_error_analysis.json"
    with open(report_path, "w") as f:
        json.dump(result["failures"], f, indent=2)
    print(f"Error analysis written to {report_path}")

    print("\n" + "=" * 60)
    print("LEVEL 1: Valid SQL Rate")
    print("=" * 60)
    print(f"  Overall: {result['valid_sql_rate']:.4f}")
    for h in ["easy", "medium", "hard", "extra"]:
        print(f"    {h}: {result['valid_sql_by_difficulty'][h]:.4f}")

    print("\n" + "=" * 60)
    print("LEVEL 2: Execution Accuracy (Primary Metric)")
    print("=" * 60)
    print(f"  Overall: {result['execution_accuracy']:.4f}")
    for h in ["easy", "medium", "hard", "extra"]:
        print(f"    {h}: {result['execution_by_difficulty'][h]:.4f}")

    print("\n" + "=" * 60)
    print("LEVEL 3: Exact Match Accuracy")
    print("=" * 60)
    print(f"  Overall: {result['exact_match_accuracy']:.4f}")
    for h in ["easy", "medium", "hard", "extra"]:
        print(f"    {h}: {result['exact_match_by_difficulty'][h]:.4f}")

    print("\n" + "=" * 60)
    print("Failure analysis")
    print("=" * 60)
    print(f"  Failures: {len(result['failures'])}")


if __name__ == "__main__":
    main()
