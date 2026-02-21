"""Generate error analysis report. Writes to outputs/reports/.

Usage:
  uv run python scripts/analyze_errors.py --run-name my_run --predictions preds.txt
  uv run python scripts/analyze_errors.py --run-name baseline   # uses trivial baseline
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
    parser = argparse.ArgumentParser(description="Generate error analysis report")
    parser.add_argument("--run-name", type=str, default="error_analysis")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to predictions file (one SQL per line). If omitted, use trivial baseline.",
    )
    parser.add_argument("--dev", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    db_dir = database_dir()
    tables_path = tables_json_path()
    dev_path_ = args.dev or dev_path()

    with open(dev_path_) as f:
        examples = json.load(f)
    if args.limit is not None:
        examples = examples[: args.limit]
    with open(tables_path) as f:
        tables_list = json.load(f)

    if args.predictions is not None:
        with open(args.predictions) as f:
            predictions = [line.strip() for line in f if line.strip()]
    else:
        predictions = trivial_baseline(examples, tables_list)
    if len(predictions) != len(examples):
        raise SystemExit(
            f"Predictions has {len(predictions)} lines, dev has {len(examples)} examples"
        )

    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    result = run_eval(
        predictions=predictions,
        examples=examples,
        db_dir=db_dir,
        tables_path=tables_path,
        output_dir=reports_dir,
    )

    report_path = reports_dir / f"{args.run_name}_error_analysis.json"
    with open(report_path, "w") as f:
        json.dump(result["failures"], f, indent=2)
    print(f"Error analysis written to {report_path}")
    print(f"Total failures: {len(result['failures'])}")


if __name__ == "__main__":
    main()
