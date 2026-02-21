"""Serialize schemas and write processed train.jsonl and dev.jsonl to data/processed/.

Usage:
  uv run python scripts/prepare_data.py
  uv run python scripts/prepare_data.py --limit 100   # smoke: first 100 train, 50 dev
"""

import argparse
import json
from pathlib import Path

from text2sql.config import data_processed_dir, data_raw_dir
from text2sql.data import build_example_rows, load_examples, load_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare processed JSONL from raw Spider data")
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Limit train examples (for smoke runs)",
    )
    parser.add_argument(
        "--limit-dev",
        type=int,
        default=None,
        help="Limit dev examples (for smoke runs)",
    )
    args = parser.parse_args()

    raw_dir = data_raw_dir()
    processed_dir = data_processed_dir()
    tables_path = raw_dir / "tables.json"
    if not tables_path.exists():
        raise SystemExit(f"Tables not found: {tables_path}")

    tables_list = load_tables(tables_path)

    for split, limit in [("train", args.limit_train), ("dev", args.limit_dev)]:
        examples = load_examples(split)
        if limit is not None:
            examples = examples[:limit]
        rows = build_example_rows(examples, tables_list)
        out_path = processed_dir / f"{split}.jsonl"
        processed_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for row in rows:
                # Write one JSON object per line (JSONL)
                rec = {
                    "id": row["id"],
                    "db_id": row["db_id"],
                    "question": row["question"],
                    "query": row["query"],
                    "input_text": row["input_text"],
                    "target_text": row["target_text"],
                }
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(rows)} examples to {out_path}")


if __name__ == "__main__":
    main()
