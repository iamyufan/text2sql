"""Spot-check 20 serialized inputs: print question, schema (CREATE TABLE), and gold SQL for manual review."""

import json
import random
from pathlib import Path

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

from text2sql.config import data_raw_dir
from text2sql.data import get_schema_for_db, serialize_schema

SEED = 42
N_SPOTCHECK = 20


def main() -> None:
    raw_dir = data_raw_dir()
    with open(raw_dir / "dev.json") as f:
        dev = json.load(f)
    with open(raw_dir / "tables.json") as f:
        tables_list = json.load(f)

    random.seed(SEED)
    indices = random.sample(range(len(dev)), min(N_SPOTCHECK, len(dev)))
    chosen = [dev[i] for i in sorted(indices)]

    lines = []
    for i, ex in enumerate(chosen, 1):
        db_id = ex["db_id"]
        question = ex["question"]
        gold_sql = ex.get("query", "")
        schema = get_schema_for_db(db_id, tables_list)
        schema_str = serialize_schema(schema) if schema else "(schema not found)"
        block = [
            f"--- Example {i} ---",
            f"db_id: {db_id}",
            f"question: {question}",
            f"schema: {schema_str}",
            f"gold SQL: {gold_sql}",
            "",
        ]
        lines.extend(block)

    out_text = "\n".join(lines)
    print(out_text)

    out_file = PROJECT_ROOT / "spotcheck_20.txt"
    out_file.write_text(out_text, encoding="utf-8")
    print(f"\nWritten to {out_file}")


if __name__ == "__main__":
    main()
