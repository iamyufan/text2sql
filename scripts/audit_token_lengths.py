"""Audit token length distribution for model input: question: {question} | schema: {CREATE TABLE...}.

Uses T5 tokenizer. Reports min, max, mean, percentiles for train (spider + others) and dev.
"""

import json
import os
from pathlib import Path

# Use project-local cache for HuggingFace to avoid permission issues
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / ".cache" / "huggingface"))

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SPIDER_DATA_DIR = Path(
    __import__("os").environ.get("SPIDER_DATA_DIR", str(PROJECT_ROOT / "spider_data"))
).resolve()

from text2sql.schema import get_schema_for_db, serialize_schema


def load_json(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    tables_list = load_json(SPIDER_DATA_DIR / "tables.json")
    train_spider = load_json(SPIDER_DATA_DIR / "train_spider.json")
    train_others = load_json(SPIDER_DATA_DIR / "train_others.json")
    dev = load_json(SPIDER_DATA_DIR / "dev.json")
    train = train_spider + train_others

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    def input_string(example: dict) -> str:
        schema = get_schema_for_db(example["db_id"], tables_list)
        if schema is None:
            return ""
        schema_str = serialize_schema(schema)
        return f"question: {example['question']} | schema: {schema_str}"

    def lengths(examples: list) -> list[int]:
        out = []
        for ex in examples:
            s = input_string(ex)
            if not s:
                continue
            out.append(len(tokenizer.encode(s, add_special_tokens=True)))
        return out

    train_lens = lengths(train)
    dev_lens = lengths(dev)

    def stats(name: str, lens: list[int]) -> None:
        if not lens:
            print(f"{name}: (no data)")
            return
        lens = sorted(lens)
        n = len(lens)
        mean = sum(lens) / n
        p50 = lens[int(0.50 * n)] if n else 0
        p95 = lens[int(0.95 * n)] if n else 0
        p99 = lens[int(0.99 * n)] if n else 0
        print(f"{name}: n={n} min={lens[0]} max={lens[-1]} mean={mean:.1f} p50={p50} p95={p95} p99={p99}")

    print("Input format: question: {question} | schema: {CREATE TABLE ...}")
    print("Tokenizer: t5-base")
    print()
    stats("train (spider + others)", train_lens)
    stats("dev", dev_lens)
    print()
    # Simple text histogram (buckets)
    for label, lens in [("train", train_lens), ("dev", dev_lens)]:
        if not lens:
            continue
        buckets = [0] * 10
        max_len = max(lens)
        step = max(1, (max_len + 99) // 100 * 10)  # ~10 buckets
        for L in lens:
            idx = min(L // step, 9)
            buckets[idx] += 1
        print(f"{label} histogram (bucket size ~{step}): {buckets}")


if __name__ == "__main__":
    main()
