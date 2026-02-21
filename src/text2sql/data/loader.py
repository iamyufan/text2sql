"""Load raw Spider JSON files from data/raw/."""

import json
from pathlib import Path

from text2sql.config import (
    dev_path,
    tables_json_path,
    train_others_path,
    train_spider_path,
)


def load_tables(tables_path: Path | None = None) -> list[dict]:
    """Load tables.json. Uses config path if tables_path is None."""
    path = tables_path or tables_json_path()
    with open(path) as f:
        return json.load(f)


def load_examples(split: str, data_dir: Path | None = None) -> list[dict]:
    """Load Spider examples for split 'train' or 'dev'.

    Train = train_spider.json + train_others.json. Dev = dev.json.
    data_dir is unused (paths come from config); kept for API consistency.
    """
    if split == "train":
        with open(train_spider_path()) as f:
            train_spider = json.load(f)
        with open(train_others_path()) as f:
            train_others = json.load(f)
        return train_spider + train_others
    if split == "dev":
        with open(dev_path()) as f:
            return json.load(f)
    raise ValueError(f"split must be 'train' or 'dev', got {split!r}")
