"""Tests for the training data module (Spider load + tokenize)."""

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _spider_available() -> bool:
    for base in [PROJECT_ROOT / "data" / "raw", PROJECT_ROOT / "spider_data"]:
        if not base.exists():
            continue
        for name in ["tables.json", "train_spider.json", "train_others.json", "dev.json"]:
            if not (base / name).exists():
                break
        else:
            return True
    return False


@pytest.fixture
def spider_data_available() -> bool:
    return _spider_available()


def test_load_tables(spider_data_available: bool) -> None:
    if not spider_data_available:
        pytest.skip("Spider data not available")
    from text2sql.data import load_tables
    base = PROJECT_ROOT / "data" / "raw" if (PROJECT_ROOT / "data" / "raw" / "tables.json").exists() else PROJECT_ROOT / "spider_data"
    tables = load_tables(base / "tables.json")
    assert isinstance(tables, list)
    assert len(tables) > 0
    for t in tables[:3]:
        assert "db_id" in t
        assert "table_names_original" in t
        assert "column_names_original" in t


def test_load_examples(spider_data_available: bool) -> None:
    if not spider_data_available:
        pytest.skip("Spider data not available")
    from text2sql.data import load_examples

    train = load_examples("train")
    dev = load_examples("dev")
    assert len(train) > 0
    assert len(dev) > 0
    for ex in train[:1] + dev[:1]:
        assert "db_id" in ex
        assert "question" in ex
        assert "query" in ex


def test_get_spider_dataset_smoke(spider_data_available: bool) -> None:
    if not spider_data_available:
        pytest.skip("Spider data not available")
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

    from transformers import AutoTokenizer

    from text2sql.data import get_spider_dataset, load_tables

    tables_list = load_tables()
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = get_spider_dataset(
        "train",
        tables_list,
        tokenizer,
        max_input_length=128,
        max_target_length=64,
        limit=10,
    )
    assert train_ds.num_rows == 10
    assert "input_ids" in train_ds.column_names
    assert "attention_mask" in train_ds.column_names
    assert "labels" in train_ds.column_names

    dev_ds = get_spider_dataset(
        "dev",
        tables_list,
        tokenizer,
        max_input_length=128,
        max_target_length=64,
        limit=5,
    )
    assert dev_ds.num_rows == 5
