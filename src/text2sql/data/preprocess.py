"""Tokenization and input construction for seq2seq training."""

from typing import Any

from datasets import Dataset

from text2sql.data.loader import load_examples
from text2sql.data.schema import get_schema_for_db, serialize_schema


def build_example_rows(
    examples: list[dict],
    tables_list: list[dict],
) -> list[dict[str, Any]]:
    """Build list of dicts with input_text, target_text, db_id, id.

    Skips examples whose db_id is not in tables_list.
    """
    rows = []
    for i, ex in enumerate(examples):
        db_id = ex.get("db_id")
        if not db_id:
            continue
        schema = get_schema_for_db(db_id, tables_list)
        if schema is None:
            continue
        schema_str = serialize_schema(schema)
        input_text = f"question: {ex['question']} | schema: {schema_str}"
        target_text = ex.get("query", "")
        rows.append(
            {
                "id": i,
                "db_id": db_id,
                "question": ex["question"],
                "query": ex.get("query", ""),
                "input_text": input_text,
                "target_text": target_text,
            }
        )
    return rows


def tokenize_seq2seq(
    examples: dict[str, list],
    tokenizer: Any,
    max_input_length: int = 1024,
    max_target_length: int = 256,
) -> dict[str, list]:
    """Tokenize input_text and target_text for seq2seq. Returns input_ids, attention_mask, labels."""
    inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    targets = tokenizer(
        examples["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    labels = []
    for ids in targets["input_ids"]:
        # -100 is ignored by CrossEntropyLoss
        labels.append([x if x != tokenizer.pad_token_id else -100 for x in ids])
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
    }


def get_spider_dataset(
    split: str,
    tables_list: list[dict],
    tokenizer: Any,
    *,
    max_input_length: int = 1024,
    max_target_length: int = 256,
    limit: int | None = None,
) -> Dataset:
    """Build HuggingFace Dataset for train or dev with tokenized columns.

    Preserves db_id and id for evaluation. If limit is set, uses first
    limit examples (for smoke tests).
    """
    examples = load_examples(split)
    if limit is not None:
        examples = examples[:limit]
    rows = build_example_rows(examples, tables_list)
    if not rows:
        raise ValueError(f"No valid examples for split={split!r} (limit={limit})")
    ds = Dataset.from_list(rows)

    def tokenize_fn(batch: dict) -> dict:
        return tokenize_seq2seq(
            batch,
            tokenizer,
            max_input_length=max_input_length,
            max_target_length=max_target_length,
        )

    ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["input_text", "target_text"],
        desc=f"Tokenize {split}",
    )
    # Keep id, db_id, question, query for eval callback (run_eval expects examples with these keys)
    return ds
