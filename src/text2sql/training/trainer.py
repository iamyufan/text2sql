"""HuggingFace Trainer setup for text-to-SQL fine-tuning."""

from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from text2sql.evaluation import run_eval

from text2sql.training.callbacks import Text2SqlEvalCallback


def _examples_from_eval_dataset(eval_dataset: Dataset) -> list[dict]:
    """Build list of {db_id, question, query} from eval dataset for run_eval."""
    return [
        {"db_id": r["db_id"], "question": r["question"], "query": r["query"]}
        for r in eval_dataset
    ]


def make_compute_metrics(
    eval_examples: list[dict],
    tokenizer: Any,
    db_dir: Path,
    tables_path: Path,
    output_dir: Path,
):
    """Return a compute_metrics function for Seq2SeqTrainer (decode + run_eval)."""

    def compute_metrics(eval_pred):
        pred_ids = getattr(
            eval_pred,
            "predictions",
            eval_pred[0] if isinstance(eval_pred, (list, tuple)) else None,
        )
        if pred_ids is None:
            return {}
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        # Ensure CPU list of Python ints in vocab range to avoid OverflowError in tokenizer
        if hasattr(pred_ids, "cpu"):
            pred_ids = pred_ids.cpu().numpy()
        pred_ids = pred_ids.tolist()
        vocab_size = tokenizer.vocab_size
        pred_ids = [
            [min(max(int(t), 0), vocab_size - 1) for t in seq] for seq in pred_ids
        ]
        predictions = tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        predictions = [p.strip() for p in predictions]
        examples = eval_examples
        if len(predictions) != len(examples):
            return {}
        result = run_eval(
            predictions=predictions,
            examples=examples,
            db_dir=db_dir,
            tables_path=tables_path,
            output_dir=output_dir,
        )
        return {
            "valid_sql_rate": result["valid_sql_rate"],
            "execution_accuracy": result["execution_accuracy"],
            "exact_match_accuracy": result["exact_match_accuracy"],
            "exec_acc_easy": result["execution_by_difficulty"]["easy"],
            "exec_acc_medium": result["execution_by_difficulty"]["medium"],
            "exec_acc_hard": result["execution_by_difficulty"]["hard"],
            "exec_acc_extra": result["execution_by_difficulty"]["extra"],
        }

    return compute_metrics


def build_trainer(
    *,
    model_name: str,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: Path,
    db_dir: Path,
    tables_path: Path,
    config: dict[str, Any],
) -> Seq2SeqTrainer:
    """Build Seq2SeqTrainer with model, data collator, compute_metrics, and callbacks."""
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if getattr(model.config, "decoder_start_token_id", None) is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    # Reduce GPU memory so full runs fit on ~22GB
    if torch.cuda.is_available() and not config.get("no_cuda", False):
        model.gradient_checkpointing_enable()

    use_lora = config.get("use_lora", False)
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            target_modules=config.get("lora_target_modules", ["q", "v"]),
            lora_dropout=config.get("lora_dropout", 0.05),
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 16)
    max_target_length = config.get("max_target_length", 256)
    smoke = config.get("smoke", False)

    train_batch = 4 if smoke else batch_size
    eval_batch = 4 if smoke else batch_size
    effective_batch = train_batch * config.get("gradient_accumulation_steps", 2)

    # Device: use_cuda=False or device="cpu" in config, or set CUDA_VISIBLE_DEVICES="" for CPU
    no_cuda = config.get("no_cuda", False) or (
        str(config.get("device", "")).lower() == "cpu"
    )
    use_fp16 = not no_cuda and torch.cuda.is_available()

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1 if smoke else epochs,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
        learning_rate=config.get("learning_rate", 5e-4),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        weight_decay=config.get("weight_decay", 0.01),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        predict_with_generate=True,
        generation_max_length=max_target_length,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=not smoke,
        metric_for_best_model="eval_execution_accuracy",
        greater_is_better=True,
        logging_steps=50,
        fp16=use_fp16,
        gradient_checkpointing=True,
        report_to="wandb",
        remove_unused_columns=False,
    )
    if smoke:
        training_args.max_steps = min(
            50, (len(train_dataset) // (train_batch * 2)) + 1
        )

    eval_examples = _examples_from_eval_dataset(eval_dataset)
    # Remove non-tensor columns so DataCollator and model.generate() don't see them
    cols_to_remove = ["question", "query", "db_id", "id"]
    train_dataset = train_dataset.remove_columns([c for c in cols_to_remove if c in train_dataset.column_names])
    eval_dataset = eval_dataset.remove_columns([c for c in cols_to_remove if c in eval_dataset.column_names])

    compute_metrics_fn = make_compute_metrics(
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        db_dir=db_dir,
        tables_path=tables_path,
        output_dir=output_dir,
    )
    eval_callback = Text2SqlEvalCallback(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        generation_max_length=max_target_length,
        latency_sample_size=min(50, len(eval_dataset)),
    )

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[eval_callback],
    )
