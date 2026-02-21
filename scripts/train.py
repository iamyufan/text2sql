"""Train T5/Flan-T5 on Spider for text-to-SQL. Uses configs/ and writes to outputs/checkpoints/.

Usage:
  uv run python scripts/train.py
  uv run python scripts/train.py --config configs/training/lora.yaml --model-config configs/model/flan_t5_base.yaml
  uv run python scripts/train.py --run-name t5-base-epoch10 --smoke
"""

import os
from pathlib import Path

import yaml

from text2sql.config import PROJECT_ROOT, database_dir, tables_json_path
from text2sql.data import get_spider_dataset, load_tables
from text2sql.training import build_trainer


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train T5/Flan-T5 on Spider text-to-SQL"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/default.yaml"),
        help="Training config YAML",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/t5_base.yaml"),
        help="Model config YAML",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for checkpoint dir (default: model name + timestamp)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None, help="Override model name"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-input-length", type=int, default=None)
    parser.add_argument("--max-target-length", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device for training (default: auto; use CUDA_VISIBLE_DEVICES to pick GPU(s))",
    )
    parser.add_argument("--use-lora", action="store_true", help="Use PEFT LoRA")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 1 epoch, 100 train / 20 dev examples",
    )
    args = parser.parse_args()

    # Project-local HuggingFace cache
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

    train_cfg = load_config(PROJECT_ROOT / args.config)
    model_cfg = load_config(PROJECT_ROOT / args.model_config)
    config = {**train_cfg, **model_cfg}
    if args.model_name is not None:
        config["model_name"] = args.model_name
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.max_input_length is not None:
        config["max_input_length"] = args.max_input_length
    if args.max_target_length is not None:
        config["max_target_length"] = args.max_target_length
    if args.use_lora:
        config["use_lora"] = True
    if args.device is not None:
        config["device"] = args.device
    config["smoke"] = args.smoke

    model_name = config.get("model_name", "t5-base")
    run_name = args.run_name or model_name.replace("/", "_")
    output_dir = args.output_dir or (
        PROJECT_ROOT / "outputs" / "checkpoints" / run_name
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    db_dir = database_dir()
    tables_path = tables_json_path()
    if not tables_path.exists():
        raise SystemExit(f"Tables not found: {tables_path}")
    if not db_dir.exists():
        raise SystemExit(f"Database dir not found: {db_dir}")

    tables_list = load_tables()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_input = config.get("max_input_length", 1024)
    max_target = config.get("max_target_length", 256)
    train_limit = 100 if args.smoke else None
    eval_limit = 20 if args.smoke else None

    train_dataset = get_spider_dataset(
        "train",
        tables_list,
        tokenizer,
        max_input_length=max_input,
        max_target_length=max_target,
        limit=train_limit,
    )
    eval_dataset = get_spider_dataset(
        "dev",
        tables_list,
        tokenizer,
        max_input_length=max_input,
        max_target_length=max_target,
        limit=eval_limit,
    )

    import wandb
    from wandb.errors import CommError

    from text2sql.config import WANDB_ENTITY, WANDB_MODE, WANDB_PROJECT

    wandb_config = {
        "model_name": model_name,
        "run_name": run_name,
        **config,
    }
    try:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY or None,
            config=wandb_config,
            mode=WANDB_MODE,
        )
    except CommError as e:
        print(
            f"W&B upload failed ({e}). Continuing with W&B disabled (logs local only)."
        )
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY or None,
            config=wandb_config,
            mode="disabled",
        )

    trainer = build_trainer(
        model_name=model_name,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        db_dir=db_dir,
        tables_path=tables_path,
        config=config,
    )

    trainer.train()
    trainer.save_state()
    if not args.smoke:
        trainer.save_model(str(output_dir / "best"))

    print(f"Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
