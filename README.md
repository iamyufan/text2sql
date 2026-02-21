# Text2SQL Part I: T5 baseline

A baseline for **text-to-SQL translation** by fine-tuning a T5-based seq2seq model on the Spider dataset. This baseline is the comparison anchor for future approaches (few-shot prompting, RAG, agentic workflows). Tech stack: HuggingFace transformers + datasets, PEFT LoRA, Weights & Biases, custom eval pipeline, SQLite, uv.

**Phases completed:**

- **Phase 1 – Setup:** Spider download/verify, Python env (uv), W&B config, schema serialization, token-length audit, spot-check.
- **Phase 2 – Evaluation pipeline:** Valid SQL (sqlglot), execution accuracy, exact match (Spider evaluation.py), metrics by difficulty, failure analysis; tested with a trivial baseline.
- **Phase 3 – Training prep:** Dataset load/preprocess (HuggingFace Dataset, tokenization), model/tokenizer init (t5-base or flan-t5-base, optional LoRA), training script with W&B logging and custom eval callback, 1-epoch smoke test on 100 train / 20 dev examples.

## Setup

- **Python**: 3.10+
- **Package manager**: [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

Optional dev tools (tests, download script): already included via `dependency-groups.dev`; run `uv sync` to install.

Copy the env template and fill in your values (do not commit `.env`):

```bash
cp .env.example .env
```

## Environment

- **Data paths**: Raw Spider data lives under `data/raw/` (override with `DATA_RAW`). Processed JSONL is written to `data/processed/` (override with `DATA_PROCESSED`). See `.env.example`.
- **Weights & Biases**:
  - Online logging: set `WANDB_API_KEY` or run `wandb login`.
  - Project name: `WANDB_PROJECT` (default: `text2sql`). Optional: `WANDB_ENTITY`.
  - Local runs without an account: `WANDB_MODE=disabled`.
  - Offline logging (sync later): `WANDB_MODE=offline`.

## Project layout

```
text2sql/
├── .venv/                    # managed by uv (do not touch manually)
├── .env                      # API keys, W&B token, paths (never commit)
├── .env.example              # template of .env (commit this)
├── pyproject.toml
├── uv.lock
├── README.md
├── data/
│   ├── raw/                  # original Spider download (train_*.json, dev.json, tables.json, database/)
│   └── processed/            # serialized data from prepare_data.py (train.jsonl, dev.jsonl)
├── configs/
│   ├── model/                # t5_base.yaml, flan_t5_base.yaml
│   └── training/             # default.yaml, lora.yaml
├── src/text2sql/             # installable package (uv install -e .)
│   ├── data/                 # loader, schema, preprocess
│   ├── training/             # trainer, callbacks
│   ├── evaluation/           # executor, metrics, difficulty, error_analysis
│   └── utils/                # logging
├── scripts/
│   ├── prepare_data.py       # serialize schemas → data/processed/
│   ├── train.py              # main training entry point
│   ├── evaluate.py           # run eval on dev set
│   └── analyze_errors.py     # generate error analysis report
├── notebooks/                # EDA and debugging only
├── outputs/                  # auto-generated (checkpoints, predictions, reports)
└── tests/                    # test_schema.py, test_metrics.py, test_executor.py
```

## Data

Place the Spider dataset under `data/raw/`: `train_spider.json`, `train_others.json`, `dev.json`, `tables.json`, and `database/{db_id}/{db_id}.sqlite`. You can download from the [Spider dataset](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing) and extract there, or symlink an existing `spider_data/` to `data/raw` and set `DATA_RAW=spider_data` in `.env`.

**Prepare processed JSONL** (optional; training can also build from raw on the fly):

```bash
uv run python scripts/prepare_data.py
```

## Evaluation pipeline

The pipeline is reusable across all future approaches (e.g. fine-tuned T5, RAG, agents). It runs three levels of evaluation and writes structured failure logs for error analysis.

### Levels

1. **Valid SQL rate** – Fraction of model outputs that parse as valid SQL (sqlglot).
2. **Execution accuracy** (primary metric) – Fraction of predictions that return the same result set as the gold query (Spider definition; column-aligned).
3. **Exact match accuracy** – Structural match after normalization (official Spider evaluation).

Metrics are reported **overall** and **by difficulty** (easy / medium / hard / extra), using Spider’s hardness labels derived from the gold SQL.

### Failure analysis

For every example where execution does not match gold, the pipeline logs:

- `question`, `gold_sql`, `pred_sql`, `db_id`, `difficulty`
- `is_valid_sql`
- `error_type`: one of `invalid_syntax`, `wrong_table`, `wrong_column`, `wrong_aggregation`, `missing_join`, `wrong_condition`, `wrong_value`, `other`

Reports are written to `outputs/reports/{run_name}_error_analysis.json`.

### Running evaluation

**Trivial baseline** (predict `SELECT * FROM <first_table>` per example):

```bash
uv run python scripts/evaluate.py --run-name baseline
```

**With a predictions file** (one SQL per line, same order as `dev.json`):

```bash
uv run python scripts/evaluate.py --run-name my_run --predictions preds.txt
```

**Options:** `--run-name`, `--predictions`, `--dev`, `--limit`. Predictions and the error report are written to `outputs/predictions/` and `outputs/reports/`.

**Error analysis only** (from an existing predictions file):

```bash
uv run python scripts/analyze_errors.py --run-name my_run --predictions preds.txt
```

## Training

Training uses the data module to build input strings `question: {question} | schema: {CREATE TABLE ...}` and tokenized HuggingFace Datasets, then runs `Seq2SeqTrainer` with optional LoRA and a custom eval callback that runs the evaluation pipeline and logs metrics (including inference latency and avg input tokens) to W&B. Checkpoints and the best model are saved under `outputs/checkpoints/{run_name}/`.

**Smoke test** (1 epoch, 100 train / 20 dev examples):

```bash
uv run python scripts/train.py --smoke
```

Disable W&B for a quick local run: `WANDB_MODE=disabled uv run python scripts/train.py --smoke`.

**Full training** (using configs):

```bash
uv run python scripts/train.py --config configs/training/default.yaml --model-config configs/model/t5_base.yaml --run-name t5-base
```

**With LoRA:**

```bash
uv run python scripts/train.py --config configs/training/lora.yaml --model-config configs/model/flan_t5_base.yaml --run-name flan-lora
```

**Options:** `--config`, `--model-config`, `--run-name`, `--model-name`, `--epochs`, `--batch-size`, `--max-input-length`, `--max-target-length`, `--output-dir`, `--use-lora`, `--smoke`. Best checkpoint is selected by `eval_execution_accuracy` and saved to `outputs/checkpoints/{run_name}/best/`.

## Scripts summary

| Script | Purpose |
|--------|--------|
| `prepare_data.py` | Serialize schemas and write `data/processed/train.jsonl` and `dev.jsonl`. |
| `train.py` | Main training entry point; uses configs and writes to `outputs/checkpoints/`. |
| `evaluate.py` | Run evaluation on dev set; writes to `outputs/predictions/` and `outputs/reports/`. |
| `analyze_errors.py` | Generate error analysis report from a predictions file (or trivial baseline). |

Optional: `audit_token_lengths.py`, `spotcheck_serialized.py` for token-length audit and spot-checking serialized inputs (use data from `data/raw/` via config).

## Tests

From the project root:

```bash
uv run pytest tests/ -v
```

- **test_schema.py** – Schema serialization (CREATE TABLE format).
- **test_metrics.py** – Eval metrics: `is_valid_sql`, `execute_query`.
- **test_executor.py** – SQL execution and execution-equivalence logic.

Tests that need Spider data will skip if `data/raw/` (or `spider_data/` with `DATA_RAW` set) is not present.
