# Text2SQL

Text-to-SQL baseline: fine-tune T5 on the Spider dataset. Phase 1 covers data setup, schema serialization, and audits.

## Setup

- **Python**: 3.10+
- **Package manager**: [uv](https://docs.astral.sh/uv/)

```bash
uv sync
```

Optional dev tools (tests, download script): already included via `dependency-groups.dev`; run `uv sync` to install.

## Environment

- **Spider data**: Default root is `spider_data/`. Override with `SPIDER_DATA_DIR`.
- **Weights & Biases**:
  - Online logging: set `WANDB_API_KEY` or run `wandb login`.
  - Project name: `WANDB_PROJECT` (default: `text2sql-baseline`). Optional: `WANDB_ENTITY`.
  - Local runs without an account: `WANDB_MODE=disabled`.
  - Offline logging (sync later): `WANDB_MODE=offline`.

## Phase 1 commands

1. **Download Spider** (if needed): run `scripts/download_spider.py` or download the [Spider dataset](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing) and extract into `spider_data/`.
2. **Verify data**: `uv run python scripts/verify_spider_data.py`
3. **Token length audit**: `uv run python scripts/audit_token_lengths.py`
4. **Spot-check 20 examples**: `uv run python scripts/spotcheck_serialized.py`

## Project layout

- `src/text2sql/` – package (schema, config)
- `scripts/` – download, verify, audit, spot-check
- `spider_data/` – Spider JSON + `database/{db_id}/{db_id}.sqlite`
- `tests/` – pytest (e.g. `test_schema.py`)
