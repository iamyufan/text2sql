"""Configuration: paths and Weights & Biases settings.

Loads environment from a .env file at the project root if present. Shell env
vars override .env values. Project root is the parent of src/.

Supported .env / environment variables:
  DATA_RAW        Path to raw Spider data (default: data/raw, relative to project root)
  DATA_PROCESSED  Path to processed JSONL output (default: data/processed)
  WANDB_PROJECT   W&B project name (default: text2sql-baseline)
  WANDB_ENTITY    W&B entity/team (optional)
  WANDB_MODE      online | offline | disabled (default: online)
  HF_TOKEN        HuggingFace token for private models / higher rate limits (optional)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root: parent of src/ (parent of this file's parent)
_PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _PACKAGE_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Data paths: raw (Spider JSON + database/) and processed (train.jsonl, dev.jsonl)
_DATA_RAW = os.environ.get("DATA_RAW", "data/raw")
_DATA_PROCESSED = os.environ.get("DATA_PROCESSED", "data/processed")

DATA_RAW_DIR = Path(_DATA_RAW).resolve()
if not DATA_RAW_DIR.is_absolute():
    DATA_RAW_DIR = (PROJECT_ROOT / DATA_RAW_DIR).resolve()

DATA_PROCESSED_DIR = Path(_DATA_PROCESSED).resolve()
if not DATA_PROCESSED_DIR.is_absolute():
    DATA_PROCESSED_DIR = (PROJECT_ROOT / DATA_PROCESSED_DIR).resolve()


def data_raw_dir() -> Path:
    """Raw Spider data root (train_spider.json, dev.json, tables.json, database/)."""
    return DATA_RAW_DIR


def data_processed_dir() -> Path:
    """Processed JSONL output directory (train.jsonl, dev.jsonl)."""
    return DATA_PROCESSED_DIR


def database_dir() -> Path:
    return data_raw_dir() / "database"


def tables_json_path() -> Path:
    return data_raw_dir() / "tables.json"


def train_spider_path() -> Path:
    return data_raw_dir() / "train_spider.json"


def train_others_path() -> Path:
    return data_raw_dir() / "train_others.json"


def dev_path() -> Path:
    return data_raw_dir() / "dev.json"


# Weights & Biases (from .env or environment)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
WANDB_MODE = os.environ.get("WANDB_MODE", "online")  # online | offline | disabled

# HuggingFace (optional; improves rate limits when set)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
