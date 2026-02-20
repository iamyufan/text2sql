"""Configuration: paths and Weights & Biases settings.

Loads environment from a .env file at the project root if present. Shell env
vars override .env values. Project root is the parent of src/.

Supported .env / environment variables:
  SPIDER_DATA_DIR   Path to Spider data (default: spider_data, relative to project root)
  WANDB_PROJECT     W&B project name (default: text2sql-baseline)
  WANDB_ENTITY      W&B entity/team (optional)
  WANDB_MODE        online | offline | disabled (default: online)
  HF_TOKEN          HuggingFace token for private models / higher rate limits (optional)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root: parent of src/ (parent of this file's parent)
_PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _PACKAGE_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Spider data root (override with SPIDER_DATA_DIR in .env or environment)
SPIDER_DATA_DIR = Path(os.environ.get("SPIDER_DATA_DIR", "spider_data")).resolve()
if not SPIDER_DATA_DIR.is_absolute():
    SPIDER_DATA_DIR = (PROJECT_ROOT / SPIDER_DATA_DIR).resolve()


def spider_path(*parts: str) -> Path:
    return SPIDER_DATA_DIR.joinpath(*parts)


def database_dir() -> Path:
    return spider_path("database")


def tables_json_path() -> Path:
    return spider_path("tables.json")


def train_spider_path() -> Path:
    return spider_path("train_spider.json")


def train_others_path() -> Path:
    return spider_path("train_others.json")


def dev_path() -> Path:
    return spider_path("dev.json")


# Weights & Biases (from .env or environment)
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "text2sql-baseline")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
WANDB_MODE = os.environ.get("WANDB_MODE", "online")  # online | offline | disabled

# HuggingFace (optional; improves rate limits when set)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
