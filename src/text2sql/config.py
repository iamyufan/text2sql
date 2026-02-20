"""Configuration: paths and Weights & Biases settings."""

import os
from pathlib import Path

# Spider data root (can override with SPIDER_DATA_DIR env var)
SPIDER_DATA_DIR = Path(os.environ.get("SPIDER_DATA_DIR", "spider_data")).resolve()

# Paths under Spider data
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

# Weights & Biases
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "text2sql-baseline")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
WANDB_MODE = os.environ.get("WANDB_MODE", "online")  # online | offline | disabled
