"""Pytest configuration. Prefer data/raw for paths; fall back to spider_data if data/raw is missing."""

import os
from pathlib import Path

# Before any text2sql.config import, set DATA_RAW so loader/tests find Spider data
_root = Path(__file__).resolve().parent.parent
if not (_root / "data" / "raw" / "tables.json").exists() and (_root / "spider_data" / "tables.json").exists():
    os.environ["DATA_RAW"] = str(_root / "spider_data")
