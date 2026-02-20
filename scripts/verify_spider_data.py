"""Verify Spider data: required JSON files and SQLite DBs for each db_id in tables.json.

Exit 0 if all checks pass; non-zero otherwise.
"""

import json
import sqlite3
import sys
from pathlib import Path

# Project root = parent of scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SPIDER_DATA_DIR = Path(
    __import__("os").environ.get("SPIDER_DATA_DIR", str(PROJECT_ROOT / "spider_data"))
).resolve()

REQUIRED_FILES = [
    "train_spider.json",
    "train_others.json",
    "dev.json",
    "tables.json",
]


def main() -> int:
    errors = []
    for name in REQUIRED_FILES:
        p = SPIDER_DATA_DIR / name
        if not p.exists():
            errors.append(f"Missing file: {p}")

    tables_list = []
    tables_path = SPIDER_DATA_DIR / "tables.json"
    if tables_path.exists():
        try:
            with open(tables_path) as f:
                tables_list = json.load(f)
        except Exception as e:
            errors.append(f"Invalid tables.json: {e}")

    missing_db = []
    invalid_sqlite = []
    for t in tables_list:
        db_id = t.get("db_id")
        if not db_id:
            continue
        sqlite_path = SPIDER_DATA_DIR / "database" / db_id / f"{db_id}.sqlite"
        if not sqlite_path.exists():
            missing_db.append(db_id)
            continue
        try:
            conn = sqlite3.connect(str(sqlite_path))
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1"
            ).fetchone()
            conn.close()
        except Exception as e:
            invalid_sqlite.append((db_id, str(e)))

    if missing_db:
        suffix = f"... and {len(missing_db) - 10} more" if len(missing_db) > 10 else ""
        errors.append(
            f"Missing SQLite files ({len(missing_db)} db(s)): {missing_db[:10]} {suffix}"
        )
    if invalid_sqlite:
        for db_id, msg in invalid_sqlite[:5]:
            errors.append(f"Invalid/corrupt SQLite {db_id}: {msg}")
        if len(invalid_sqlite) > 5:
            errors.append(
                f"... and {len(invalid_sqlite) - 5} more invalid SQLite files"
            )

    total = len(tables_list)
    ok_db = total - len(missing_db) - len(invalid_sqlite)
    print(f"Spider data dir: {SPIDER_DATA_DIR}")
    print(
        f"Required JSON: {'OK' if not any('Missing file' in e for e in errors) else 'MISSING'}"
    )
    print(
        f"Databases: {ok_db}/{total} OK, {len(missing_db)} missing, {len(invalid_sqlite)} invalid"
    )
    if errors:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
