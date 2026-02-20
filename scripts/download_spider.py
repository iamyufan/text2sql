"""Download Spider 1.0 dataset and extract so spider_data/database/{db_id}/{db_id}.sqlite exist.

Official zip: https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view?usp=sharing
Run from project root. Uses gdown if available; otherwise prints manual instructions.
"""

import os
import sys
import zipfile
from pathlib import Path

# Project root = parent of scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SPIDER_DATA_DIR = Path(
    os.environ.get("SPIDER_DATA_DIR", str(PROJECT_ROOT / "spider_data"))
).resolve()

GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"


def download_with_gdown() -> Path | None:
    try:
        import gdown
    except ImportError:
        return None
    out = SPIDER_DATA_DIR.parent / "spider.zip"
    url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
    gdown.download(url, str(out), quiet=False, fuzzy=True)
    return out if out.exists() else None


def main() -> None:
    print(f"Spider data directory: {SPIDER_DATA_DIR}")
    SPIDER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = SPIDER_DATA_DIR.parent / "spider.zip"
    if not zip_path.exists():
        print("Attempting download with gdown...")
        downloaded = download_with_gdown()
        if downloaded is None:
            if "gdown" not in sys.modules:
                print("gdown not installed. Install with: uv add --dev gdown")
                print("Or download manually:")
                print(
                    f"  1. Open https://drive.google.com/file/d/{GDRIVE_ID}/view?usp=sharing"
                )
                print(
                    "  2. Download the zip and save as spider.zip in the project root."
                )
                print(f"  3. Re-run this script.")
            sys.exit(1)
        zip_path = downloaded
    else:
        print(f"Using existing zip: {zip_path}")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Spider zip typically has top-level train_spider.json, database/, etc.
        for name in zf.namelist():
            if name.startswith("database/") or name in (
                "train_spider.json",
                "train_others.json",
                "dev.json",
                "tables.json",
                "README.txt",
            ):
                dest = SPIDER_DATA_DIR / name
                if name.endswith("/"):
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(name))
    print("Done. Run: uv run python scripts/verify_spider_data.py")


if __name__ == "__main__":
    main()
