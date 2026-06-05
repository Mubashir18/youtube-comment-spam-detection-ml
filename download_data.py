"""download_data.py

Download and normalize an alternative dataset for the "extra" submission.

Dataset:
  - UCI "YouTube Spam Collection"
  - https://archive.ics.uci.edu/dataset/380/youtube+spam+collection+dataset
  - Download: https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip
  - License: CC BY 4.0

Output:
  - data/youtube_spam.csv with columns: label (ham/spam), message (text)

Notes:
  - Drops COMMENT_ID/AUTHOR/DATE to minimize risk of personal identifiers.
"""

from __future__ import annotations

import os
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd


UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip"


def _read_csv_loose(path: Path) -> pd.DataFrame:
    # The dataset is small, but CSVs may contain quotes/commas in CONTENT.
    # Use a permissive parser configuration.
    return pd.read_csv(
        path,
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip",
    )


def download_and_prepare(output_csv: Path) -> None:
    data_dir = output_csv.parent
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "youtube_spam_collection.zip"
    extract_dir = data_dir / "youtube_spam_collection_raw"
    extract_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading UCI YouTube Spam Collection dataset...")
    urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
    print(f"[OK] Downloaded: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"[OK] Extracted to: {extract_dir}")

    csv_paths = sorted(extract_dir.glob("Youtube*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No Youtube*.csv files found under {extract_dir}")

    frames: list[pd.DataFrame] = []
    for p in csv_paths:
        df = _read_csv_loose(p)
        # Expected columns: COMMENT_ID, AUTHOR, DATE, CONTENT, CLASS
        if "CONTENT" not in df.columns or "CLASS" not in df.columns:
            raise ValueError(f"Unexpected columns in {p.name}: {list(df.columns)}")
        frames.append(df[["CONTENT", "CLASS"]].copy())

    full = pd.concat(frames, ignore_index=True)
    full = full.rename(columns={"CONTENT": "message", "CLASS": "label"})
    full = full.dropna(subset=["message", "label"]).copy()

    # CLASS: 1=spam, 0=ham
    full["label"] = full["label"].astype(int).map({0: "ham", 1: "spam"})
    full = full[["label", "message"]]

    # Basic normalization
    full["message"] = full["message"].astype(str)
    full = full.reset_index(drop=True)

    output_csv.write_text("", encoding="utf-8")  # ensure path is writable
    full.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"[OK] Saved normalized dataset: {output_csv}")
    print("\nDataset Info:")
    print(f"  Total rows: {len(full)}")
    print(f"  Spam: {(full['label'] == 'spam').sum()}")
    print(f"  Ham: {(full['label'] == 'ham').sum()}")
    print(f"  Spam %: {100 * (full['label'] == 'spam').mean():.2f}%")


def main() -> int:
    out_csv = Path("data") / "youtube_spam.csv"
    try:
        download_and_prepare(out_csv)
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
