"""Repair the sample_file column of Dataset/maked/ratings.csv.

627 rows (36 scenery stimuli) hold the literal string "#NAME?" instead of a
filename. Those filenames all begin with "-" or "--", and a spreadsheet
opening the CSV reads a leading "-" as the start of a formula; saving from
there writes Excel's evaluation error back into the file. Any extraction
script that resolves an image by `sample_file` then silently skips those 36
stimuli, which is why clip_ft and qwen4b cover 6490 images while the older
clip and qwen8b extractions cover 6526.

The intact filenames are in the untouched copy of the dataset (XPASS-VIS).
This script copies them across by sample_id -- it does not guess: it refuses
to run unless every non-broken row already agrees between the two files, so a
mismatch means the source is not the same dataset and nothing is written.

Usage
    python -m src.data.repair_ratings_csv --source <path to XPASS-VIS ratings.csv>
    python -m src.data.repair_ratings_csv --source ... --check   # report only

A timestamped backup of the current file is written next to it before any
change. Re-running is safe: with nothing left to repair it exits cleanly.

*** keep it repaired ***
Do not open Dataset/maked/ratings.csv in Excel or Google Sheets. To view it,
use a text editor, pandas, or a CSV viewer that does not evaluate formulas.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BROKEN = "#NAME?"
ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "Dataset" / "maked" / "ratings.csv"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True,
                    help="ratings.csv from the untouched dataset copy")
    ap.add_argument("--target", default=str(TARGET))
    ap.add_argument("--check", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    src_path, tgt_path = Path(args.source), Path(args.target)
    src, tgt = pd.read_csv(src_path), pd.read_csv(tgt_path)

    broken = tgt["sample_file"].astype(str) == BROKEN
    print(f"target : {tgt_path}")
    print(f"source : {src_path}")
    print(f"rows with {BROKEN}: {int(broken.sum())} "
          f"({tgt.loc[broken, 'sample_id'].nunique()} stimuli)")
    if not broken.any():
        print("nothing to repair.")
        return 0

    # --- refuse to guess -------------------------------------------------
    if len(src) != len(tgt) or list(src.columns) != list(tgt.columns):
        print("ABORT: source and target differ in shape or columns.")
        return 1
    others = [c for c in tgt.columns if c != "sample_file"]
    if not src[others].equals(tgt[others]):
        print("ABORT: the two files disagree outside sample_file; "
              "this source is not the same dataset.")
        return 1
    intact = ~broken
    if not (src.loc[intact, "sample_file"].astype(str).values
            == tgt.loc[intact, "sample_file"].astype(str).values).all():
        print("ABORT: the intact filenames do not match between the files.")
        return 1
    if (src["sample_file"].astype(str) == BROKEN).any():
        print("ABORT: the source is damaged too.")
        return 1

    repaired = src.loc[broken, ["sample_id", "sample_file"]].drop_duplicates()
    print(f"checks passed; {len(repaired)} filenames recoverable, e.g.")
    print(repaired.head(3).to_string(index=False))

    if args.check:
        print("\n--check given, nothing written.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = tgt_path.with_name(f"{tgt_path.stem}.backup_{stamp}{tgt_path.suffix}")
    shutil.copy2(tgt_path, backup)
    print(f"\nbackup  -> {backup.name}")

    tgt.loc[broken, "sample_file"] = src.loc[broken, "sample_file"].values
    tgt.to_csv(tgt_path, index=False)

    after = pd.read_csv(tgt_path)
    left = int((after["sample_file"].astype(str) == BROKEN).sum())
    print(f"written -> {tgt_path.name}  ({BROKEN} rows remaining: {left})")
    return 0 if left == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
