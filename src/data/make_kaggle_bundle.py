"""Build the single .zip to upload as a Kaggle Dataset.

Kaggle unpacks an uploaded zip for you, so everything the fine-tune needs --
code, ratings, splits, and the image archives -- goes in one file with the
layout the notebook expects:

    piaa_bundle/
      src/...                     the repo's src/ (code)
      Dataset/maked/*.csv         ratings + QIP tables
      Dataset/split_v4_10group/   the fold definitions
      images/art.zip              left zipped; the notebook unpacks them to
      images/fashion.zip          /kaggle/working, which is much faster than
      images/scenery_image.zip    reading ~10k loose files from /kaggle/input

Run it before uploading:

    python -m src.data.make_kaggle_bundle --images <XPASS-VIS>/sample

It refuses to build if ratings.csv still holds the "#NAME?" damage, because a
bundle built from the broken file silently drops 36 scenery images -- the bug
this whole exercise was about.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
IMAGE_ZIPS = ["art.zip", "fashion.zip", "scenery_image.zip"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True,
                    help="folder holding art.zip / fashion.zip / scenery_image.zip")
    ap.add_argument("--out", default=str(ROOT / "piaa_kaggle_bundle.zip"))
    args = ap.parse_args()

    ratings = ROOT / "Dataset" / "maked" / "ratings.csv"
    broken = (pd.read_csv(ratings)["sample_file"].astype(str) == "#NAME?").sum()
    if broken:
        print(f"ABORT: {ratings.name} still has {broken} '#NAME?' rows.\n"
              f"Repair it first:  python -m src.data.repair_ratings_csv --source ...")
        return 1
    print(f"ratings.csv is clean (0 '#NAME?' rows)")

    img_dir = Path(args.images)
    missing = [z for z in IMAGE_ZIPS if not (img_dir / z).exists()]
    if missing:
        print(f"ABORT: missing image archives in {img_dir}: {missing}")
        return 1

    out = Path(args.out)
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted((ROOT / "src").rglob("*.py")):
            z.write(p, Path("piaa_bundle") / p.relative_to(ROOT))
            total += 1
        for p in sorted((ROOT / "Dataset" / "maked").glob("*.csv")):
            z.write(p, Path("piaa_bundle") / p.relative_to(ROOT))
            total += 1
        for p in sorted((ROOT / "Dataset" / "split_v4_10group").rglob("*")):
            if p.is_file():
                z.write(p, Path("piaa_bundle") / p.relative_to(ROOT))
                total += 1
        for name in IMAGE_ZIPS:                 # stored as-is, already compressed
            src = img_dir / name
            print(f"  adding {name} ({src.stat().st_size/1e6:.0f} MB) ...", flush=True)
            z.write(src, f"piaa_bundle/images/{name}")
            total += 1

    print(f"\nwrote {out}  ({out.stat().st_size/1e6:.0f} MB, {total} entries)")
    print("upload it at https://www.kaggle.com/datasets -> New Dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
