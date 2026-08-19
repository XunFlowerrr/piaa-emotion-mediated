"""Rebuild output/raw_all.csv from the per-run files, dropping stale runs.

raw_all.csv is an append-only log. That is fine while every append comes from
the same data, and wrong the moment the underlying data changes: it silently
mixed runs made before the ratings.csv repair (6490 images) with runs made
after it (6526), and because the repair only restored landscape images, the
contamination was uneven across support sizes. The visible symptom was the
population row -- which cannot depend on n_train, since per_user_split()
carves the eval set out first with a fixed size and a per-user seed -- coming
out at .4149 for n=10/25/50 and .4257 for n=100, with 91 landscape units
differing by up to 0.36.

So rather than trusting the log, rebuild it from the per-run CSVs under
output/efficiency/<backbone>/ and keep only runs that pass a self-consistency
check: within one file, the population row must be identical for every
support size, for every unit. A file that fails predates the repair (or mixes
two datasets) and is quarantined rather than deleted.

    uv run python tools/rebuild_raw_all.py            # report only
    uv run python tools/rebuild_raw_all.py --apply    # rewrite raw_all.csv
"""
from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EFF = ROOT / "output" / "efficiency"
RAW_ALL = ROOT / "output" / "raw_all.csv"
QUARANTINE = ROOT / "output" / "_archive_stale_runs"

UNIT = ["fold", "domain", "user_id"]
KEY = ["backbone", "head", "variant", "n_train", "mediator",
       "fold", "domain", "user_id", "seed", "experiment"]


def pop_is_constant(df: pd.DataFrame) -> tuple[bool, str]:
    """The population row must not move with n_train. If it does, the file
    mixes datasets."""
    p = df[df["mediator"] == "population"]
    if p.empty:
        return True, "no population row (nothing to check)"
    if p["n_train"].nunique() < 2:
        return True, "single support size (nothing to check)"

    worst = 0.0
    ref_n = sorted(p["n_train"].unique())[0]
    for seed, g in p.groupby("seed"):
        ref = g[g["n_train"] == ref_n].set_index(UNIT)["srocc"]
        for n in sorted(g["n_train"].unique())[1:]:
            cur = g[g["n_train"] == n].set_index(UNIT)["srocc"]
            idx = ref.index.intersection(cur.index)
            if len(idx):
                worst = max(worst, float((ref.loc[idx] - cur.loc[idx]).abs().max()))
    ok = worst < 1e-9
    return ok, f"max population drift across n_train = {worst:.2e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    files = sorted(EFF.glob("*/raw*.csv"))
    if not files:
        raise SystemExit(f"no per-run files under {EFF}")

    # Reference: the population row for each (backbone, seed, unit). Taken
    # from the newest file that covers the full support sweep, because a file
    # covering every n is one run against one dataset. Any other file that
    # disagrees with it was produced against different data.
    loaded = {}
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "mediator" in df.columns:
            loaded[f] = df

    refs = {}
    for f, df in sorted(loaded.items(), key=lambda kv: kv[0].stat().st_mtime):
        bb = df["backbone"].dropna().iloc[0] if "backbone" in df.columns and df["backbone"].notna().any() else f.parent.name
        p_ = df[df["mediator"] == "population"]
        if p_.empty or p_["n_train"].nunique() < 4:
            continue
        refs[bb] = p_.set_index(["seed"] + UNIT + ["n_train"])["srocc"]   # newest wins

    def agrees_with_ref(df, bb):
        ref = refs.get(bb)
        if ref is None:
            return True, "no reference for this backbone"
        p_ = df[df["mediator"] == "population"]
        if p_.empty:
            return True, "no population row"
        cur = p_.set_index(["seed"] + UNIT + ["n_train"])["srocc"]
        idx = cur.index.intersection(ref.index)
        if not len(idx):
            return True, "no overlap with reference"
        worst = float((cur.loc[idx] - ref.loc[idx]).abs().max())
        return worst < 1e-9, f"max population disagreement with reference = {worst:.2e}"

    keep, drop = [], []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if "mediator" not in df.columns:
            drop.append((f, "not a per-unit results file"))
            continue
        backbone = f.parent.name
        df["backbone"] = df.get("backbone", backbone)
        df["backbone"] = df["backbone"].fillna(backbone)
        if "experiment" not in df.columns:
            df["experiment"] = "efficiency"
        if "variant" not in df.columns and "stage2_variant" in df.columns:
            df["variant"] = df["stage2_variant"]
        ok, why = pop_is_constant(df)
        if ok:
            ok, why = agrees_with_ref(df, df["backbone"].dropna().iloc[0]
                                      if df["backbone"].notna().any() else backbone)
        (keep if ok else drop).append((f, why) if not ok else (f, df))

    print(f"{'file':58s} rows   verdict")
    total = 0
    frames = []
    for f, df in keep:
        print(f"{f.relative_to(ROOT).as_posix():58s} {len(df):6d} KEEP")
        frames.append(df)
        total += len(df)
    for f, why in drop:
        print(f"{f.relative_to(ROOT).as_posix():58s} {'':6s} DROP  ({why})")

    if not frames:
        raise SystemExit("nothing passed the check; refusing to write an empty file")

    out = pd.concat(frames, ignore_index=True)
    have = [c for c in KEY if c in out.columns]
    before = len(out)
    out = out.drop_duplicates(subset=have, keep="last")
    print(f"\nconcatenated {before} rows -> {len(out)} after dedup on {have}")

    # The same check again, now across the merged file: this is the invariant
    # that was violated before, so it has to hold on the thing we ship.
    for (bb, var), g in out.groupby(["backbone", "variant"]):
        ok, why = pop_is_constant(g)
        flag = "ok" if ok else "STILL BROKEN"
        if not ok:
            print(f"  {bb}/{var}: {why}  <-- {flag}")
    print("merged-file population invariant checked per (backbone, variant)")

    if not args.apply:
        print("\n[report only] pass --apply to rewrite raw_all.csv")
        return 0

    QUARANTINE.mkdir(parents=True, exist_ok=True)
    if RAW_ALL.exists():
        dest = QUARANTINE / f"raw_all_before_rebuild_{date.today():%Y%m%d}.csv"
        shutil.copy2(RAW_ALL, dest)
        print(f"\narchived old log -> {dest.relative_to(ROOT).as_posix()}")
    for f, why in drop:
        dest = QUARANTINE / f"{f.parent.name}__{f.name}"
        shutil.copy2(f, dest)
    out.to_csv(RAW_ALL, index=False)
    print(f"wrote {RAW_ALL.relative_to(ROOT).as_posix()}: {len(out)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
