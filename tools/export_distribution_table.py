"""Export the distributional Stage-1 table as a plain CSV, per backbone.

The notebook renders this table with styling; this writes the same numbers
with no formatting so they can be opened in Excel and laid out by hand.

    uv run python tools/export_distribution_table.py
    uv run python tools/export_distribution_table.py --variant B --head ridge

Writes output/tables/distribution_<backbone>_<variant>.csv and prints each
table. Every number is recomputed from output/raw_all.csv at run time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "output" / "raw_all.csv"
OUT = ROOT / "output" / "tables"

UNIT = ["fold", "domain", "user_id"]
MED = ["population", "identity", "emotion", "emotion_sd", "emotion_hist"]
NAME = {"population": "population", "identity": "Direct (512-d)",
        "emotion": "Hybrid 7-d", "emotion_sd": "Hybrid 14-d (+sd)",
        "emotion_hist": "Hybrid 35-d (hist)"}
LABEL = {"clip": "CLIP frozen", "qwen8b": "Qwen3-VL 8B",
         "clip_ft": "CLIP-ft (score)", "clip_ft_emo": "CLIP-ft (emotion)",
         "qwen4b": "Qwen3-VL 4B"}


def build(d: pd.DataFrame) -> pd.DataFrame | None:
    rows = []
    for n in (10, 25, 50, 100):
        g = d[d.n_train == n]
        if g.empty:
            continue
        g = g.groupby(UNIT + ["mediator"], as_index=False)["srocc"].mean()
        p = g.pivot_table(index=UNIT, columns="mediator", values="srocc")
        if not set(MED) <= set(p.columns):
            continue
        p = p.dropna(subset=MED)
        row = {"n_train": n, "units": len(p)}
        for m in MED:
            row[NAME[m]] = round(p[m].mean(), 6)
        # derived, kept to the right so the scores read as a block
        for m in ("emotion", "emotion_sd", "emotion_hist"):
            row[f"{NAME[m]} vs pop p"] = round(wilcoxon(p[m], p["population"])[1], 6)
            row[f"{NAME[m]} gap-to-Direct"] = round((p["identity"] - p[m]).mean(), 6)
        rows.append(row)
    return pd.DataFrame(rows).set_index("n_train") if rows else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="C")
    ap.add_argument("--head", default="ridge")
    args = ap.parse_args()

    raw = pd.read_csv(RAW, low_memory=False)
    raw = raw[(raw["head"] == args.head) & (raw.variant == args.variant)
              & raw.mediator.isin(MED)]
    OUT.mkdir(parents=True, exist_ok=True)

    wrote = 0
    for bb in ["clip", "qwen8b", "clip_ft", "clip_ft_emo", "qwen4b"]:
        t = build(raw[raw.backbone == bb])
        if t is None:
            print(f"{LABEL.get(bb, bb)}: not run yet")
            continue
        f = OUT / f"distribution_{bb}_{args.variant}.csv"
        t.to_csv(f)
        print(f"\n=== {LABEL.get(bb, bb)} | anchor {args.variant} | {args.head} head ===")
        print(t.to_string())
        print(f"-> {f.relative_to(ROOT).as_posix()}")
        wrote += 1
    print(f"\nwrote {wrote} file(s) under {OUT.relative_to(ROOT).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
