"""Summarize per-unit results into CSV: mean, sd, best-in-group, significance.

Convention used by every summary CSV:
    <metric>_mean, <metric>_sd   across units
    <metric>_best                True if best in its comparison group
    <metric>_sig                 paired Wilcoxon vs. a reference row, p < 0.05
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.metrics import mean_sd, wilcoxon_paired


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"  wrote {p}")
    return p


def summarize_groups(df: pd.DataFrame, group_cols: list[str], metrics: list[str],
                     compare_within: list[str] | None = None,
                     reference: dict | None = None) -> pd.DataFrame:
    """One row per group, with mean/sd/best/sig per metric.

    compare_within  restrict "best" comparisons to rows sharing these columns
                     (e.g. keep upper bounds out of the comparison)
    reference        {col: value} identifying the row significance is tested
                     against; omit to skip the sig column
    """
    rows = []
    for key, g in df.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        for m in metrics:
            mean, sd = mean_sd(g[m])
            row[f"{m}_mean"] = mean
            row[f"{m}_sd"] = sd
        rows.append(row)
    out = pd.DataFrame(rows)

    for m in metrics:
        if compare_within:
            out[f"{m}_best"] = out.groupby(compare_within)[f"{m}_mean"].transform(
                lambda s: np.isclose(s, s.max(), equal_nan=False))
        else:
            top = out[f"{m}_mean"].max()
            out[f"{m}_best"] = np.isclose(out[f"{m}_mean"], top)

    if reference is not None:
        ref_mask = np.ones(len(df), dtype=bool)
        for k, v in reference.items():
            ref_mask &= (df[k] == v)
        ref_vals = df.loc[ref_mask].set_index(["fold", "domain", "user_id"])

        for m in metrics:
            sig = []
            for _, row in out.iterrows():
                mask = np.ones(len(df), dtype=bool)
                for c in group_cols:
                    mask &= (df[c] == row[c])
                is_ref = all(row.get(k) == v for k, v in reference.items())
                if is_ref:
                    sig.append(False)
                    continue
                sub = df.loc[mask].set_index(["fold", "domain", "user_id"])
                joined = sub[[m]].join(ref_vals[[m]], rsuffix="_ref").dropna()
                p = wilcoxon_paired(joined[m], joined[f"{m}_ref"])
                sig.append(bool(np.isfinite(p) and p < 0.05))
            out[f"{m}_sig"] = sig

    return out
