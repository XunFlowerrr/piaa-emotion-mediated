"""Does the emotion mediator still help as the image backbone gets stronger?

Direct vs. Hybrid on 4 backbones, 100 ratings/user, same users/images so we
can use a paired test. Fine-tuned backbones must load per-fold features
(see modeling/backbones.py) or this leaks.

Output: output/backbone/per_unit.csv, summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.modeling.backbones import backbone_label
from src.utils.metrics import mean_sd, wilcoxon_paired


def run(cfg, backbone_names: list[str]) -> pd.DataFrame:
    from main import build

    out_dir = cfg.run_dir("backbone")
    frames = []
    for name in backbone_names:
        print(f"[backbone] {name}")
        _, _, _, pipe = build(cfg, name)
        df = pipe.run_grid(mediators=["identity", "emotion"], heads=["ridge"],
                           include_population=False, include_gt_upper_bound=False)
        df["backbone"] = name
        frames.append(df)
    allf = pd.concat(frames, ignore_index=True)
    allf.to_csv(out_dir / "per_unit.csv", index=False)

    summary = summarize(allf, backbone_names)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return allf


def summarize(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        d = df[df.backbone == name]
        direct = d[d.mediator == "identity"].set_index(["fold", "domain", "user_id"])
        hybrid = d[d.mediator == "emotion"].set_index(["fold", "domain", "user_id"])
        r = dict(backbone=name, label=backbone_label(name))
        for m in ("srocc", "plcc"):
            j = direct[[m]].merge(hybrid[[m]], left_index=True, right_index=True,
                                  suffixes=("_direct", "_hybrid")).dropna()
            dm, dsd = mean_sd(j[f"{m}_direct"])
            hm, hsd = mean_sd(j[f"{m}_hybrid"])
            p = wilcoxon_paired(j[f"{m}_hybrid"], j[f"{m}_direct"])
            r[f"direct_{m}_mean"], r[f"direct_{m}_sd"] = dm, dsd
            r[f"hybrid_{m}_mean"], r[f"hybrid_{m}_sd"] = hm, hsd
            r[f"hybrid_{m}_best"] = bool(hm >= dm)
            r[f"hybrid_{m}_sig"] = bool(np.isfinite(p) and p < 0.05)
        rows.append(r)
    return pd.DataFrame(rows)
