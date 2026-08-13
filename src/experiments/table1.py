"""Main results table: every mediator x head combo at 100 ratings/user.

Rows: Population (no personal ratings used), Direct (no mediator, 512
params/user), Random/Shuffled (content-free mediators), PCA (unsupervised),
Hybrid (our 7-dim emotion mediator), plus two upper bounds -- GT emotions
(uses true ratings instead of predicted ones) and test-retest reliability.

Everything is scored on the same users/images so rows compare directly with
a paired test.

Every row is run under 3 seeds (0,1,2) and averaged per unit before
summarizing, since random/shuffled mediators and the MLP head are stochastic.
seed=0 alone reproduces the original single-seed run bit-for-bit.

Writes per_unit.csv (seed-averaged), per_unit_by_seed.csv (raw, one row per
seed), and summary.csv (mean/sd per domain, plus a best-row flag and
Wilcoxon significance vs. Hybrid+Ridge) to output/table1/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.data import DOMAINS
from src.utils.metrics import mean_sd, plcc, srocc, wilcoxon_paired

#: row order (mediator, head)
ROWS = [
    ("population", "ridge"), ("population", "mlp"),
    ("identity", "ridge"), ("identity", "mlp"),
    ("random", "ridge"), ("random", "mlp"),
    ("shuffled", "ridge"), ("shuffled", "mlp"),
    ("pca", "ridge"), ("pca", "mlp"),
    ("emotion", "ridge"), ("emotion", "mlp"),
]

REFERENCE = ("emotion", "ridge")   # row that Wilcoxon significance is computed against

#: rows involving random/shuffled mediators or an MLP head are stochastic
#: (random projection, label permutation, MLP weight init + internal split);
#: we repeat the whole grid under these seeds and average per unit so a
#: single unlucky draw doesn't set the reported number. seed=0 reproduces
#: the original single-seed run bit-for-bit (see pipeline.run_grid).
SEEDS = (0, 1, 2)


def run_one_seed(cfg, pipeline, seed: int) -> pd.DataFrame:
    """One seed's full grid, written to its own file so seeds can run as
    separate processes in parallel (they are completely independent)."""
    out_dir = cfg.run_dir("table1")
    print(f"[table1] seed {seed}")
    d = pipeline.run_grid(
        mediators=["identity", "random", "shuffled", "pca", "emotion"],
        heads=["ridge", "mlp"],
        include_population=True,
        include_gt_upper_bound=True,
        seed=seed,
    )
    d["seed"] = seed
    d.to_csv(out_dir / f"per_unit_seed{seed}.csv", index=False)
    print(f"[table1] seed {seed} written ({len(d)} rows)")
    return d


def run(cfg, pipeline, dataset, seeds=None) -> pd.DataFrame:
    """Run the seeds that aren't on disk yet, then merge and summarize."""
    out_dir = cfg.run_dir("table1")
    seeds = list(SEEDS if seeds is None else seeds)

    raw = []
    for s in seeds:
        f = out_dir / f"per_unit_seed{s}.csv"
        if f.exists():
            print(f"[table1] seed {s} already on disk, reusing")
            raw.append(pd.read_csv(f))
        else:
            raw.append(run_one_seed(cfg, pipeline, s))

    raw_df = pd.concat(raw, ignore_index=True)
    raw_df.to_csv(out_dir / "per_unit_by_seed.csv", index=False)

    key = ["mediator", "head", "fold", "domain", "user_id"]
    value_cols = ["ccc", "srocc", "plcc", "eff_dof"]
    df = raw_df.groupby(key, as_index=False)[value_cols].mean()
    df.to_csv(out_dir / "per_unit.csv", index=False)

    retest = test_retest(cfg, dataset)
    summary = summarize(df, retest)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


def test_retest(cfg, dataset) -> dict:
    """User's self-agreement across sessions (a second upper bound)

    Uses the second-occurrence ratings that were held out by the
    first-session filter.
    """
    pairs = dataset.retest_pairs(cfg.data_dir)
    out = {}
    for dom in DOMAINS:
        d = pairs[pairs["domain"] == dom]
        y1 = d["overall_r1"].to_numpy(float)
        y2 = d["overall_r2"].to_numpy(float)
        out[dom] = dict(srocc=srocc(y1, y2), plcc=plcc(y1, y2), n=len(d))
    y1 = pairs["overall_r1"].to_numpy(float)
    y2 = pairs["overall_r2"].to_numpy(float)
    out["avg"] = dict(srocc=srocc(y1, y2), plcc=plcc(y1, y2), n=len(pairs))
    return out


def _key(df, med, head):
    # bracket notation for "head" -- df.head is DataFrame.head(), not the column
    return df[(df.mediator == med) & (df["head"] == head)].set_index(
        ["fold", "domain", "user_id"])


def summarize(df, retest) -> pd.DataFrame:
    ref = _key(df, *REFERENCE)
    rows = []
    for med, head in ROWS + [("gt_emotion", "ridge")]:
        s = _key(df, med, head)
        if len(s) == 0:
            continue
        r = dict(mediator=med, head=head, eff_dof=s["eff_dof"].mean())
        for dom in DOMAINS:
            d = s[s.index.get_level_values("domain") == dom]
            for m in ("srocc", "plcc"):
                mean, sd = mean_sd(d[m])
                r[f"{dom}_{m}_mean"], r[f"{dom}_{m}_sd"] = mean, sd
        for m in ("srocc", "plcc"):
            mean, sd = mean_sd(s[m])
            r[f"avg_{m}_mean"], r[f"avg_{m}_sd"] = mean, sd
            j = s[[m]].merge(ref[[m]], left_index=True, right_index=True,
                             suffixes=("", "_ref")).dropna()
            p = (np.nan if (med, head) == REFERENCE
                 else wilcoxon_paired(j[m], j[f"{m}_ref"]))
            r[f"avg_{m}_sig"] = bool(np.isfinite(p) and p < 0.05)
        rows.append(r)

    rt = dict(mediator="test_retest", head="---", eff_dof=np.nan)
    for dom in DOMAINS:
        rt[f"{dom}_srocc_mean"] = retest[dom]["srocc"]
        rt[f"{dom}_plcc_mean"] = retest[dom]["plcc"]
        rt[f"{dom}_srocc_sd"] = rt[f"{dom}_plcc_sd"] = np.nan
    for m in ("srocc", "plcc"):
        rt[f"avg_{m}_mean"] = retest["avg"][m]
        rt[f"avg_{m}_sd"] = np.nan
        rt[f"avg_{m}_sig"] = False
    rows.append(rt)

    out = pd.DataFrame(rows)
    upper_bound = out.mediator.isin(["gt_emotion", "test_retest"])
    for m in ("srocc", "plcc"):
        top = out.loc[~upper_bound, f"avg_{m}_mean"].max()
        out[f"avg_{m}_best"] = np.isclose(out[f"avg_{m}_mean"], top) & ~upper_bound
    return out
