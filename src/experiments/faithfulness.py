"""Is the per-user formula actually a faithful explanation, or just numbers
that happen to fit?

Two checks:
1. Formula swap - score a user with their own formula vs. the population-
   mean formula vs. another random user's formula. If formulas are really
   personal, "own" should win.
2. Weight vs. empirical correlation - do the 7 ridge weights line up with
   how much each emotion actually correlates with that user's ratings?
   Should be positive if the formula means what it looks like it means.

Output: output/faithfulness/{formula_swap,weight_vs_empirical}.csv, summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.data import CORE7, DOMAINS
from src.utils.metrics import mean_sd, plcc, srocc, wilcoxon_paired

N_SWAP = 5          # number of other users' formulas sampled per unit
SWAP_SEED = 0


def run(cfg, pipeline):
    out_dir = cfg.run_dir("faithfulness")
    store = pipeline.collect_user_heads(mediator="emotion")

    swap = formula_swap(store)
    swap.to_csv(out_dir / "formula_swap.csv", index=False)

    align = weight_vs_empirical(store)
    align.to_csv(out_dir / "weight_vs_empirical.csv", index=False)

    summary = pd.concat([summarize_swap(swap), summarize_align(align)],
                        keys=["formula_swap", "weight_vs_empirical"],
                        names=["test"])
    summary.to_csv(out_dir / "summary.csv")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    return swap, align


def formula_swap(store) -> pd.DataFrame:
    by_dom: dict[str, list[int]] = {}
    for i, s in enumerate(store):
        by_dom.setdefault(s["domain"], []).append(i)

    rng = np.random.RandomState(SWAP_SEED)
    rows = []
    for i, s in enumerate(store):
        pool = [j for j in by_dom[s["domain"]] if j != i]
        if len(pool) < 2:
            continue
        y, M = s["y_eval"], s["M_eval"]
        own = s["head"].predict(M)
        pick = rng.choice(pool, size=min(N_SWAP, len(pool)), replace=False)
        others = [store[j]["head"].predict(M) for j in pick]
        # population-mean formula = average prediction of everyone else in the domain
        pop = np.mean([store[j]["head"].predict(M) for j in pool], axis=0)
        rows.append(dict(
            fold=s["fold"], domain=s["domain"], user_id=s["user_id"],
            own_srocc=srocc(y, own), own_plcc=plcc(y, own),
            other_srocc=float(np.mean([srocc(y, o) for o in others])),
            other_plcc=float(np.mean([plcc(y, o) for o in others])),
            pop_srocc=srocc(y, pop), pop_plcc=plcc(y, pop)))
    return pd.DataFrame(rows)


def weight_vs_empirical(store) -> pd.DataFrame:
    rows = []
    for s in store:
        w = s["head"].weights()
        y, E = s["y_eval"], s["E_eval"]
        emp = np.array([np.corrcoef(E[:, j], y)[0, 1] if E[:, j].std() > 0 else np.nan
                        for j in range(len(CORE7))], float)
        ok = np.isfinite(emp)
        if ok.sum() < 3:
            continue
        rows.append(dict(fold=s["fold"], domain=s["domain"], user_id=s["user_id"],
                         spearman=float(spearmanr(w[ok], emp[ok]).statistic),
                         pearson=float(np.corrcoef(w[ok], emp[ok])[0, 1])))
    return pd.DataFrame(rows)


def summarize_swap(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, k in [("own", "own"), ("population_mean", "pop"), ("other_user", "other")]:
        r = dict(formula=label)
        for m in ("srocc", "plcc"):
            mean, sd = mean_sd(b[f"{k}_{m}"])
            r[f"{m}_mean"], r[f"{m}_sd"] = mean, sd
            if k != "own":
                p = wilcoxon_paired(b[f"own_{m}"], b[f"{k}_{m}"])
                r[f"{m}_sig_vs_own"] = bool(np.isfinite(p) and p < 0.05)
            else:
                r[f"{m}_sig_vs_own"] = False
        rows.append(r)
    return pd.DataFrame(rows).set_index("formula")


def summarize_align(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dom in DOMAINS + ["ALL"]:
        d = b if dom == "ALL" else b[b.domain == dom]
        r = dict(domain=dom, pct_positive=100 * float((d["spearman"] > 0).mean()))
        for m in ("spearman", "pearson"):
            mean, sd = mean_sd(d[m])
            zero = np.zeros(len(d))
            p = wilcoxon_paired(d[m], zero)
            r[f"{m}_mean"], r[f"{m}_sd"] = mean, sd
            r[f"{m}_sig"] = bool(np.isfinite(p) and p < 0.05)
        rows.append(r)
    return pd.DataFrame(rows).set_index("domain")
