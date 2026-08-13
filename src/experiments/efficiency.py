"""How much does the mediator help when a user has few ratings?

Sweeps support size n (10/25/50/100) against a fixed eval set, so results
are comparable across budgets. Compares Pop-zero (population formula, no
personal data), Direct, and Hybrid.

Output: output/efficiency/per_unit.csv, summary.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.data import DOMAINS
from src.modeling.heads import make_head
from src.utils.metrics import evaluate, mean_sd, wilcoxon_paired


def run(cfg, pipeline, n_list: list[int]) -> pd.DataFrame:
    out_dir = cfg.run_dir("efficiency")
    rows = []
    for fold in pipeline.split.folds():
        feats = pipeline.backbone.features_for_fold(fold.index)
        for dom in DOMAINS:
            Xg, Eg, yg, meds = pipeline.shared_context(fold, dom, feats)
            emotion = meds["emotion"]
            # population-level formula on the emotion mediator (Pop-zero).
            # shared component -> penalty selected on the val user group
            Xv, _, yv = pipeline.val_data(fold, dom, feats)
            popform = make_head("ridge", cfg).fit(
                emotion.transform(Xg), yg, val=(emotion.transform(Xv), yv))

            for n in n_list:
                for unit in pipeline.iter_units(fold, dom, feats, n_train=n):
                    base = dict(n_train=n, fold=unit.fold, domain=unit.domain,
                                user_id=unit.user_id)
                    M_tr = emotion.transform(unit.X_train)
                    M_ev = emotion.transform(unit.X_eval)

                    hy = make_head("ridge", cfg).fit(M_tr, unit.y_train)
                    di = make_head("ridge", cfg).fit(unit.X_train, unit.y_train)
                    rows.append({**base, "model": "pop_zero",
                                 **evaluate(unit.y_eval, popform.predict(M_ev))})
                    rows.append({**base, "model": "direct",
                                 **evaluate(unit.y_eval, di.predict(unit.X_eval))})
                    rows.append({**base, "model": "hybrid",
                                 **evaluate(unit.y_eval, hy.predict(M_ev))})
        print(f"  fold {fold.index} done ({len(rows)} rows)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_unit.csv", index=False)
    summary = summarize(df, n_list)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


def summarize(df: pd.DataFrame, n_list: list[int]) -> pd.DataFrame:
    rows = []
    for n in n_list:
        d = df[df.n_train == n]
        piv = {m: d[d.model == m].set_index(["fold", "domain", "user_id"])
               for m in ("pop_zero", "direct", "hybrid")}
        r = dict(n_train=n)
        for metric in ("srocc", "plcc"):
            means = {m: mean_sd(piv[m][metric]) for m in piv}
            top = max(v[0] for v in means.values() if np.isfinite(v[0]))
            j = piv["hybrid"][[metric]].merge(
                piv["direct"][[metric]], left_index=True, right_index=True,
                suffixes=("_h", "_d")).dropna()
            p = wilcoxon_paired(j[f"{metric}_h"], j[f"{metric}_d"])
            for m in ("pop_zero", "direct", "hybrid"):
                mean, sd = means[m]
                r[f"{m}_{metric}_mean"] = mean
                r[f"{m}_{metric}_sd"] = sd
                r[f"{m}_{metric}_best"] = bool(np.isclose(mean, top))
            r[f"hybrid_{metric}_sig_vs_direct"] = bool(np.isfinite(p) and p < 0.05)
        rows.append(r)
    return pd.DataFrame(rows)
