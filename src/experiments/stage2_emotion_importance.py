"""Do the ridge weights reflect real emotion importance, or just how easy
each emotion is to predict?

Fits two heads per unit: one on the true (GT) emotion ratings, one on the
Stage-1 predicted emotions. If a low weight in the predicted setting is
just because that emotion is hard to predict (not because it doesn't
matter), the GT-fit weight should be higher.

Output: output/stage2_emotion_importance/per_unit.csv, summary.csv
"""
from __future__ import annotations

import pandas as pd

from src.data.data import CORE7, DOMAINS
from src.modeling.heads import make_head

#: A coefficient this small is zero for our purposes, and the sign of a zero is
#: not a fact about anyone's taste. Ridge shrinks weak emotions all the way
#: down: 21 of 387 GT weights come out exactly 0.0 and 19 more land below 1e-9
#: (values like 5e-31). Whether such a weight ends up at +5e-31, -3e-31 or 0.0
#: depends on the platform's last bit, so testing `w > 0` moved the reported
#: percentage by one unit between machines. Requiring a real margin makes the
#: count identical everywhere. Nothing sits between 1e-9 and 1e-3, so the
#: threshold has no effect on any coefficient that means anything.
SIGN_TOL = 1e-9


def run(cfg, pipeline):
    out_dir = cfg.run_dir("stage2_emotion_importance")
    rows = []
    for fold in pipeline.split.folds():
        feats = pipeline.backbone.features_for_fold(fold.index)
        for dom in DOMAINS:
            _, _, _, meds = pipeline.shared_context(fold, dom, feats)
            emotion = meds["emotion"]
            for unit in pipeline.iter_units(fold, dom, feats):
                gt_head = make_head("ridge", cfg).fit(unit.E_train, unit.y_train)
                pred_head = make_head("ridge", cfg).fit(
                    emotion.transform(unit.X_train), unit.y_train)
                w_gt = gt_head.weights()
                w_pred = pred_head.weights()
                for j, e in enumerate(CORE7):
                    rows.append(dict(fold=unit.fold, domain=unit.domain,
                                     user_id=unit.user_id, emotion=e,
                                     w_gt=w_gt[j], w_pred=w_pred[j]))
        print(f"  fold {fold.index} done ({len(rows)} rows)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_unit.csv", index=False)

    lines = []
    for setting, col in [("gt", "w_gt"), ("pred", "w_pred")]:
        for e in CORE7:
            d = df[df.emotion == e]
            row = dict(setting=setting, emotion=e)
            for dom in DOMAINS:
                dd = d[d.domain == dom][col]
                row[f"{dom}_mean"] = dd.abs().mean()
                row[f"{dom}_sd"] = dd.abs().std()
                row[f"{dom}_pct_pos"] = 100 * (dd > SIGN_TOL).mean()
            all_ = d[col]
            row["avg_mean"] = all_.abs().mean()
            row["avg_sd"] = all_.abs().std()
            row["avg_pct_pos"] = 100 * (all_ > SIGN_TOL).mean()
            lines.append(row)

    summary = pd.DataFrame(lines)
    summary["rank"] = summary.groupby("setting")["avg_mean"].rank(ascending=False).astype(int)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return summary
