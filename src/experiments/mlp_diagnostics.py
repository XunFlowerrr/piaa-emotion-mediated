"""Did the MLP head actually train, or just underfit to a constant?

Fits MLPHead on a few real units with the pipeline's usual hyperparameters,
but keeps sklearn's loss_curve_/validation_scores_ (normally thrown away)
so we can plot them. Loss should drop by an order of magnitude and flatten;
val R^2 should climb somewhere near where training stops.

Output: output/mlp_diagnostics/loss_curves.csv, loss_curves.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.plots import save, setup

N_UNITS = 6           # how many units to sample for the plot
SAMPLE_SEED = 0


def run(cfg, pipeline):
    out_dir = cfg.run_dir("mlp_diagnostics")
    fold = pipeline.split.load_fold(0)
    feats = pipeline.backbone.features_for_fold(0)
    _, _, _, meds = pipeline.shared_context(fold, "art", feats)
    emotion = meds["emotion"]

    units = list(pipeline.iter_units(fold, "art", feats))
    rng = np.random.RandomState(SAMPLE_SEED)
    sample = rng.choice(len(units), size=min(N_UNITS, len(units)), replace=False)

    rows = []
    curves = []
    for i in sample:
        unit = units[i]
        M_tr = emotion.transform(unit.X_train)
        from src.modeling.heads import MLPHead
        head = MLPHead(cfg, seed=unit.user_id)
        head.fit(M_tr, unit.y_train)
        mlp = head._pipe.named_steps["mlpregressor"]

        rows.append(dict(
            user_id=unit.user_id, best_lr=head.best_lr_, n_iter=mlp.n_iter_,
            first_loss=float(mlp.loss_curve_[0]), final_loss=float(mlp.loss_curve_[-1]),
            loss_ratio=float(mlp.loss_curve_[-1] / mlp.loss_curve_[0]),
            final_val_r2=float(mlp.validation_scores_[-1]),
            best_val_r2=float(max(mlp.validation_scores_)),
            hit_max_iter=bool(mlp.n_iter_ >= cfg.mlp_max_iter),
        ))
        for it, (loss, val) in enumerate(zip(mlp.loss_curve_, mlp.validation_scores_)):
            curves.append(dict(user_id=unit.user_id, iteration=it, train_loss=loss,
                               val_r2=val))
        print(f"  user {unit.user_id}: n_iter={mlp.n_iter_}, "
              f"loss {mlp.loss_curve_[0]:.3f} -> {mlp.loss_curve_[-1]:.4f}, "
              f"val R^2 {mlp.validation_scores_[-1]:.3f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "loss_curves_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(out_dir / "loss_curves.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plot(pd.DataFrame(curves), out_dir / "loss_curves")
    return summary


def plot(curves: pd.DataFrame, path):
    import matplotlib.pyplot as plt

    setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for uid, g in curves.groupby("user_id"):
        axes[0].plot(g.iteration, g.train_loss, lw=1.0, label=f"user {uid}")
        axes[1].plot(g.iteration, g.val_r2, lw=1.0, label=f"user {uid}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("training iteration")
    axes[0].set_ylabel("training loss (MSE, log scale)")
    axes[1].set_xlabel("training iteration")
    axes[1].set_ylabel("validation R^2")
    axes[1].axhline(0, color="grey", lw=0.6, ls="--")
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save(fig, path)
