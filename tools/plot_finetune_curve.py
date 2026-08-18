"""Training curves for the CLIP-ft (emotion) fine-tune.

*** provenance ***
These numbers are transcribed from the Kaggle console log of 18 Aug 2026,
which covers folds 0 and 1 only -- that run was interrupted during fold 2.
The .npz files actually used by the experiments carry no loss history
(finetune_clip_perfold.py printed the losses but did not save them; it does
now), so it CANNOT be verified that this log is the same run that produced
them. Hyperparameters are identical (8 epochs, batch 32, lr 1e-5, seed 42).
Treat the shape of the curve as indicative, not as a record of the run
behind the reported features. Re-run the fine-tune to get a verified curve.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (epoch, train_loss, val_loss), from the Kaggle log
FOLD0 = [(0, .7093, .6596), (1, .4777, .5759), (2, .3060, .5774), (3, .1963, .5943),
         (4, .1260, .5992), (5, .0812, .6058), (6, .0553, .6012), (7, .0388, .5963)]
FOLD1 = [(0, .7398, .6309), (1, .5076, .5942), (2, .3433, .6032), (3, .2255, .5940),
         (4, .1442, .6174), (5, .0913, .6225), (6, .0607, .6283), (7, .0416, .6239)]

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, hist, name in zip(axes, (FOLD0, FOLD1), ("Fold 0", "Fold 1")):
        ep = [h[0] for h in hist]
        tr = [h[1] for h in hist]
        va = [h[2] for h in hist]
        ax.plot(ep, tr, "o-", color="#2c7fb8", label="train MSE")
        ax.plot(ep, va, "s-", color="#d95f02", label="val MSE")
        b = min(range(len(va)), key=lambda i: va[i])
        ax.axvline(b, color="grey", ls=":", lw=1)
        ax.annotate(f"checkpoint kept\n(epoch {b}, val {va[b]:.4f})",
                    xy=(b, va[b]), xytext=(b + 0.6, va[b] + 0.16),
                    fontsize=7, color="grey",
                    arrowprops=dict(arrowstyle="->", color="grey", lw=.8))
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("MSE (7 emotions)")
    axes[0].legend(fontsize=8, loc="center right")
    fig.suptitle("CLIP-ft (emotion): train and validation loss", fontsize=11)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_clipft_emo_training.{ext}", dpi=200,
                    bbox_inches="tight")
    print("wrote", OUT / "fig_clipft_emo_training.png")
    for name, hist in (("fold0", FOLD0), ("fold1", FOLD1)):
        va = [h[2] for h in hist]
        b = min(range(len(va)), key=lambda i: va[i])
        print(f"  {name}: best val at epoch {b} ({va[b]:.4f}); "
              f"epoch 0 was {va[0]:.4f}; final epoch {va[-1]:.4f}")


if __name__ == "__main__":
    main()
