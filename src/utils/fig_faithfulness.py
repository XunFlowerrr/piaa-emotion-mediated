"""Faithfulness figure -- formula swap, concept ablation, weight-vs-empirical.

Reads output/faithfulness/{ablation,formula_swap,weight_vs_empirical}.csv and
computes every mean/error bar directly from them -- nothing hardcoded.

err="sd"  -> bars show +/- 1 sample sd across user-domain units
err="sem" -> bars show +/- 1 standard error of the mean (sd / sqrt(n))

Output: figures_dir/fig_faithfulness_{sd,sem}.pdf
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.data import DOMAINS
from src.utils.plots import AMBER, BLUE, GREY, TEAL, grid, save, setup

DOM_LABEL = {"art": "Artwork", "fashion": "Fashion", "landscape": "Landscape"}
GROUPS = DOMAINS + [None]
XLABELS = [DOM_LABEL[d] for d in DOMAINS] + ["Average"]


def stat(series: pd.Series, err: str) -> tuple[float, float]:
    v = series.to_numpy(float)
    v = v[np.isfinite(v)]
    m, sd = v.mean(), v.std(ddof=1)
    return m, (sd / np.sqrt(len(v)) if err == "sem" else sd)


def grouped(df: pd.DataFrame, col: str, err: str) -> tuple[np.ndarray, np.ndarray]:
    means, errs = [], []
    for dom in GROUPS:
        d = df if dom is None else df[df.domain == dom]
        m, e = stat(d[col], err)
        means.append(m)
        errs.append(e)
    return np.array(means), np.array(errs)


def bar_group(ax, series: dict[str, tuple[np.ndarray, np.ndarray]], colors: dict):
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(XLABELS))
    for i, (label, (means, errs)) in enumerate(series.items()):
        off = (i - (n - 1) / 2) * width
        ax.bar(x + off, means, width, yerr=errs, capsize=2, label=label,
              color=colors[label], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(XLABELS)
    grid(ax)
    ax.set_axisbelow(True)


def panel_swap(ax, swap: pd.DataFrame, err: str):
    series = {
        "Own Formula": grouped(swap, "own_srocc", err),
        "Population Mean": grouped(swap, "pop_srocc", err),
        "Other User (Swap)": grouped(swap, "other_srocc", err),
    }
    colors = {"Own Formula": BLUE, "Population Mean": "#5B9BD5", "Other User (Swap)": "#AFC9E8"}
    bar_group(ax, series, colors)
    ax.set_ylabel("Spearman SROCC")
    ax.set_title("(a) Formula Swap (Personalization)", fontsize=8.5)
    ax.legend(fontsize=6, loc="upper right")


def panel_ablation(ax, abl: pd.DataFrame, err: str):
    series = {
        "Full Model": grouped(abl, "base_srocc", err),
        "w/o Top-1 Concept": grouped(abl, "top1_srocc", err),
        "w/o Avg-1 Concept": grouped(abl, "avg1_srocc", err),
        "w/o Bottom-1 Concept": grouped(abl, "bottom1_srocc", err),
    }
    colors = {"Full Model": BLUE, "w/o Top-1 Concept": "#8C4A0F",
             "w/o Avg-1 Concept": AMBER, "w/o Bottom-1 Concept": "#E8B87A"}
    bar_group(ax, series, colors)
    ax.set_ylabel("Spearman SROCC")
    ax.set_title("(b) Concept Ablation (Importance)", fontsize=8.5)
    ax.legend(fontsize=6, loc="upper right")


def panel_align(ax, align: pd.DataFrame, err: str):
    series = {
        "Spearman": grouped(align, "spearman", err),
        "Pearson": grouped(align, "pearson", err),
    }
    colors = {"Spearman": BLUE, "Pearson": TEAL}
    bar_group(ax, series, colors)
    ax.axhline(0, color="#888888", lw=0.6, ls="--")
    ax.set_ylabel("weight vs. empirical corr.")
    ax.set_title("(c) Weight vs. Empirical Correlation", fontsize=8.5)
    ax.legend(fontsize=6, loc="upper right")


def run(cfg, err: str = "sd"):
    import matplotlib.pyplot as plt

    setup()
    out_dir = cfg.output_dir / "faithfulness"
    abl = pd.read_csv(out_dir / "ablation.csv")
    swap = pd.read_csv(out_dir / "formula_swap.csv")
    align = pd.read_csv(out_dir / "weight_vs_empirical.csv")

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.6))
    panel_swap(axes[0], swap, err)
    panel_ablation(axes[1], abl, err)
    panel_align(axes[2], align, err)
    fig.tight_layout()
    save(fig, cfg.figures_dir / f"fig_faithfulness_{err}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from src.config import Config

    cfg = Config()
    run(cfg, "sd")
    run(cfg, "sem")
