"""Efficiency curves 

from output/efficiency/per_unit.csv 

err="sd"  -> band is +/- 1 sample sd across user-domain units
err="sem" -> band is +/- 1 standard error of the mean (sd / sqrt(n))

Output: figures_dir/fig_efficiency_{sd,sem}.pdf
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.plots import AMBER, BLUE, GREY, grid, save, setup

METHODS = ("pop_zero", "direct", "hybrid")

STYLE = {
    "pop_zero": dict(color=AMBER, ls=":", marker=None, label="Population (0 params)"),
    "direct":   dict(color=GREY, ls="--", marker="^", label="Direct (512 params)"),
    "hybrid":   dict(color=BLUE, ls="-", marker="o", label="Hybrid (7 params)"),
}


def stats(df: pd.DataFrame, method: str, metric: str, n_list: list[int], err: str, domain: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    d = df[df.model == method]
    if domain is not None:
        d = d[d.domain == domain]
    means, errs = [], []
    for n in n_list:
        v = d[d.n_train == n][metric].to_numpy(float)
        v = v[np.isfinite(v)]
        m, sd = v.mean(), v.std(ddof=1)
        means.append(m)
        errs.append(sd / np.sqrt(len(v)) if err == "sem" else sd)
    return np.array(means), np.array(errs)


def plot_panel(ax, df, metric, n_list, err, domain: str | None = None, annotate_crossover: bool = False):
    for method in METHODS:
        means, errs = stats(df, method, metric, n_list, err, domain)
        s = STYLE[method]
        ms, lw = (4, 1.4) if domain is None else (3.5, 1.2)
        if method == "pop_zero":
            ax.plot(n_list, means, color=s["color"], ls=s["ls"], lw=1.3, label=s["label"])
        else:
            ax.plot(n_list, means, color=s["color"], ls=s["ls"], marker=s["marker"],ms=ms, lw=lw, label=s["label"])
            # error band
            ax.fill_between(n_list, means - errs, means + errs, color=s["color"], alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(n_list)
    ax.set_xticklabels([str(n) for n in n_list])
    grid(ax)
    ax.set_axisbelow(True)

    if annotate_crossover:
        hy, _ = stats(df, "hybrid", metric, n_list, err, domain)
        pop, _ = stats(df, "pop_zero", metric, n_list, err, domain)
        direct, _ = stats(df, "direct", metric, n_list, err, domain)
        
        final_points = [("hybrid", hy[-1], BLUE),("pop_zero", pop[-1], AMBER),("direct", direct[-1], GREY)]
        # Annotate by rank
        final_points.sort(key=lambda item: item[1], reverse=True)
        offsets = [5, -1, -5]
        for rank, (name, value, color) in enumerate(final_points):
            ax.annotate(f"{value:.3f}", xy=(n_list[-1], value), xytext=(6, offsets[rank]),
                       textcoords="offset points", fontsize=7, color=color)

def run(cfg, err: str = "sd", domain_split: bool = False):
    import matplotlib.pyplot as plt

    setup()
    df = pd.read_csv(cfg.output_dir / "efficiency" / "per_unit.csv")
    n_list = sorted(df.n_train.unique().tolist())

    if domain_split:
        domains = ["art", "fashion", "landscape"]
        titles = ["Artwork", "Fashion", "Landscape"]
        fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.4), sharex=True)
        for col, (dom, title) in enumerate(zip(domains, titles)):
            plot_panel(axes[0, col], df, "srocc", n_list, err, domain=dom, annotate_crossover=True)
            axes[0, col].set_title(title, fontsize=9)
            plot_panel(axes[1, col], df, "plcc", n_list, err, domain=dom, annotate_crossover=True)
            axes[1, col].set_xlabel("ratings/user", fontsize=7.5)
        axes[0, 0].set_ylabel("SROCC")
        axes[1, 0].set_ylabel("PLCC")
        axes[0, 0].legend(fontsize=6.5, loc="lower right")
        fig.tight_layout()
        save(fig, cfg.output_dir / f"fig_efficiency_domains_{err}")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
        plot_panel(axes[0], df, "srocc", n_list, err, annotate_crossover=True)
        axes[0].set_ylabel("SROCC")
        axes[0].set_xlabel("ratings per user")
        plot_panel(axes[1], df, "plcc", n_list, err, annotate_crossover=True)
        axes[1].set_ylabel("PLCC")
        axes[1].set_xlabel("ratings per user")
        axes[1].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        save(fig, cfg.figures_dir / f"fig_efficiency_{err}")

if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from src.config import Config

    cfg = Config()
    run(cfg, "sd", domain_split=False)
    run(cfg, "sem", domain_split=False)
