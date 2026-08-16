"""Central config for every experiment.

Anything that can change the reported numbers lives here, once, and gets
dumped as config.json next to every output so a run can be traced back to
its settings later.

ridge_alphas: logspace(-2, 3, 11), 11 values from 1e-2 to 1e3.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # paths
    data_dir: Path = PROJECT_ROOT / "Dataset" / "maked"
    split_dir: Path = PROJECT_ROOT / "Dataset" / "split_v4_10group"
    features_dir: Path = PROJECT_ROOT / "features"
    output_dir: Path = PROJECT_ROOT / "output"
    figures_dir: Path = PROJECT_ROOT / "paper" / "figures"

    # evaluation protocol
    first_session_only: bool = True #
    n_folds: int = 5
    n_eval: int = 50        # eval images per user, fixed across n_train
    n_train: int = 100      # ratings per user used to fit the head
    split_seed: int = 42    # per-user split seed (used as seed + user_id)
    min_test: int = 20      # skip a user if fewer eval images remain

    backbone: str = "clip"  #"qwen3-vl-4b", "qwen3-vl-8b"

    # ridge head. The grid runs to 1e6, well past the point where a 7- or
    # 512-feature head on <=100 standardized samples is fully shrunk, so the
    # top of the grid is a safety floor the selector can actually reach rather
    # than a cliff it is cut off before. Matters most for stage2_variant B/C,
    # where full shrinkage lands on the population formula instead of on a
    # constant.
    ridge_alphas: tuple = field(default_factory=lambda: tuple(np.logspace(-2, 6, 17)))

    # Lasso / ElasticNet personal heads. Their penalty is on a different scale
    # from ridge's: with standardized features the smallest alpha that zeroes
    # every coefficient is order 1, so the grid runs from far below any useful
    # penalty up past full sparsity. Same tie-break rule as ridge (strongest
    # penalty wins a tie), which here means the sparsest model.
    sparse_alphas: tuple = field(default_factory=lambda: tuple(np.logspace(-4, 1, 17)))
    elastic_l1_ratio: float = 0.5
    sparse_max_iter: int = 5000

    # Stage-2 variant (how the personal head relates to the population model):
    #   plain  ordinary ridge on the mediator, shrinks toward 0
    #   A      append the GIAA prediction as an extra feature, so w_pop=1 with
    #          all other weights 0 reproduces the population model exactly
    #   B      shrink the weights toward w_pop (the pooled training-group
    #          Stage-2 coefficients) instead of toward 0
    #   C      fit the head on the residual y - y_pop
    stage2_variant: str = "plain"

    # Which mediators the variant applies to. Direct is in the list because
    # otherwise Hybrid alone would carry the population prior and its edge over
    # Direct would be a fact about the prior, not about the bottleneck. PCA is
    # in it because it is a real alternative concept space, so it has to be
    # compared to Hybrid on equal terms.
    #
    # Random and Shuffled are deliberately left out. Their whole job is to show
    # that a mediator with no content buys nothing, and handing them the
    # population prediction gives them content: measured at 100 ratings,
    # Shuffled rises from .245 to .410 and Random from .197 to .390, both
    # landing next to the population baseline (.416) purely on the borrowed
    # prior. A control that scores like the population model is no longer
    # controlling for anything.
    stage2_variant_mediators: tuple = ("identity", "pca", "emotion")

    # MLP head
    mlp_hidden: int = 128
    mlp_alpha: float = 0.0
    mlp_lr_grid: tuple = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    mlp_max_iter: int = 2000           # fixed epoch budget, stated in advance
    mlp_early_stopping: bool = False   # no internal validation split -- see heads.py
    mlp_validation_fraction: float = 0.15   # unused while early_stopping=False
    mlp_n_iter_no_change: int = 20
    mlp_search_val_frac: float = 0.2   # val fraction when searching lr (personal head only)

    mediator_width: int = 7

    def dump(self, path: Path) -> None:
        # make config.json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {}
        for k, v in asdict(self).items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                # numeric grids dump as floats; name lists (e.g.
                # stage2_variant_mediators) dump as-is
                d[k] = [float(x) if isinstance(x, (int, float)) else x
                        for x in v]
            else:
                d[k] = v
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    def run_dir(self, name: str) -> Path:
        d = Path(self.output_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        self.dump(d / "config.json")
        return d
