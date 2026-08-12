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

    # ridge head
    ridge_alphas: tuple = field(default_factory=lambda: tuple(np.logspace(-2, 3, 11)))

    # MLP head
    mlp_hidden: int = 128
    mlp_alpha: float = 0.0
    mlp_lr_grid: tuple = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    mlp_max_iter: int = 2000 
    mlp_early_stopping: bool = True
    mlp_validation_fraction: float = 0.15
    mlp_n_iter_no_change: int = 20
    mlp_search_val_frac: float = 0.2   # val fraction when searching lr

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
                d[k] = [float(x) for x in v]
            else:
                d[k] = v
        path.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

    def run_dir(self, name: str) -> Path:\
        d = Path(self.output_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        self.dump(d / "config.json")
        return d
