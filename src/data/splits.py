"""Leak-free data split protocol (v4, 10 groups).

129 users are split into 10 groups: 7 train, 1 val, 2 test. Rotate over 5
folds so everyone is a test user once.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DOMAINS = ["art", "fashion", "landscape"]

def _read_ids(path: Path) -> list[int]:
    return [int(x) for x in Path(path).read_text(encoding="utf-8").split()]

@dataclass
class Fold:
    index: int
    train_users: set[int]
    val_users: set[int]
    test_users: set[int]
    giaa_images: set[str]

class V4Split:
    """Reads folds from Dataset/split_v4_10group/fold{k}/."""

    def __init__(self, split_dir: str | Path, n_folds: int = 5):
        self.split_dir = Path(split_dir)
        self.n_folds = n_folds
        if not self.split_dir.exists():
            raise FileNotFoundError(f"split folder not found: {self.split_dir}")

    def load_fold(self, i: int) -> Fold:
        d = self.split_dir / f"fold{i}"
        return Fold(
            index=i,
            train_users=set(_read_ids(d / "train_users.txt")),
            val_users=set(_read_ids(d / "val_users.txt")),
            test_users=set(_read_ids(d / "test_users.txt")),
            giaa_images={str(s) for s in _read_ids(d / "giaa_train_images.txt")},
        )

    def folds(self):
        fold_list = []
        for i in range(self.n_folds):
            fold_list.append(self.load_fold(i))
        return fold_list

    def verify_disjoint(self, verbose: bool = True) -> dict:
        report, test_counts = [], {}
        for f in self.folds():
            assert not (f.train_users & f.test_users), f"fold{f.index}: train/test overlap"
            assert not (f.train_users & f.val_users), f"fold{f.index}: train/val overlap"
            assert not (f.val_users & f.test_users), f"fold{f.index}: val/test overlap"
            for u in f.test_users:
                test_counts[u] = test_counts.get(u, 0) + 1
            report.append(dict(fold=f.index, n_train=len(f.train_users),
                               n_val=len(f.val_users), n_test=len(f.test_users),
                               n_giaa_images=len(f.giaa_images)))
        n_users = len(test_counts)
        once = all(v == 1 for v in test_counts.values())
        assert once, "some user is a test user more than once, or never"
        if verbose:
            for r in report:
                print(f"  fold{r['fold']}: train={r['n_train']} val={r['n_val']} "
                      f"test={r['n_test']} giaa_img={r['n_giaa_images']}")
            print(f"  all {n_users} users are test users exactly once: OK")
        return dict(n_users=n_users, each_user_test_once=once, folds=report)


def per_user_split(stimulus_ids, n_eval: int, rng: np.random.RandomState):
    # Split one user's images (single domain) into (support_pool, eval_fixed).
    # eval is the first n_eval images after shuffling, so it's the same set regardless of n_train
    
    stim = list(dict.fromkeys(str(s) for s in stimulus_ids))   # unique, order-preserving
    rng.shuffle(stim)
    return stim[n_eval:], stim[:n_eval]  # (support, eval)


def user_rng(split_seed: int, user_id: int) -> np.random.RandomState:
    # RandomState(split_seed + user_id), deterministic per user.
    return np.random.RandomState(split_seed + int(user_id))
