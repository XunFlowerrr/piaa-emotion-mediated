"""Checks the data split is actually leak-free.

Calls V4Split.verify_disjoint (doesn't duplicate that logic) and then
checks, against the loaded ratings data itself, that train-user images and
test-user images never overlap in any fold/domain.

Usage:
    uv run main.py verify --splits
"""
from __future__ import annotations

from src.data.data import DOMAINS, XpassDataset
from src.data.splits import V4Split


def check_splits(cfg) -> bool:
    print("== Checking splits ==")
    sp = V4Split(cfg.split_dir, n_folds=cfg.n_folds)
    sp.verify_disjoint(verbose=True)

    ds = XpassDataset(cfg.data_dir, first_session_only=cfg.first_session_only,
                      verbose=False)
    ok = True
    for fold in sp.folds():
        for dom in DOMAINS:
            tr_img = set(ds.subset(domain=dom, users=fold.train_users)
                         ["stimulus_id"].astype(str))
            te_img = set(ds.subset(domain=dom, users=fold.test_users)
                         ["stimulus_id"].astype(str))
            overlap = tr_img & te_img
            if overlap:
                print(f"  fold{fold.index}/{dom}: {len(overlap)} overlapping images")
                ok = False
    print("  Train and test users image sets are disjoint: "
          + ("OK" if ok else "FAILED"))
    return ok
