"""Two checks you can run against this code.

1. `--splits`  the data split is actually leak-free. Calls
   V4Split.verify_disjoint (doesn't duplicate that logic) and then checks,
   against the loaded ratings data itself, that train-user images and
   test-user images never overlap in any fold/domain.

2. `--repro EXPERIMENT`  running the same experiment twice gives byte-identical
   results. Worth running because the failure it catches is silent: nothing
   crashes, the numbers just move. See ALPHA_TIE_RTOL in modeling/heads.py and
   the BLAS thread pinning at the top of main.py for what makes this pass.

Usage:
    uv run main.py verify --splits
    uv run main.py verify --repro efficiency
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from src.data.data import DOMAINS, XpassDataset
from src.data.splits import V4Split


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def check_repro(cfg, experiment: str, run_experiment) -> bool:
    """Run `experiment` twice and compare every CSV it wrote, byte for byte."""
    print(f"== Checking reproducibility of '{experiment}' (running it twice) ==")
    out = Path(cfg.output_dir) / experiment

    run_experiment()
    first = {p.name: _md5(p) for p in sorted(out.glob("*.csv"))}
    keep = out.parent / f"_repro_{experiment}"
    shutil.rmtree(keep, ignore_errors=True)
    shutil.copytree(out, keep)

    run_experiment()
    second = {p.name: _md5(p) for p in sorted(out.glob("*.csv"))}
    shutil.rmtree(keep, ignore_errors=True)

    ok = True
    for name in sorted(set(first) | set(second)):
        same = first.get(name) == second.get(name)
        ok &= same
        print(f"  {name:28s} {'IDENTICAL' if same else 'DIFFERS'}  {first.get(name)}")
    print("  reproducible: " + ("YES" if ok else "NO - results are not stable"))
    return ok


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
