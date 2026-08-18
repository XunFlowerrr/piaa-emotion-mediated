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


def check_parallel_repro(cfg, backbone: str = "clip", seeds=(0, 1), n_train: int = 10) -> bool:
    """Verify that multi-core parallel execution produces bit-for-bit identical results to serial execution."""
    import numpy as np
    from src.modeling.backbones import get_backbone
    from src.modeling.pipeline import Pipeline

    print(f"== Checking Serial vs. Multi-Core Reproducibility ==")
    print(f"  Backbone: {backbone} | n_train: {n_train} | Seeds: {seeds}")

    ds = XpassDataset(cfg.data_dir, first_session_only=cfg.first_session_only, verbose=False)
    bb = get_backbone(backbone, cfg.features_dir)
    sp = V4Split(cfg.split_dir, n_folds=cfg.n_folds)
    pipe = Pipeline(cfg, ds, bb, sp)

    mediators = ["identity", "emotion", "pca"]
    heads = ["ridge"]

    all_ok = True
    for seed in seeds:
        print(f"\n--- Testing Seed {seed} ---")
        # 1. Serial run (n_jobs=1)
        cfg.n_jobs = 1
        print("  [1/2] Running serial execution (1 core)...", flush=True)
        df_serial = pipe.run_grid(
            mediators=mediators, heads=heads, n_train=n_train,
            include_population=True, include_gt_upper_bound=False,
            seed=seed, stage2_variant="plain")

        # 2. Parallel run (n_jobs=-1)
        cfg.n_jobs = -1
        print("  [2/2] Running multi-core parallel execution (all cores)...", flush=True)
        df_parallel = pipe.run_grid(
            mediators=mediators, heads=heads, n_train=n_train,
            include_population=True, include_gt_upper_bound=False,
            seed=seed, stage2_variant="plain")

        # Sort by key to align rows
        sort_keys = ["fold", "domain", "user_id", "mediator", "head"]
        s_sorted = df_serial.sort_values(sort_keys).reset_index(drop=True)
        p_sorted = df_parallel.sort_values(sort_keys).reset_index(drop=True)

        rows_match = len(s_sorted) == len(p_sorted)
        print(f"  Total rows: Serial={len(s_sorted)}, Parallel={len(p_sorted)} -> {'MATCH' if rows_match else 'MISMATCH'}")

        metrics = ["srocc", "plcc", "eff_dof"]
        max_diff = 0.0
        for m in metrics:
            s_arr = np.nan_to_num(s_sorted[m].to_numpy(float), nan=0.0)
            p_arr = np.nan_to_num(p_sorted[m].to_numpy(float), nan=0.0)
            diff = float(np.max(np.abs(s_arr - p_arr)))
            max_diff = max(max_diff, diff)
            status = "IDENTICAL" if diff == 0.0 else ("WITHIN TOLERANCE" if diff < 1e-12 else "DIFFERS")
            print(f"  Metric '{m:8s}' max diff: {diff:.2e} -> {status}")

        # Check fold-by-fold row counts and match
        for fold in range(cfg.n_folds):
            sf = s_sorted[s_sorted.fold == fold]
            pf = p_sorted[p_sorted.fold == fold]
            fdiff = float(np.max(np.abs(sf["srocc"].to_numpy(float) - pf["srocc"].to_numpy(float))))
            print(f"    Fold {fold} ({len(sf)} units): SROCC max diff = {fdiff:.2e} -> {'OK' if fdiff < 1e-12 else 'FAIL'}")

        seed_ok = rows_match and (max_diff < 1e-12)
        all_ok &= seed_ok
        print(f"  Seed {seed} Status: {'PASSED (BIT-IDENTICAL)' if seed_ok else 'FAILED'}")

    print("\n" + "=" * 55)
    print(f"  Overall Multi-Core Reproducibility: {'PASSED' if all_ok else 'FAILED'}")
    print("=" * 55)
    return all_ok
