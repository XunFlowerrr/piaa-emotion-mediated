---
name: safe-upstream-merge
description: Safely fetch, inspect, and merge upstream updates into the working branch while strictly verifying zero mathematical regression, bit-for-bit reproducibility, and data split integrity.
---

# Safe Upstream Merge Workflow (Zero Mathematical Disturbance)

This skill provides a standardized, battle-tested procedure for pulling and merging changes from the `upstream` repository into the active working branch (`main` / feature branches) without introducing regression, mathematical distortions, or breaking local custom suite management infrastructure.

---

## 1. Phase 1: Fast Fetch & Pre-Merge Diff Inspection

### 1.1 Fast Fetch (Optimized for Large Git Blobs / Figures)
Always use buffer tuning flags to avoid transfer bottlenecks on large image/notebook commits:
```bash
git -c http.postBuffer=524288000 -c core.compression=0 fetch upstream main
```

### 1.2 Review Upstream Commits
```bash
git log HEAD..upstream/main --oneline --stat
```

### 1.3 Audit Core Mathematical & Modeling Diffs
Examine changes in core modeling, evaluation, splits, and mediator files:
```bash
git diff HEAD..upstream/main -- src/ main.py
```
Check if any changes alter:
- Stage 1 Ridge / MLP formulas or hyperparameter grids
- Stage 2 Residual Anchoring math (Variants A, B, C)
- Cross-validation splits (fold partitioning, image disjointness)
- Evaluation metrics (SROCC, PLCC, TOST equivalence)

---

## 2. Phase 2: Merge & Conflict Resolution

### 2.1 Perform Merge
```bash
git merge upstream/main --no-edit
```

### 2.2 Standard Conflict Resolution Rules
If merge conflicts arise, follow these non-destructive resolution guidelines:
- **`uv.lock`**:
  Do NOT resolve lockfile markers manually. Re-lock cleanly from `pyproject.toml`:
  ```bash
  git checkout --ours uv.lock
  uv lock
  ```
- **`output/**/config.json`**:
  Preserve multi-core execution settings (`"n_jobs": -1`) while adopting upstream's path/parameter additions.
- **`output/raw_all.csv`**:
  Preserve combined evaluation rows.
- **`pyproject.toml`**:
  Preserve all project dependencies (`coolname`, `rich`, `tabulate`, etc.).

### 2.3 Finalize Merge Commit
```bash
git add <resolved_files>
git commit --no-edit
```

---

## 3. Phase 3: Mathematical Verification Protocol (Zero Disturbance)

Run the two-stage verification suite to mathematically prove zero regression:

### 3.1 Step 1: Data Split & Leakage Verification
```bash
uv run main.py verify --splits
```
* **Required Pass Criteria**:
  - `all 129 users are test users exactly once: OK`
  - `Train and test users image sets are disjoint: OK`

### 3.2 Step 2: Bit-for-Bit Determinism & Parallel Reproducibility
```bash
uv run main.py verify --parallel --seed 0,1
```
* **Required Pass Criteria**:
  - Metric `srocc` max difference = `0.00e+00` (`IDENTICAL`)
  - Metric `plcc` max difference = `0.00e+00` (`IDENTICAL`)
  - Metric `eff_dof` max difference = `0.00e+00` (`IDENTICAL`)
  - All 5 folds report bit-identical results across serial and parallel workers.

### 3.3 Step 3: Experiment Suite Integrity Check
```bash
uv run suite.py list
```
* Confirm all suites (`first`, `hayashi`, `rebuttal`, `crimson_falcon`, etc.) parse cleanly with accurate completion badges.

---

## 4. Phase 4: Report Verification Summary

Provide a structured summary of:
1. Upstream commit range and functional description of upstream changes.
2. Resolution details for any conflicting files.
3. Explicit verification results (Split disjointness + Max numerical difference `0.00e+00`).
