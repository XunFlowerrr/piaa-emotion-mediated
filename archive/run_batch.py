"""Batch Experiment Runner with Overarching Suite Codename, Sub-run Codenames, and Output Tracking.

Suite Codename (Default: 'hayashi'):
  Outputs are organized into: output/<batch_name>/<sub_codename>/
  Archive created at:         <batch_name>_all_runs.zip

Sub-run Codenames:
  • joint-c        (1): Joint vs Sequential under Anchor C
  • joint-plain    (2): Joint vs Sequential under Plain (Unanchored)
  • dist-c         (3): Distributional Stage-1 (mean, sd, hist) under Anchor C
  • dist-plain     (4): Distributional Stage-1 under Plain (Unanchored)
  • mlp-head-c     (5): Personal Ridge vs MLP Heads under Anchor C
  • table1-mlp-c   (6): Table 1 MLP Grid under Anchor C

Usage Examples:
  # 1. Run with default suite codename 'hayashi':
  uv run run_batch.py --run joint-c
  uv run run_batch.py --run joint-c,dist-c
  uv run run_batch.py --all

  # 2. Run under a custom suite codename:
  uv run run_batch.py --batch hayashi_v2 --all
  uv run run_batch.py --batch review_exp --run joint-c,dist-c

  # 3. List all runs and codenames:
  uv run run_batch.py --list
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"

STEPS = [
    {
        "id": 1,
        "codename": "joint-c",
        "title": "Joint vs Sequential Bottleneck (Anchor C)",
        "desc": "Tests sequential (emotion, emotion_mlp) vs joint (emotion_joint) Stage-1 bottleneck with Anchor C.",
        "cmd": [
            sys.executable, "main.py", "efficiency",
            "--backbone", "clip",
            "--mediators", "emotion,emotion_mlp,emotion_joint",
            "--n-train", "10,25,50,100",
            "--seed", "0,1,2",
            "--stage2", "C"
        ]
    },
    {
        "id": 2,
        "codename": "joint-plain",
        "title": "Joint vs Sequential Bottleneck (Plain / Unanchored)",
        "desc": "Baseline comparison for sequential vs joint Stage-1 without population anchoring.",
        "cmd": [
            sys.executable, "main.py", "efficiency",
            "--backbone", "clip",
            "--mediators", "emotion,emotion_mlp,emotion_joint",
            "--n-train", "10,25,50,100",
            "--seed", "0,1,2",
            "--stage2", "plain"
        ]
    },
    {
        "id": 3,
        "codename": "dist-c",
        "title": "Distributional Stage-1 (Anchor C)",
        "desc": "Tests distribution-valued Stage-1 (mean, sd, hist) under Anchor C.",
        "cmd": [
            sys.executable, "main.py", "efficiency",
            "--backbone", "clip",
            "--mediators", "emotion,emotion_sd,emotion_hist",
            "--n-train", "10,25,50,100",
            "--seed", "0,1,2",
            "--stage2", "C"
        ]
    },
    {
        "id": 4,
        "codename": "dist-plain",
        "title": "Distributional Stage-1 (Plain / Unanchored)",
        "desc": "Baseline comparison for distribution-valued Stage-1 without population anchoring.",
        "cmd": [
            sys.executable, "main.py", "efficiency",
            "--backbone", "clip",
            "--mediators", "emotion,emotion_sd,emotion_hist",
            "--n-train", "10,25,50,100",
            "--seed", "0,1,2",
            "--stage2", "plain"
        ]
    },
    {
        "id": 5,
        "codename": "mlp-head-c",
        "title": "Personal MLP & Ridge Heads (Anchor C)",
        "desc": "Efficiency sweep comparing Ridge vs MLP personal heads with residual anchoring (Variant C).",
        "cmd": [
            sys.executable, "main.py", "efficiency",
            "--backbone", "clip",
            "--heads", "ridge,mlp",
            "--n-train", "10,25,50,100",
            "--seed", "0,1,2",
            "--stage2", "C"
        ]
    },
    {
        "id": 6,
        "codename": "table1-mlp-c",
        "title": "Table 1 Full Grid for MLP (Anchor C)",
        "desc": "Recomputes Table 1 MLP rows across all 5 mediators under Anchor C (seeds 0,1,2).",
        "cmd": [
            sys.executable, "main.py", "table1",
            "--backbone", "clip",
            "--heads", "mlp",
            "--stage2", "C"
        ]
    }
]


def snapshot_output(batch_name: str) -> dict[Path, float]:
    """Capture mtimes of all files in output/ (excluding the current suite folder)."""
    snap = {}
    if not OUTPUT_DIR.exists():
        return snap
    for p in OUTPUT_DIR.rglob("*"):
        if p.is_file() and batch_name not in p.parts:
            try:
                snap[p] = p.stat().st_mtime
            except OSError:
                pass
    return snap


def list_steps():
    print("=" * 80)
    print("SUITE EXPERIMENT CODENAMES & COMMANDS")
    print("=" * 80)
    for s in STEPS:
        print(f"[{s['id']}] | Codename: {s['codename']:<14} | {s['title']}")
        print(f"    Description: {s['desc']}")
        print(f"    Command:     {' '.join(s['cmd'][2:])}\n")
    print("=" * 80)
    print("Usage Examples:")
    print("  # Run specific steps under default suite 'hayashi':")
    print("  uv run run_batch.py --run joint-c")
    print("  uv run run_batch.py --run joint-c,dist-c,mlp-head-c")
    print("  uv run run_batch.py --all")
    print("")
    print("  # Run under custom suite name:")
    print("  uv run run_batch.py --batch my_batch_name --all")
    print("  uv run run_batch.py --batch my_batch_name --run joint-c")
    print("=" * 80)


def resolve_selection(query: str) -> list[dict]:
    tokens = [t.strip().lower() for t in query.split(",") if t.strip()]
    selected = []
    for token in tokens:
        match = None
        for s in STEPS:
            if str(s["id"]) == token or s["codename"].lower() == token:
                match = s
                break
        if match:
            if match not in selected:
                selected.append(match)
        else:
            valid_keys = [f"{s['id']}/{s['codename']}" for s in STEPS]
            print(f"Error: Unknown run or codename '{token}'. Valid options are: {', '.join(valid_keys)}")
            sys.exit(1)
    return selected


def run_steps(steps_to_run: list[dict], batch_name: str = "hayashi"):
    suite_dir = OUTPUT_DIR / batch_name
    suite_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    names = [s["codename"] for s in steps_to_run]
    print("=" * 80)
    print(f"SUITE BATCH: '{batch_name}'")
    print(f"SELECTED RUNS ({len(steps_to_run)}): {', '.join(names)}")
    print(f"OUTPUT DESTINATION: {suite_dir.relative_to(ROOT)}/")
    print("=" * 80)

    for idx, step in enumerate(steps_to_run, 1):
        codename = step["codename"]
        title = step["title"]
        cmd = step["cmd"]
        target_dir = suite_dir / codename
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx}/{len(steps_to_run)}] >>> [{batch_name}/{codename}] {title}")
        print(f"Executing: {' '.join(cmd)}")

        before_snap = snapshot_output(batch_name)
        step_start = time.time()

        res = subprocess.run(cmd, cwd=ROOT)
        if res.returncode != 0:
            print(f"\n[ERROR] Step '{codename}' failed with exit code {res.returncode}")
            sys.exit(res.returncode)

        step_elapsed = time.time() - step_start
        print(f"\n[{codename}] Finished in {step_elapsed:.1f}s. Detecting generated outputs...")

        after_snap = snapshot_output(batch_name)
        changed_files = []
        for p, mtime in after_snap.items():
            if p.name == "raw_all.csv":
                continue  # database file tracked globally
            if p not in before_snap or mtime > before_snap[p]:
                changed_files.append(p)

        copied_count = 0
        for src_file in changed_files:
            dest = target_dir / src_file.name
            shutil.copy2(src_file, dest)
            copied_count += 1
            print(f"  [+] Output Saved: {dest.relative_to(ROOT)}")

        print(f"[{batch_name}/{codename}] {copied_count} output file(s) stored in {target_dir.relative_to(ROOT)}")

    # Update consolidated zip archive for this suite
    zip_path = ROOT / f"{batch_name}_all_runs.zip"
    print(f"\nPackaging all '{batch_name}' suite results into {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in suite_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(OUTPUT_DIR)
                zf.write(file_path, arcname=str(arcname))
        raw_csv = OUTPUT_DIR / "raw_all.csv"
        if raw_csv.exists():
            zf.write(raw_csv, arcname="raw_all.csv")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"SUITE '{batch_name}' COMPLETE in {total_elapsed / 60:.2f} minutes!")
    print(f"Consolidated Zip Archive: {zip_path.relative_to(ROOT)}")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(description="Batch Experiment Runner with Overarching Suite Name and Codenames")
    ap.add_argument("--batch", "-b", type=str, default="hayashi",
                    help="Overarching suite / batch codename (default: 'hayashi')")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all 6 experiments in this suite")
    group.add_argument("--run", "-r", type=str, help="Sub-run codename(s) or number(s) to run (e.g. 'joint-c,dist-c' or '1,3')")
    group.add_argument("--list", "-l", action="store_true", help="List all codenames, descriptions, and commands")
    args = ap.parse_args()

    if args.list:
        list_steps()
        return

    if args.all:
        selected = STEPS
    else:
        selected = resolve_selection(args.run)

    run_steps(selected, batch_name=args.batch)


if __name__ == "__main__":
    main()
