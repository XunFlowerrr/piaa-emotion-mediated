"""Batch runner for Hayashi requested experiments.

Dynamically detects and tracks all generated files for each run,
copies them to output/hayashi_runs/<run_name>/, and packages all into a zip.
"""
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
HAYASHI_DIR = OUTPUT_DIR / "hayashi_runs"

RUNS = [
    {
        "name": "A1_headline_joint_vs_seq_C_n100",
        "desc": "A1. Joint vs Sequential (CLIP, n=100, seeds=0,1,2, stage2=C)",
        "cmd": [
            sys.executable, "main.py", "efficiency",
            "--backbone", "clip",
            "--mediators", "emotion,emotion_mlp,emotion_joint",
            "--n-train", "100",
            "--seed", "0,1,2",
            "--stage2", "C"
        ]
    },
    {
        "name": "A2_headline_joint_vs_seq_plain_n100",
        "desc": "A2. Joint vs Sequential (CLIP, n=100, seeds=0,1,2, stage2=plain)",
        "cmd": [
            sys.executable, "main.py", "efficiency",
            "--backbone", "clip",
            "--mediators", "emotion,emotion_mlp,emotion_joint",
            "--n-train", "100",
            "--seed", "0,1,2",
            "--stage2", "plain"
        ]
    },
    {
        "name": "B_fig2_support_sweep_C",
        "desc": "B. Fig 2 Support-size sweep (CLIP, n=10,25,50,100, seeds=0,1,2, stage2=C)",
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
        "name": "C_mlp_head_sweep_C",
        "desc": "C. MLP head sweep under anchor C (CLIP, heads=ridge,mlp, n=10,25,50,100, seeds=0,1,2, stage2=C)",
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
        "name": "D_distributional_stage1_C",
        "desc": "D. Distributional Stage-1 under anchor C (CLIP, mediators=emotion,emotion_sd,emotion_hist, n=10,25,50,100, seeds=0,1,2, stage2=C)",
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
        "name": "E_table1_mlp_C",
        "desc": "E. Table 1 MLP under anchor C (CLIP, heads=mlp, stage2=C, seeds=0,1,2)",
        "cmd": [
            sys.executable, "main.py", "table1",
            "--backbone", "clip",
            "--heads", "mlp",
            "--stage2", "C"
        ]
    }
]


def snapshot_output():
    """Return dict of relative_path -> mtime for all files in output/ (excluding hayashi_runs)."""
    snap = {}
    if not OUTPUT_DIR.exists():
        return snap
    for p in OUTPUT_DIR.rglob("*"):
        if p.is_file() and "hayashi_runs" not in str(p):
            try:
                snap[p] = p.stat().st_mtime
            except OSError:
                pass
    return snap


def main():
    print("=" * 70)
    print("STARTING HAYASHI REQUESTED EXPERIMENT RUNS (AUTOMATIC OUTPUT TRACKING)")
    print(f"Destination folder: {HAYASHI_DIR}")
    print("=" * 70)

    HAYASHI_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    for idx, run_info in enumerate(RUNS, 1):
        run_name = run_info["name"]
        run_desc = run_info["desc"]
        cmd = run_info["cmd"]
        target_dir = HAYASHI_DIR / run_name
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx}/{len(RUNS)}] >>> {run_desc}")
        print(f"Command: {' '.join(cmd)}")

        before_snap = snapshot_output()
        step_start = time.time()

        res = subprocess.run(cmd, cwd=ROOT)
        if res.returncode != 0:
            print(f"ERROR: Command failed with exit code {res.returncode}")
            sys.exit(res.returncode)

        step_elapsed = time.time() - step_start
        print(f"Finished in {step_elapsed:.1f}s. Detecting generated outputs...")

        after_snap = snapshot_output()
        changed_files = []
        for p, mtime in after_snap.items():
            if p.name == "raw_all.csv":
                continue  # raw_all is tracked globally
            if p not in before_snap or mtime > before_snap[p]:
                changed_files.append(p)

        copied_count = 0
        for src_file in changed_files:
            dest = target_dir / src_file.name
            shutil.copy2(src_file, dest)
            copied_count += 1
            print(f"  [+] Tracked & Copied: {src_file.name} -> {dest.relative_to(ROOT)}")

        print(f"[{run_name}] {copied_count} output file(s) saved to {target_dir.relative_to(ROOT)}")

    # Create consolidated zip archive
    zip_path = ROOT / "hayashi_runs_all.zip"
    print(f"\nCompressing all tracked results into {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in HAYASHI_DIR.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(OUTPUT_DIR)
                zf.write(file_path, arcname=str(arcname))
        raw_csv = OUTPUT_DIR / "raw_all.csv"
        if raw_csv.exists():
            zf.write(raw_csv, arcname="raw_all.csv")

    total_elapsed = time.time() - total_start
    print("=" * 70)
    print(f"ALL 6 EXPERIMENTS COMPLETE in {total_elapsed / 60:.2f} minutes!")
    print(f"Archived to: {zip_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
