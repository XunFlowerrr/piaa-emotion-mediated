"""Modal Serverless CPU Execution Engine for PIAA Experiment Suites.

Enables massive multi-core cloud execution on Modal (e.g. 32-core CPU containers)
with automatic remote synchronization of features and local retrieval of output CSVs.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output"

try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False

if HAS_MODAL:
    # 1. Define Modal App
    app = modal.App(name="piaa-emotion-mediated")

    # 2. Define Container Image with all required scientific dependencies
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "numpy>=1.26",
            "pandas>=2.0",
            "scipy>=1.11",
            "scikit-learn>=1.4",
            "matplotlib>=3.8",
            "tabulate>=0.10.0",
            "rich>=15.0.0",
            "coolname>=5.0.0",
            "joblib>=1.3.0",
        )
    )

    # 3. Remote Serverless Function (32 Cores, 32 GB RAM)
    @app.function(
        image=image,
        cpu=32.0,
        memory=32768,
        timeout=3600,
        mounts=[
            modal.Mount.from_local_dir(ROOT / "src", remote_path="/root/project/src"),
            modal.Mount.from_local_dir(ROOT / "Dataset", remote_path="/root/project/Dataset"),
            modal.Mount.from_local_dir(ROOT / "features", remote_path="/root/project/features"),
            modal.Mount.from_local_file(ROOT / "main.py", remote_path="/root/project/main.py"),
        ],
    )
    def run_step_remote(cmd_args: list[str]) -> tuple[int, str, dict[str, bytes]]:
        """Run an experiment step inside Modal's 32-core container and return generated files."""
        import subprocess

        proj_dir = Path("/root/project")
        out_dir = proj_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        full_cmd = [sys.executable, "main.py"] + cmd_args
        print(f"[Modal Worker (32-Core CPU)] Running: {' '.join(full_cmd)}")

        start_t = time.time()
        res = subprocess.run(
            full_cmd,
            cwd=proj_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        elapsed = time.time() - start_t
        print(f"[Modal Worker] Completed in {elapsed:.1f}s with returncode {res.returncode}")

        # Collect all generated CSV and config.json files in output/
        collected_files: dict[str, bytes] = {}
        for p in out_dir.rglob("*"):
            if p.is_file() and (p.suffix in (".csv", ".json") or p.name == "raw_all.csv"):
                rel_path = str(p.relative_to(out_dir))
                collected_files[rel_path] = p.read_bytes()

        return res.returncode, res.stdout, collected_files


def run_suite_steps_modal(suite, steps_to_run):
    """Execute selected suite steps on Modal Serverless 32-core CPU containers."""
    if not HAS_MODAL:
        print("[ERROR] 'modal' package is not installed. Please run: uv add modal")
        sys.exit(1)

    from src.utils.suite_engine import play_sound

    suite_dir = suite.output_dir
    suite_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    names = [s.codename for s in steps_to_run]
    print("=" * 80)
    print(f"🚀 EXECUTING SUITE ON MODAL SERVERLESS (32-CORE CPU): '{suite.name}' ({suite.title})")
    print(f"SELECTED STEPS ({len(steps_to_run)}): {', '.join(names)}")
    print(f"OUTPUT DESTINATION: {suite_dir.relative_to(ROOT)}/")
    print("=" * 80)

    with app.run():
        for idx, step in enumerate(steps_to_run, 1):
            target_dir = suite_dir / step.folder
            target_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[{idx}/{len(steps_to_run)}] >>> [MODAL / {suite.name}/{step.folder}] {step.title}")
            # Filter out sys.executable / "main.py" to extract pure CLI arguments
            clean_args = []
            skip_next = False
            for arg in step.cmd:
                if skip_next:
                    skip_next = False
                    continue
                if arg.endswith("python") or arg.endswith("python3") or arg == "main.py":
                    continue
                clean_args.append(arg)

            print(f"Dispatched to Modal: python main.py {' '.join(clean_args)}")
            step_start = time.time()

            retcode, stdout_log, files = run_step_remote.remote(clean_args)
            if retcode != 0:
                print(stdout_log)
                play_sound("error")
                print(f"\n[ERROR] Modal Step '{step.codename}' failed with returncode {retcode}")
                sys.exit(retcode)

            step_elapsed = time.time() - step_start
            print(f"\n[{step.codename}] Modal run finished in {step_elapsed:.1f}s.")

            # Save retrieved files locally
            saved_count = 0
            for rel_path, data in files.items():
                p = Path(rel_path)
                # 1. Save in local output/efficiency/ if applicable
                local_out_file = OUTPUT_DIR / rel_path
                local_out_file.parent.mkdir(parents=True, exist_ok=True)
                local_out_file.write_bytes(data)

                # 2. Save in step target directory
                if p.name != "raw_all.csv":
                    step_dest = target_dir / p.name
                    step_dest.write_bytes(data)
                    saved_count += 1
                    print(f"  [+] Output Retrieved & Saved: {step_dest.relative_to(ROOT)}")

            print(f"[{suite.name}/{step.folder}] {saved_count} output file(s) synchronized locally.")
            play_sound("step_done")

    total_elapsed = time.time() - total_start
    play_sound("suite_done")
    print("\n" + "=" * 80)
    print(f"🎉 SUITE '{suite.name}' MODAL RUN COMPLETE in {total_elapsed / 60:.2f} minutes!")
    print(f"Outputs stored in: {suite_dir.relative_to(ROOT)}/")
    print(f"To package all outputs into a zip archive, run: uv run run_{suite.name}.py --zip")
    print("=" * 80)
