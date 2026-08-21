"""Modal Serverless 16-Core CPU Concurrent Execution Engine for PIAA Experiment Suites.

Dispatches experiment steps in parallel across dedicated 16-core CPU containers on Modal Cloud,
enabling entire multi-seed suites to complete concurrently in ~3-4 minutes total.
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

    # 2. Persistent Features Volume
    features_vol = modal.Volume.from_name("piaa-features-vol", create_if_missing=True)

    # 3. Lightweight Container Image (Dependencies + Code)
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
        .add_local_dir(str(ROOT / "src"), remote_path="/root/project/src")
        .add_local_dir(str(ROOT / "Dataset"), remote_path="/root/project/Dataset")
        .add_local_file(str(ROOT / "main.py"), remote_path="/root/project/main.py")
    )

    # 4. Remote Serverless Function (16 Cores, 16 GB RAM per worker)
    @app.function(
        image=image,
        cpu=16.0,
        memory=16384,
        timeout=3600,
        volumes={"/root/project/features": features_vol},
    )
    def run_step_remote(step_dict: dict) -> dict:
        """Run an experiment step inside a dedicated 16-core Modal container."""
        import subprocess

        step_id = step_dict["id"]
        codename = step_dict["codename"]
        folder = step_dict["folder"]
        cmd_args = step_dict["cmd_args"]

        proj_dir = Path("/root/project")
        out_dir = proj_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        full_cmd = [sys.executable, "main.py"] + cmd_args
        print(f"[{codename} | 16-Core Container] Running: {' '.join(full_cmd)}")

        start_t = time.time()
        res = subprocess.run(
            full_cmd,
            cwd=proj_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        elapsed = time.time() - start_t
        print(f"[{codename}] Completed in {elapsed:.1f}s (Exit code: {res.returncode})")

        # Collect generated files
        collected_files: dict[str, bytes] = {}
        for p in out_dir.rglob("*"):
            if p.is_file() and (p.suffix in (".csv", ".json") or p.name == "raw_all.csv"):
                rel_path = str(p.relative_to(out_dir))
                collected_files[rel_path] = p.read_bytes()

        return {
            "id": step_id,
            "codename": codename,
            "folder": folder,
            "retcode": res.returncode,
            "stdout": res.stdout,
            "elapsed": elapsed,
            "files": collected_files,
        }


def run_suite_steps_modal(suite, steps_to_run):
    """Execute selected suite steps concurrently across 16-core Modal containers."""
    if not HAS_MODAL:
        print("[ERROR] 'modal' package is not installed. Please run: uv add modal")
        sys.exit(1)

    from src.utils.suite_engine import play_sound

    suite_dir = suite.output_dir
    suite_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    names = [s.codename for s in steps_to_run]
    print("=" * 80)
    print(f"🚀 CONCURRENT MODAL SERVERLESS EXECUTION (16 CORES / STEP): '{suite.name}' ({suite.title})")
    print(f"DISPATCHING {len(steps_to_run)} PARALLEL CONTAINER(S): {', '.join(names)}")
    print(f"OUTPUT DESTINATION: {suite_dir.relative_to(ROOT)}/")
    print("=" * 80)

    # Prepare payloads for each step
    payloads = []
    for step in steps_to_run:
        clean_args = []
        skip_next = False
        for arg in step.cmd:
            if skip_next:
                skip_next = False
                continue
            if arg.endswith("python") or arg.endswith("python3") or arg == "main.py":
                continue
            clean_args.append(arg)

        payloads.append({
            "id": step.id,
            "codename": step.codename,
            "folder": step.folder,
            "cmd_args": clean_args,
        })

    with modal.enable_output():
        with app.run():
            print(f"\n[+] Launching {len(payloads)} parallel Modal container(s) simultaneously...")
            # Use Modal .map() to run all steps concurrently in parallel!
            for res in run_step_remote.map(payloads):
                codename = res["codename"]
                folder = res["folder"]
                retcode = res["retcode"]
                elapsed = res["elapsed"]
                files = res["files"]

                target_dir = suite_dir / folder
                target_dir.mkdir(parents=True, exist_ok=True)

                if retcode != 0:
                    print(f"\n[ERROR] Step '{codename}' failed on Modal:\n{res['stdout']}")
                    play_sound("error")
                    sys.exit(retcode)

                print(f"\n[✔ DONE] Step '{codename}' finished in {elapsed:.1f}s! Synchronizing {len(files)} file(s)...")

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
                        print(f"  [+] Saved: {step_dest.relative_to(ROOT)}")

                print(f"[{suite.name}/{folder}] {saved_count} output file(s) synchronized locally.")
                play_sound("step_done")

    total_elapsed = time.time() - total_start
    play_sound("suite_done")
    print("\n" + "=" * 80)
    print(f"🎉 SUITE '{suite.name}' CONCURRENT MODAL RUN COMPLETE in {total_elapsed / 60:.2f} minutes!")
    print(f"Outputs stored in: {suite_dir.relative_to(ROOT)}/")
    print(f"To package all outputs into a zip archive, run: uv run run_{suite.name}.py --zip")
    print("=" * 80)
