"""Modal Serverless Execution and Cloud Volume Engine for PIAA Experiment Suites.

Supports:
1. Interactive Real-time Execution (--modal): Live streaming logs with immediate local retrieval.
2. Detached Background Mode (--modal --detach): Fire-and-forget in cloud, safe against lost connections.
3. Cloud Volume Synchronization (--sync): Download completed outputs from Modal persistent storage anytime.
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

    # 2. Persistent Features and Outputs Volumes
    features_vol = modal.Volume.from_name("piaa-features-vol", create_if_missing=True)
    outputs_vol = modal.Volume.from_name("piaa-outputs-vol", create_if_missing=True)

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

    # 4. Remote Serverless Function (16 Cores, 16 GB RAM, 24h Timeout)
    @app.function(
        image=image,
        cpu=16.0,
        memory=16384,
        timeout=86400,
        volumes={
            "/root/project/features": features_vol,
            "/root/project/cloud_output": outputs_vol,
        },
    )
    def run_step_remote(step_dict: dict) -> dict:
        """Run an experiment step inside a dedicated 16-core Modal container."""
        import subprocess

        suite_name = step_dict.get("suite_name", "experiment")
        step_id = step_dict["id"]
        codename = step_dict["codename"]
        folder = step_dict["folder"]
        cmd_args = step_dict["cmd_args"]

        proj_dir = Path("/root/project")
        out_dir = proj_dir / "output"
        cloud_vol_dir = proj_dir / "cloud_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        cloud_vol_dir.mkdir(parents=True, exist_ok=True)

        full_cmd = [sys.executable, "-u", "main.py"] + cmd_args
        print(f"[{codename} | 16-Core Container] Running: {' '.join(full_cmd)}", flush=True)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        start_t = time.time()
        proc = subprocess.Popen(
            full_cmd,
            cwd=proj_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        stdout_lines = []
        if proc.stdout:
            for line in proc.stdout:
                print(f"[{codename}] {line}", end="", flush=True)
                stdout_lines.append(line)
        proc.wait()
        elapsed = time.time() - start_t
        print(f"[{codename}] Completed in {elapsed:.1f}s (Exit code: {proc.returncode})", flush=True)

        # Collect generated files and persist to cloud_output Volume
        collected_files: dict[str, bytes] = {}
        for p in out_dir.rglob("*"):
            if p.is_file() and (p.suffix in (".csv", ".json") or p.name == "raw_all.csv"):
                rel_path = str(p.relative_to(out_dir))
                data = p.read_bytes()
                collected_files[rel_path] = data

                # Save directly into persistent Cloud Volume for detached safety
                cloud_dest = cloud_vol_dir / suite_name / folder / p.name
                cloud_dest.parent.mkdir(parents=True, exist_ok=True)
                cloud_dest.write_bytes(data)

                # Also save to efficiency mirror in volume
                if p.name != "raw_all.csv":
                    eff_dest = cloud_vol_dir / "efficiency" / rel_path
                    eff_dest.parent.mkdir(parents=True, exist_ok=True)
                    eff_dest.write_bytes(data)

        # Commit volume so files are permanently stored in cloud
        try:
            outputs_vol.commit()
            print(f"[{codename}] Successfully committed {len(collected_files)} files to Cloud Volume.", flush=True)
        except Exception as e:
            print(f"[{codename}] Volume commit note: {e}", flush=True)

        return {
            "suite_name": suite_name,
            "id": step_id,
            "codename": codename,
            "folder": folder,
            "retcode": proc.returncode,
            "stdout": "".join(stdout_lines),
            "elapsed": elapsed,
            "files": collected_files,
        }


def sync_modal_outputs(suite):
    """Download and synchronize all finished outputs from Modal Cloud Volume to local disk."""
    if not HAS_MODAL:
        print("[ERROR] 'modal' package is not installed. Please run: uv add modal")
        sys.exit(1)

    from src.utils.suite_engine import play_sound

    suite_dir = suite.output_dir / "modal"
    suite_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"📥 SYNCHRONIZING CLOUD OUTPUTS FROM MODAL VOLUME: '{suite.name}'")
    print(f"Destination: {suite_dir.relative_to(ROOT)}/")
    print("=" * 80)

    try:
        remote_prefix = suite.name
        entries = list(outputs_vol.iterdir(remote_prefix, recursive=True))
    except Exception as e:
        print(f"[!] No remote directory for suite '{suite.name}' on Volume yet: {e}")
        return

    synced_count = 0
    for entry in entries:
        # Skip directories and hidden files
        if getattr(entry, "type", None) and "DIRECTORY" in str(entry.type):
            continue

        rel_str = entry.path[len(remote_prefix):].lstrip("/")
        if not rel_str or Path(rel_str).name.startswith("."):
            continue

        local_path = suite_dir / rel_str
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Read file from volume
        try:
            data = b"".join(outputs_vol.read_file(entry.path))
        except Exception as e:
            continue

        local_path.write_bytes(data)
        synced_count += 1
        print(f"  [+] Synced: {local_path.relative_to(ROOT)} ({len(data) / 1024:.1f} KB)")

        # Mirror to local efficiency/ folder
        p_name = Path(rel_str).name
        if p_name != "raw_all.csv" and p_name != "config.json":
            for bb in ["qwen8b", "qwen4b", "clip_ft_emo", "clip_ft", "clip"]:
                if bb in p_name:
                    eff_dest = OUTPUT_DIR / "efficiency" / bb / p_name
                    eff_dest.parent.mkdir(parents=True, exist_ok=True)
                    eff_dest.write_bytes(data)
                    break

    print("=" * 80)
    if synced_count > 0:
        play_sound("suite_done")
        print(f"🎉 SUCCESS: {synced_count} file(s) synchronized locally into {suite_dir.relative_to(ROOT)}/!")
    else:
        print(f"No output files found in Cloud Volume for '{suite.name}'.")
    print("=" * 80)


def run_suite_steps_modal(suite, steps_to_run, detach: bool = False):
    """Execute selected suite steps on Modal (Interactive or Detached mode)."""
    if not HAS_MODAL:
        print("[ERROR] 'modal' package is not installed. Please run: uv add modal")
        sys.exit(1)

    from src.utils.suite_engine import play_sound

    suite_dir = suite.output_dir / "modal"
    suite_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    names = [s.codename for s in steps_to_run]
    print("=" * 80)
    mode_text = "DETACHED (BACKGROUND CLOUD)" if detach else "INTERACTIVE (REAL-TIME STREAMING)"
    print(f"🚀 MODAL SERVERLESS EXECUTION [{mode_text}]: '{suite.name}' ({suite.title})")
    print(f"DISPATCHING {len(steps_to_run)} PARALLEL 16-CORE CONTAINER(S): {', '.join(names)}")
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
            "suite_name": suite.name,
            "id": step.id,
            "codename": step.codename,
            "folder": step.folder,
            "cmd_args": clean_args,
        })

    if detach:
        # Detached mode: Spawn and exit immediately!
        print(f"\n[+] Dispatching {len(payloads)} task(s) to Modal Cloud in the background...")
        with app.run():
            for p in payloads:
                handle = run_step_remote.spawn(p)
                print(f"  [🚀 SPAWNED] Step '{p['codename']}' -> Cloud Task Handle: {handle}")

        print("\n" + "=" * 80)
        print("🛡️  DETACHED EXECUTION ACTIVE:")
        print("• All jobs are now computing independently in Modal Cloud on 16-core CPU containers.")
        print("• You can safely close your laptop, turn off Wi-Fi, or disconnect at any time.")
        print("• Every completed fold is automatically saved to the persistent Modal Cloud Volume.")
        print(f"\n👉 When you are ready to download the results, run:")
        print(f"   uv run run_{suite.name}.py --sync")
        print("=" * 80)
        return

    # Interactive mode: Stream live and retrieve results
    with modal.enable_output():
        with app.run():
            print(f"\n[+] Launching {len(payloads)} parallel Modal container(s) simultaneously...")
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
                    local_out_file = OUTPUT_DIR / rel_path
                    local_out_file.parent.mkdir(parents=True, exist_ok=True)
                    local_out_file.write_bytes(data)

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
    print(f"🎉 SUITE '{suite.name}' MODAL RUN COMPLETE in {total_elapsed / 60:.2f} minutes!")
    print(f"Outputs stored in: {suite_dir.relative_to(ROOT)}/")
    print(f"To package all outputs into a zip archive, run: uv run run_{suite.name}.py --zip --modal")
    print("=" * 80)
