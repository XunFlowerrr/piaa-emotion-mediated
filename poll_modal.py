"""Standalone Modal Cloud Poller and macOS Native Notification Watcher.

Usage:
  # 1. Watch a specific suite, poll until completed, trigger silent macOS Notification banner, and auto-sync:
  uv run poll_modal.py logical_capybara

  # 2. Watch with custom poll interval (in seconds):
  uv run poll_modal.py logical_capybara --interval 10

  # 3. Watch without auto-syncing:
  uv run poll_modal.py logical_capybara --no-sync
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False

from suites import get_suite


def send_macos_banner(title: str, subtitle: str, message: str):
    """Trigger a silent native macOS Notification Center banner (no sound)."""
    clean_title = title.replace('"', '\\"')
    clean_sub = subtitle.replace('"', '\\"')
    clean_msg = message.replace('"', '\\"')
    script = f'display notification "{clean_msg}" with title "{clean_title}" subtitle "{clean_sub}"'
    try:
        subprocess.run(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def send_persistent_macos_alert(title: str, message: str, open_folder: Path | None = None):
    """Trigger a persistent native macOS Alert Dialog popup that STAYS ON SCREEN until clicked."""
    clean_title = title.replace('"', '\\"')
    clean_msg = message.replace('"', '\\"')

    if open_folder and open_folder.exists():
        script = f'''
        set btn to button returned of (display alert "{clean_title}" message "{clean_msg}" as informational buttons {{"Open Folder", "OK"}} default button "OK")
        if btn is "Open Folder" then
            tell application "Finder" to reveal POSIX file "{str(open_folder.resolve())}"
            tell application "Finder" to activate
        end if
        '''
    else:
        script = f'display alert "{clean_title}" message "{clean_msg}" as informational buttons {{"OK"}} default button "OK"'

    try:
        subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def poll_suite(suite_name: str, interval: int = 15, auto_sync: bool = True, notify_mode: str = "both"):
    """Poll Modal Volume and Function status until suite completion, then send selected macOS Notification."""
    if not HAS_MODAL:
        print("[ERROR] 'modal' package is not installed.")
        sys.exit(1)

    suite = get_suite(suite_name)
    if not suite:
        print(f"[ERROR] Suite '{suite_name}' not found. Please check the suite name.")
        sys.exit(1)

    try:
        from rich.console import Console
        from rich.live import Live
        from rich.table import Table
        from rich import box
        console = Console()
        use_rich = True
    except ImportError:
        console = None
        use_rich = False

    outputs_vol = modal.Volume.from_name("piaa-outputs-vol", create_if_missing=True)
    expected_folders = [s.folder for s in suite.steps]
    total_steps = len(suite.steps)

    print("=" * 80)
    print(f"📡 MODAL CLOUD WATCHER: Suite '{suite.name}' ({suite.title})")
    print(f"Tracking {total_steps} step(s): {', '.join(expected_folders)}")
    print(f"Poll Interval: {interval}s | Notification Mode: {notify_mode.upper()}")
    print(f"Auto-Sync: {'ENABLED' if auto_sync else 'DISABLED'}")
    print("=" * 80)

    start_time = time.time()
    completed_steps = set()

    while True:
        # Check files on Modal Volume
        try:
            entries = list(outputs_vol.iterdir(suite.name, recursive=True))
            found_paths = [e.path for e in entries if getattr(e, "type", None) and int(e.type) == 1]
        except Exception:
            found_paths = []

        for s in suite.steps:
            step_has_files = any(s.folder in p and (p.endswith(".csv") or p.endswith(".json")) for p in found_paths)
            if step_has_files and s.folder not in completed_steps:
                completed_steps.add(s.folder)
                elapsed_cur = time.time() - start_time
                print(f"  [✔ STEP DONE] '{s.codename}' completed on cloud in {elapsed_cur / 60:.1f}m!")
                if notify_mode in ("banner", "both"):
                    send_macos_banner(
                        title="PIAA Experiment Progress",
                        subtitle=f"Step Completed: {s.codename}",
                        message=f"Step {len(completed_steps)}/{total_steps} finished on Modal Cloud."
                    )

        done_count = len(completed_steps)
        elapsed = time.time() - start_time
        time_str = f"{int(elapsed // 60)}m {int(elapsed % 60):02d}s"

        if done_count >= total_steps:
            break

        status_line = f"⏳ [Polling Modal Cloud] Completed: {done_count}/{total_steps} steps | Elapsed: {time_str} | Next check in {interval}s..."
        if use_rich:
            console.print(f"[cyan]{status_line}[/]", end="\r")
        else:
            print(status_line, end="\r", flush=True)

        time.sleep(interval)

    total_elapsed = time.time() - start_time
    print("\n\n" + "=" * 80)
    print(f"🎉 ALL {total_steps} STEPS COMPLETED ON MODAL CLOUD in {total_elapsed / 60:.2f} minutes!")
    print("=" * 80)

    # 1. Trigger Banner if requested
    if notify_mode in ("banner", "both"):
        send_macos_banner(
            title="PIAA Experiment Finished! 🎉",
            subtitle=f"Suite '{suite.name}' 100% Completed",
            message=f"All {total_steps} steps finished on Modal Cloud ({total_elapsed / 60:.1f}m)."
        )

    # 2. Trigger Persistent Alert Dialog if requested (Stay-On-Screen)
    if notify_mode in ("dialog", "both"):
        send_persistent_macos_alert(
            title=f"Suite '{suite.name}' 100% Finished! 🎉",
            message=f"All {total_steps} steps finished on Modal Cloud in {total_elapsed / 60:.1f} minutes.\nFiles are saved in output/{suite.name}/modal/.",
            open_folder=suite.output_dir / "modal"
        )

    # 3. Auto Sync if enabled
    if auto_sync:
        from src.utils.modal_engine import sync_modal_outputs
        print("\n[+] Automatically synchronizing files from Modal Volume...")
        sync_modal_outputs(suite)

    print("\n👉 To package outputs into zip archive, run:")
    print(f"   uv run run_{suite.name}.py --zip --modal")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Standalone Modal Cloud Poller and macOS Notification Watcher")
    parser.add_argument("suite_name", type=str, help="Name of the suite to watch (e.g. logical_capybara, demonic_bobcat)")
    parser.add_argument("--interval", "-i", type=int, default=15, help="Polling interval in seconds (default: 15)")
    parser.add_argument("--no-sync", action="store_true", help="Disable automatic downloading upon completion")
    parser.add_argument(
        "--notify",
        "-n",
        choices=["both", "dialog", "banner", "none"],
        default="both",
        help="Notification mode: 'dialog' (Stay-On-Screen popup), 'banner' (sliding banner), 'both' (popup + banner), or 'none' (default: both)"
    )
    args = parser.parse_args()

    poll_suite(args.suite_name, interval=args.interval, auto_sync=not args.no_sync, notify_mode=args.notify)


if __name__ == "__main__":
    main()
