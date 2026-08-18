"""Master Suite Management and Generation Infrastructure CLI.

Usage:
  # 1. List all available suites and their status:
  uv run suite.py list

  # 2. View details / steps of a specific suite:
  uv run suite.py show rebuttal
  uv run suite.py show hayashi

  # 3. Run experiments in a suite:
  uv run suite.py run rebuttal --all
  uv run suite.py run rebuttal --run joint-c
  uv run suite.py run rebuttal --run 1,3,5

  # 4. Create a new suite with auto-generated codename (via coolname):
  uv run suite.py new
  # or with a custom name:
  uv run suite.py new --name my_suite

  # 5. Package a suite's results into a zip:
  uv run suite.py zip rebuttal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.utils.suite_engine import render_suite_table, run_suite_steps, package_suite_zip, resolve_selection, check_step_status
from suites import get_all_suites, get_suite, SUITES_DIR


def list_all_suites():
    """List summary table of all registered suites."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    all_suites = get_all_suites()
    if not all_suites:
        print("No suites registered yet in suites/ directory.")
        return

    if use_rich:
        table = Table(
            title="[bold white on blue]  REGISTERED EXPERIMENT SUITES  [/]",
            box=box.ROUNDED,
            header_style="bold cyan",
            show_lines=True,
            expand=True
        )
        table.add_column("Suite Codename", style="bold magenta", width=18)
        table.add_column("Title & Goal", style="white", ratio=2)
        table.add_column("Steps Completed", justify="center", width=18)
        table.add_column("Dedicated Runner", style="cyan", width=20)
        table.add_column("Output Directory", style="dim", ratio=1)

        for name, suite in all_suites.items():
            completed = sum(1 for s in suite.steps if check_step_status(suite, s)[0])
            total = len(suite.steps)
            pct = (completed / total * 100) if total else 0
            color = "green" if completed == total and total > 0 else ("yellow" if completed > 0 else "dim white")
            status_text = f"[{color}]{completed}/{total} ({pct:.0f}%)[/]"

            runner_file = f"run_{name}.py"
            runner_exists = (ROOT / runner_file).exists()
            runner_str = f"[green]{runner_file}[/]" if runner_exists else f"[dim](via suite.py)[/]"

            table.add_row(
                name,
                f"[bold]{suite.title}[/]\n[dim]{suite.desc}[/]",
                status_text,
                runner_str,
                f"output/{name}/"
            )

        console.print()
        console.print(table)
        console.print("\n[dim]To view steps of a suite:[/] [cyan]uv run suite.py show <suite_name>[/]")
        console.print("[dim]To create a new suite:[/]    [cyan]uv run suite.py new[/]\n")
    else:
        print("=" * 80)
        print("REGISTERED EXPERIMENT SUITES")
        print("=" * 80)
        for name, suite in all_suites.items():
            completed = sum(1 for s in suite.steps if check_step_status(suite, s)[0])
            print(f"• Suite: {name:<14} | Progress: {completed}/{len(suite.steps)} | Title: {suite.title}")
            print(f"  Runner: run_{name}.py | Output: output/{name}/")
            print("-" * 80)


def create_new_suite(custom_name: str | None = None, title: str | None = None):
    """Generate a new suite definition and its standalone runner script."""
    if custom_name:
        suite_name = custom_name.strip().lower()
    else:
        try:
            import coolname
            suite_name = coolname.generate_slug(2)
        except ImportError:
            import uuid
            suite_name = f"suite-{uuid.uuid4().hex[:6]}"

    suite_file = SUITES_DIR / f"{suite_name}.py"
    runner_file = ROOT / f"run_{suite_name}.py"

    if suite_file.exists():
        print(f"Error: Suite '{suite_name}' already exists at {suite_file.relative_to(ROOT)}")
        sys.exit(1)

    display_title = title or f"Suite {suite_name.replace('-', ' ').title()}"

    # Template for suites/<name>.py
    suite_content = f'''"""Experiment Suite: '{suite_name}'."""
import sys
from src.utils.suite_engine import Suite, SuiteStep

SUITE = Suite(
    name="{suite_name}",
    title="{display_title}",
    desc="Custom generated experiment suite.",
    steps=[
        SuiteStep(
            id=1,
            codename="exp1",
            title="Experiment 1",
            desc="Description for experiment 1",
            cmd=[sys.executable, "main.py", "efficiency", "--backbone", "clip", "--n-train", "10,25,50,100", "--seed", "0,1,2", "--stage2", "C"]
        ),
    ]
)
'''
    suite_file.write_text(suite_content, encoding="utf-8")
    print(f"[✓] Created Suite Definition: {suite_file.relative_to(ROOT)}")

    # Template for run_<name>.py
    runner_content = f'''"""Dedicated Runner for Suite '{suite_name}'."""
from suites.{suite_name} import SUITE
from src.utils.suite_engine import run_suite_cli

if __name__ == "__main__":
    run_suite_cli(SUITE)
'''
    runner_file.write_text(runner_content, encoding="utf-8")
    print(f"[✓] Created Standalone Runner: {runner_file.name}")

    print("\n" + "=" * 70)
    print(f"🎉 NEW SUITE CREATED: '{suite_name}'")
    print("=" * 70)
    print(f"1. Edit steps in:    suites/{suite_name}.py")
    print(f"2. View status:      uv run {runner_file.name} --list")
    print(f"3. Run experiments:  uv run {runner_file.name} --all")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="Master Experiment Suite Manager and Generator")
    sub = ap.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all registered experiment suites")

    # show
    p_show = sub.add_parser("show", help="Show detailed step status table for a specific suite")
    p_show.add_argument("suite_name", type=str, help="Name of the suite (e.g. rebuttal, hayashi, first)")

    # run
    p_run = sub.add_parser("run", help="Run experiments in a suite")
    p_run.add_argument("suite_name", type=str, help="Name of the suite to run")
    g_run = p_run.add_mutually_exclusive_group(required=True)
    g_run.add_argument("--all", action="store_true", help="Run all steps in this suite")
    g_run.add_argument("--run", "-r", type=str, help="Sub-run codename(s) or number(s) to run (e.g. '1,3' or 'joint-c')")

    # zip
    p_zip = sub.add_parser("zip", help="Package a suite's output files into a zip archive")
    p_zip.add_argument("suite_name", type=str, help="Name of the suite to zip")

    # new
    p_new = sub.add_parser("new", help="Create a new experiment suite and generate its runner")
    p_new.add_argument("--name", "-n", type=str, default=None, help="Custom suite codename (defaults to coolname generator)")
    p_new.add_argument("--title", "-t", type=str, default=None, help="Human-readable title for the suite")

    args = ap.parse_args()

    if args.command == "list":
        list_all_suites()
    elif args.command == "show":
        suite = get_suite(args.suite_name)
        if not suite:
            print(f"Error: Suite '{args.suite_name}' not found. Run 'uv run suite.py list' to see available suites.")
            sys.exit(1)
        render_suite_table(suite)
    elif args.command == "run":
        suite = get_suite(args.suite_name)
        if not suite:
            print(f"Error: Suite '{args.suite_name}' not found.")
            sys.exit(1)
        if args.all:
            selected = suite.steps
        else:
            selected = resolve_selection(suite, args.run)
        run_suite_steps(suite, selected)
    elif args.command == "zip":
        suite = get_suite(args.suite_name)
        if not suite:
            print(f"Error: Suite '{args.suite_name}' not found.")
            sys.exit(1)
        package_suite_zip(suite)
    elif args.command == "new":
        create_new_suite(custom_name=args.name, title=args.title)


if __name__ == "__main__":
    main()
