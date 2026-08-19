---
name: suite-planning
description: Plan, design, and implement modular experiment suites and standalone runners with flattened execution steps, coolname codenames, automated audio notifications, and consolidated packaging.
---

# Experiment Suite Planning & Management Workflow

This skill provides a comprehensive, battle-tested standard operating procedure (SOP) for designing, planning, reviewing, implementing, and running modular experiment suites in the `piaa-emotion-mediated` project.

---

## 1. Core Principles & Philosophy

1. **Flatten All Loops (No Opaque Shell Loops):**
   * Never execute nested bash `for` loops (e.g., `for BB in ...; for M in ...; do ...`).
   * Flatten every multidimensional parameter combination into discrete, sequentially indexed, individually addressable `SuiteStep`s.
   * Enables selective re-runs, independent failure isolation, and transparent progress tracking.

2. **Autonomous Coolname Slug Generation:**
   * Always generate clean, memorable 2-word slug codenames using `coolname` (e.g., `sassy_dragon`, `radiant_phoenix`, `emerald_tiger`).
   * Never use arbitrary or ad-hoc names unless explicitly requested by the user.

3. **Strict Plan-Before-Execution Review:**
   * Always present a formatted Markdown table outlining all steps, categories, backbones, mediators, parameters, and output folders for user review **before** creating files or executing code.

4. **Modular Architecture:**
   * Suite definitions live in `suites/<name>.py` exporting `SUITE = Suite(...)`.
   * Standalone generated runners live in `run_<name>.py`.
   * Central suite manager CLI in `suite.py`.
   * Central tracker updated in `EXPERIMENT_TRACKER.md`.

---

## 2. Planning Phase: Step Breakdown & Review Table

When a user requests a set of commands or a parameter sweep, structure the plan as follows:

### 2.1 Name Selection
Generate 3–5 curated `coolname` candidate slugs:
```python
import coolname
slug = coolname.generate_slug(2).replace("-", "_")
```

### 2.2 Sub-run Codename & Folder Convention
Use the standard prefix pattern: `<id>_<category>-<backbone>`
* Standard Baseline: `1_base-qwen8b`, `2_base-clip-ft`
* Joint Bottleneck: `5_joint-qwen8b`, `6_joint-clip-ft`
* Distributional Mediators: `9_dist-clip`, `10_dist-qwen8b`

### 2.3 User Review Table Template
Present the plan clearly:

```markdown
| # | Step ID | Sub-run Codename | Output Folder | Backbone | Mediators | Description |
|:---:|:---:|:---|:---|:---|:---|:---|
| **1** | `1` | `1_base-qwen8b` | `1_base-qwen8b/` | `qwen8b` | `(Default)` | Standard Stage-2 Sweep on Qwen 8B |
| **2** | `2` | `2_joint-qwen8b` | `2_joint-qwen8b/` | `qwen8b` | `emotion,emotion_mlp,emotion_joint` | Sequential vs Joint on Qwen 8B |
```

---

## 3. Implementation Phase: Suite Files Creation

### 3.1 Suite Definition (`suites/<name>.py`)
Create the declarative definition:

```python
"""Suite: '<name>'.

<Title and overview>.
"""
import sys
from src.utils.suite_engine import Suite, SuiteStep

SUITE = Suite(
    name="<name>",
    title="<Human Readable Title>",
    desc="<Detailed scientific description>.",
    steps=[
        SuiteStep(
            id=1,
            codename="1_<subrun_name>",
            folder="1_<subrun_name>",
            title="<Step Title>",
            desc="<Detailed step explanation>",
            cmd=[
                sys.executable, "main.py", "efficiency",
                "--backbone", "<backbone>",
                "--mediators", "<mediators>",
                "--n-train", "10,25,50,100",
                "--seed", "0,1,2",
                "--stage2", "C"
            ]
        ),
        # ... Additional flattened steps
    ]
)
```

### 3.2 Standalone Runner (`run_<name>.py`)
Create the one-line entrypoint:

```python
"""Dedicated Runner for Suite '<name>'."""
from suites.<name> import SUITE
from src.utils.suite_engine import run_suite_cli

if __name__ == "__main__":
    run_suite_cli(SUITE)
```

### 3.3 Register in Central Tracker (`EXPERIMENT_TRACKER.md`)
Append the suite to the Master Chronological Tracker Table:
```markdown
| **<ID>** | `<name>` | `PENDING (0/<N>)` | `output/<name>/` (`output/<name>/<name>_all_runs.zip`) | <Description> |
```

---

## 4. Execution, Audio Alerts & Packaging

### 4.1 CLI Execution Capabilities
Every runner provides out-of-the-box CLI commands:
```bash
# 1. View table and completion status
uv run run_<name>.py --list

# 2. Run all steps sequentially
uv run run_<name>.py --all

# 3. Flexible sub-run selection (by ID, prefix, or codename)
uv run run_<name>.py --run 1,3
uv run run_<name>.py --run 1_base
uv run run_<name>.py --run joint-qwen8b

# 4. Package all output files into a clean consolidated zip
uv run run_<name>.py --zip
```

### 4.2 Built-in Non-blocking Audio Alerts
The suite engine automatically triggers native macOS sound cues via `afplay`:
* 🔔 **`Glass.aiff`**: Plays immediately upon completion of each individual step.
* 🎉 **`Hero.aiff`**: Plays upon successful completion of the entire suite.
* ⚠️ **`Basso.aiff`**: Plays if an error occurs during execution.

### 4.3 Output Isolation & Sanitization
* Outputs from each step are automatically detected via filesystem snapshot diffs and copied to `output/<name>/<step_folder>/`.
* System files (`.DS_Store`, dotfiles) and zip archives are strictly filtered out of packaging and diff detection.
* Consolidated package is saved to `output/<name>/<name>_all_runs.zip`.

---

## 5. Post-Execution & Commit Checklist

Once a suite finishes running:
1. Package results: `uv run run_<name>.py --zip`.
2. Verify that `output/<name>/<name>_all_runs.zip` contains all summary and raw CSVs without `.DS_Store`.
3. Update `EXPERIMENT_TRACKER.md` status to `COMPLETED (<N>/<N>)`.
4. Stage and commit all outputs:
   ```bash
   git add output/<name> output/efficiency output/raw_all.csv EXPERIMENT_TRACKER.md
   git commit -m "feat(results): add <name> experiment outputs"
   ```
