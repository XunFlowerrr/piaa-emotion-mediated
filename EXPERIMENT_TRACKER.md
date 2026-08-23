# 📊 Experiment Execution Tracker

This document provides a chronological record of all experiment suites, their sub-run codenames, CLI commands, status, and generated output artifacts.

---

## 🕒 Chronological Suite Timeline

```
1. [first] (clip_ft_emo 4-variant efficiency sweep)
   └── Finished initial baseline experiments on fine-tuned backbone
   
2. [hayashi] (Hayashi requested runs on CLIP after git pull)
   └── Joint vs Seq (n=100 & sweep), MLP under C, Dist Stage-1 under C, Table 1 MLP under C

3. [rebuttal] (Full Rebuttal & Reviewer Suite)
   └── Full systematic comparison across all conditions (Anchor C vs Plain baselines)
```

| # | Suite Codename | Status | Target Outputs & Archive | Description |
|:---:|:---|:---:|:---|:---|
| **1** | `first` | `COMPLETED (4/4)` | `output/first/` (`output/first/first_all_runs.zip`) | 4-variant Stage-2 efficiency sweep on `clip_ft_emo` ($n \in \{10,25,50,100\}$, seeds $0,1,2$) |
| **2** | `hayashi` | `COMPLETED (6/6)` | `output/hayashi/` (`output/hayashi/hayashi_all_runs.zip`) | Headline runs requested by Hayashi (A, B, C, D, E) on CLIP after git pull |
| **3** | `rebuttal` | `COMPLETED (6/6)` | `output/rebuttal/` (`output/rebuttal/rebuttal_all_runs.zip`) | Full systematic rebuttal suite (Anchor C vs Plain baselines across all modules) |
| **4** | `crimson_falcon` | `PENDING (0/1)` | `output/crimson_falcon/` (`output/crimson_falcon/crimson_falcon_all_runs.zip`) | Qwen2.5-VL-8B Joint vs Sequential Bottleneck sweep under Anchor C |
| **5** | `sassy_dragon` | `COMPLETED (5/5)` | `output/sassy_dragon/` (`output/sassy_dragon/sassy_dragon_all_runs.zip`) | Multimodal Distributional (1A–1C) & MLP Joint (2A–2B) Bottleneck Suite (Anchor C) |
| **6** | `radiant_phoenix` | `COMPLETED (2/2)` | `output/radiant_phoenix/` (`output/radiant_phoenix/radiant_phoenix_all_runs.zip`) | Qwen Vision-Language (8B & 4B) Joint vs Sequential Bottleneck Suite (Anchor C) |
| **7** | `emerald_tiger` | `COMPLETED (13/13)` | `output/emerald_tiger/` (`output/emerald_tiger/emerald_tiger_all_runs.zip`) | Comprehensive 13-Step Multimodal Backbone & Mediator Flattened Sweep (Anchor C) |
| **8** | `silver_falcon` | `COMPLETED (5/5)` | `output/silver_falcon/` (`output/silver_falcon/silver_falcon_all_runs.zip`) | Standard Stage-2 Controls Sweep (identity, pca, emotion, random, shuffled) across 5 Backbones (Anchor C) |
| **9** | `scarlet_swallow` | `COMPLETED (10/10)` | `output/scarlet_swallow/` (`output/scarlet_swallow/scarlet_swallow_all_runs.zip`) | Plain Sweeps (No-Anchor), Plain Distributions & Qwen8B Diagnostics Suite (10 Steps) |
| **10** | `demonic_bobcat` | `COMPLETED (3/3)` | `output/demonic_bobcat/` (`output/demonic_bobcat/demonic_bobcat_all_runs.zip`) | Qwen3-VL 8B MLP Series Rebuild Suite (Anchor C, n=100, MLP Head, 3 Seeds) |
| **11** | `logical_capybara` | `PENDING (0/3)` | `output/logical_capybara/` (`output/logical_capybara/logical_capybara_all_runs.zip`) | Table 1 Control Baselines Suite (identity, random, pca under MLP Head, Stage-2 C, 3 Seeds) |
| **12** | `thundering_lobster` | `PENDING (0/3)` | `output/thundering_lobster/` (`output/thundering_lobster/thundering_lobster_all_runs.zip`) | Qwen3-VL 8B MLP Series, pure MSE (no L2) and wide LR grid, all 7 mediators under MLP Head (Anchor C, n=100, 3 Seeds) |

---

## 🗄️ Where these suites live

Suites **1–11** have been retired to [`archive/suites/`](archive/suites/) —
each one's `SUITE` definition plus its generated `run_<codename>.py`, kept
together. Their results moved to `archive-output/<codename>/`. See
[`archive/suites/README.md`](archive/suites/README.md) to bring one back.

**Live in `suites/`: `thundering_lobster` (#12).** That is what
`uv run suite.py list` reports and what the commands below apply to.

### Why #12 supersedes the MLP rows of #10 and #11

Everything MLP in `demonic_bobcat` (#10) and `logical_capybara` (#11) was
produced with weight decay searched over `(1e-3, 1e-1, 1e1)` and a
learning-rate grid of three values. The per-candidate selection log added
afterwards showed the decay axis pinned at its maximum on **every** selection,
and the guide's five-value rate grid exhausted at **both** ends. A
hyperparameter chosen on the boundary of its grid reports where the search
stopped, not where the optimum is, so those MLP numbers are not reportable.
`thundering_lobster` re-runs them on pure MSE (`mlp_alpha = 0`) over eleven
rates from `1e-4` to `1e1`. The ridge rows of #10 and #11 are unaffected.

## 🛠️ Suite Management Infrastructure (`suite.py`)

Suites are managed modularly in `suites/` using a reusable engine (`src/utils/suite_engine.py`). Each suite has its own standalone generated runner:

```bash
# 1. List all registered suites and overall completion progress
uv run suite.py list

# 2. View details & status table for any suite
uv run suite.py show <codename>

# 3. Run experiments via Master CLI or Dedicated Runner
uv run run_<codename>.py --list
uv run run_<codename>.py --run <step>
uv run run_<codename>.py --zip

# 4. Generate a brand new suite with auto-generated codename (powered by coolname)
uv run suite.py new                      # generates random coolname (e.g. swift-falcon)
uv run suite.py new --name ablations     # generates custom suite and run_ablations.py
```

---

## 1. Suite: `first` (`clip-ft-emo-sweep`) — *Completed*

* **Codename**: `first`
* **Backbone**: `clip_ft_emo` (fine-tuned on emotion ratings)
* **Budget**: $n \in \{10, 25, 50, 100\}$, Seeds: $0, 1, 2$
* **Hardware**: Apple Silicon CPU (Multi-core parallel)

### Sub-Run Codenames & Commands:

| Sub-run Codename | Description | CLI Command | Output Artifact |
|:---|:---|:---|:---|
| **`ft-plain`** | Variant Plain (Ordinary ridge, shrinks to 0) | `uv run main.py efficiency --backbone clip_ft_emo --n-train 10,25,50,100 --seed 0,1,2 --stage2 plain` | `summary_clip_ft_emo.csv` |
| **`ft-anchor-a`** | Variant A (GIAA population score as extra feature) | `uv run main.py efficiency --backbone clip_ft_emo --n-train 10,25,50,100 --seed 0,1,2 --stage2 A` | `summary_A_clip_ft_emo.csv` |
| **`ft-anchor-b`** | Variant B (Shrink toward population weights) | `uv run main.py efficiency --backbone clip_ft_emo --n-train 10,25,50,100 --seed 0,1,2 --stage2 B` | `summary_B_clip_ft_emo.csv` |
| **`ft-anchor-c`** | Variant C (Residual fit against GIAA prediction) | `uv run main.py efficiency --backbone clip_ft_emo --n-train 10,25,50,100 --seed 0,1,2 --stage2 C` | `summary_C_clip_ft_emo.csv` |

* **Archive**: [`efficiency_clip_ft_emo_results.zip`](file:///Users/xunflowerrr/Main/Work/GithubRepository/piaa-emotion-mediated/efficiency_clip_ft_emo_results.zip) *(11 MB)*

---

## 2. Suite: `hayashi` — *Completed*

* **Codename**: `hayashi`
* **Context**: Run after the 4 `clip_ft_emo` commands (post git pull).
* **Backbone**: `clip`

### Sub-Run Codenames & Commands:

| Sub-run Codename | Item | Description | CLI Command | Target Folder |
|:---|:---:|:---|:---|:---|
| **`h-joint-n100-c`** | **A.1** | Headline Joint vs Seq ($n=100$, Anchor C) | `uv run main.py efficiency --backbone clip --mediators emotion,emotion_mlp,emotion_joint --n-train 100 --seed 0,1,2 --stage2 C` | `output/hayashi/h-joint-n100-c/` |
| **`h-joint-n100-plain`** | **A.2** | Headline Joint vs Seq ($n=100$, Plain) | `uv run main.py efficiency --backbone clip --mediators emotion,emotion_mlp,emotion_joint --n-train 100 --seed 0,1,2 --stage2 plain` | `output/hayashi/h-joint-n100-plain/` |
| **`h-joint-sweep-c`** | **B** | Support-size sweep Fig 2 ($n \in \{10,25,50,100\}$, Anchor C) | `uv run main.py efficiency --backbone clip --mediators emotion,emotion_mlp,emotion_joint --n-train 10,25,50,100 --seed 0,1,2 --stage2 C` | `output/hayashi/h-joint-sweep-c/` |
| **`h-mlp-sweep-c`** | **C** | MLP & Ridge head sweep under Anchor C | `uv run main.py efficiency --backbone clip --heads ridge,mlp --n-train 10,25,50,100 --seed 0,1,2 --stage2 C` | `output/hayashi/h-mlp-sweep-c/` |
| **`h-dist-sweep-c`** | **D** | Distributional Stage-1 ($n \in \{10,25,50,100\}$, Anchor C) | `uv run main.py efficiency --backbone clip --mediators emotion,emotion_sd,emotion_hist --n-train 10,25,50,100 --seed 0,1,2 --stage2 C` | `output/hayashi/h-dist-sweep-c/` |
| **`h-table1-mlp-c`** | **E** | Table 1 MLP rows under Anchor C | `uv run main.py table1 --backbone clip --heads mlp --stage2 C` | `output/hayashi/h-table1-mlp-c/` |

---

## 3. Suite: `rebuttal` — *Archived*

* **Codename**: `rebuttal`
* **Runner**: [`archive/suites/run_rebuttal.py`](archive/suites/run_rebuttal.py)
* **Output Destination**: `archive-output/rebuttal/<codename>/`
* **Archive**: `rebuttal_all_runs.zip`
* **Purpose**: Full systematic comparison between Anchor C and Plain unanchored baselines for reviewer evaluation.

### Sub-Run Codenames & Commands:

| Sub-run Codename | Description | CLI Command | Output Directory |
|:---|:---|:---|:---|
| **`joint-c`** | Joint vs Sequential Bottleneck (Anchor C) | `uv run main.py efficiency --backbone clip --mediators emotion,emotion_mlp,emotion_joint --n-train 10,25,50,100 --seed 0,1,2 --stage2 C` | `output/rebuttal/1_joint-c/` |
| **`joint-plain`** | Joint vs Sequential Baseline (Plain) | `uv run main.py efficiency --backbone clip --mediators emotion,emotion_mlp,emotion_joint --n-train 10,25,50,100 --seed 0,1,2 --stage2 plain` | `output/rebuttal/2_joint-plain/` |
| **`dist-c`** | Distributional Stage-1 (Anchor C) | `uv run main.py efficiency --backbone clip --mediators emotion,emotion_sd,emotion_hist --n-train 10,25,50,100 --seed 0,1,2 --stage2 C` | `output/rebuttal/3_dist-c/` |
| **`dist-plain`** | Distributional Baseline (Plain) | `uv run main.py efficiency --backbone clip --mediators emotion,emotion_sd,emotion_hist --n-train 10,25,50,100 --seed 0,1,2 --stage2 plain` | `output/rebuttal/4_dist-plain/` |
| **`mlp-head-c`** | Ridge vs MLP Personal Heads (Anchor C) | `uv run main.py efficiency --backbone clip --heads ridge,mlp --n-train 10,25,50,100 --seed 0,1,2 --stage2 C` | `output/rebuttal/5_mlp-head-c/` |
| **`table1-mlp-c`**| Table 1 MLP Grid under Anchor C | `uv run main.py table1 --backbone clip --heads mlp --stage2 C` | `output/rebuttal/6_table1-mlp-c/` |

