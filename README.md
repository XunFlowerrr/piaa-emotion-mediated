# Emotion-Mediated Personalized Image Aesthetic Assessment

Code for all experiments in the paper. We predict a personal beauty score by
routing prediction through a 7-dimensional layer of aesthetic emotions,
instead of predicting from image features directly.

The core idea is a mediator that can be read: a model shared across all
users predicts how an image tends to make people feel (7 emotions), and
then a per-user linear formula with only 7 parameters turns those emotions
into that user's score - compared to a direct approach that needs 512
parameters per user and is not interpretable.


## Code layout

```
main.py                 single entry point (argument parser)
src/
  config.py              every setting that affects the numbers lives here, once
  data/
    data.py               loads ratings + the first-session filter
    splits.py              leak-free (v4) split protocol
    build_group_split.py   builds the v4 10-group split from raw user metadata
    extract_clip_features.py   extract frozen CLIP ViT-B/16 features
    extract_vlm_features.py    extract raw Qwen3-VL features (all layers)
    select_vlm_layer.py        pick one layer out of the raw VLM output
    finetune_clip_perfold.py   fine-tune CLIP per fold (leak-free)
    layer_sweep_legacy.py      sweep layers to pick the VLM layer (uses the
                                pre-v4 split; a fast upstream screening step,
                                not part of the reported pipeline - see
                                docs/DATA.md "Why layer 17 / layer 15")
  modeling/
    backbones.py           CLIP / per-fold CLIP-ft / Qwen3-VL
    mediators.py            identity, emotion, pca, random, shuffled
    heads.py                RidgeHead, MLPHead
    pipeline.py             wires the three together and runs the evaluation loop
  utils/
    metrics.py              SROCC, PLCC, CCC (unused, kept in the CSVs only), Wilcoxon, effective DoF
    tables.py               per-unit results -> summary CSV (mean/sd/best/sig)
    plots.py                figures
    verify.py               checks the data split is leak-free
  experiments/              one file per reported table
```

Every experiment is a **mediator x head** grid. Adding a new mediator or
head is one new class in `mediators.py` or `heads.py`; nothing else in
`pipeline.py` needs to change.


## Documentation

- `docs/METHODOLOGY.md` - experimental protocol, metrics, and the reasoning
  behind each design decision
- `docs/DATA.md` - dataset, split, and feature files


## Getting started

```bash
uv sync
uv run main.py verify --splits      # check the data split is leak-free
uv run main.py table1               # the main table
```

Data must be placed as follows first (see `docs/DATA.md`):

```
Dataset/maked/ratings.csv           rating data (request from the dataset owner)
Dataset/split_v4_10group/fold{0..4} user split (already in this repo)
features/clip_features.npz          pre-extracted features
```


## Commands

| command | produces |
|---|---|
| `uv run main.py table1` | main table: every mediator x head at 100 ratings/user |
| `uv run main.py backbone` | compares 4 backbones (Direct vs. Hybrid) |
| `uv run main.py efficiency` | results by rating budget (10/25/50/100) |
| `uv run main.py faithfulness` | formula swap + weight-vs-empirical-correlation |
| `uv run main.py mlp_diagnostics` | proves the MLP head actually converges (loss curves) |
| `uv run main.py verify --splits` | checks the data split for leakage |

Every command writes to `output/<experiment name>/`, with `config.json`
recording every setting that could affect the numbers, alongside
`per_unit.csv` (raw, one row per evaluation unit) and `summary.csv`
(mean/sd/best-flag/significance, ready to paste into a spreadsheet).
`output/` in this repo already has real runs of `efficiency`,
`faithfulness`, and `mlp_diagnostics` committed, so those can be inspected
without rerunning anything; `table1` and `backbone` are not pre-run here
(table1 takes several hours because of the MLP grid) - run them yourself
to reproduce.


## Verifying results match the paper

```bash
uv run main.py verify --splits
```

Checks that the data split is actually leak-free, at both the user level
and the image level - see `docs/DATA.md` for the exact per-fold numbers
this should print (0 overlapping images in every fold/domain).

### Reference values for a quick sanity check

If a run gives numbers far from these, something is wrong.

| check | expected value |
|---|---|
| rows after the first-session filter | 83,327 of 87,836 |
| evaluation units | 387 (129 users x 3 domains) |
| each user is a test user | exactly once |
| Hybrid ridge, average | SROCC 0.421, PLCC 0.431 |
| Direct ridge, average | SROCC 0.391, PLCC 0.401 |
| own formula vs. another user's formula | 0.421 vs. 0.349 (SROCC) |
| weight vs. empirical correlation | Spearman 0.492, 91% of users positive |


## Things to watch out for

**The first-session filter** (`src/data/data.py:100-101`,
`XpassDataset.__init__`, controlled by `Config.first_session_only`) changes
every number in the paper with no visible warning if set wrong. `load_raw()`
loads all 87,836 rows; `XpassDataset` then keeps only the first occurrence
of each `(user_id, stimulus_id)` pair, by file row order (`sort_values by
_row`, `keep="first"`). It prints the row count on every load - check that
it always says 83,327.

**A fine-tuned backbone must be loaded per fold** (`src/modeling/
backbones.py:55-73`, class `PerFoldBackbone`, used by `BACKBONE_SPECS["clip_
ft"]` at line 78). Using a single file fine-tuned on all users instead of
one file per fold leaks test-user information into the features and
inflated Direct SROCC from 0.432 to 0.534 during development - see
`docs/DATA.md`.

**The mediator's random-number draw order is fixed**
(`src/modeling/mediators.py:109-111`, function `build_shared_mediators`).
One `np.random.default_rng(fold_index)` per fold draws the random
projection matrix `R` first, then the shuffle permutation, in that order.
Reordering these two draws changes every downstream number even though the
method is unchanged, because it changes which random numbers `R` and the
permutation get.

**Comparing two models always requires a paired test**
(`src/utils/metrics.py:48`, `wilcoxon_paired`). Every model in this
codebase is evaluated on the same users and images, so an unpaired test
would understate significance. Every experiment file uses
`wilcoxon_paired`, keyed on `(fold, domain, user_id)`, not a plain
two-sample test.

**The MLP head needs early stopping, not weight decay, to train fairly**
(`src/modeling/heads.py:81` comment, `MLPHead._make` around line 92-104,
`Config.mlp_early_stopping`/`mlp_alpha` in `src/config.py`). With
`mlp_alpha=0` and 128 hidden units on 100 samples, training to convergence
without early stopping memorizes the data; capping `max_iter` low instead
means it never converges. `early_stopping=True` is a stopping rule, not a
regularizer, so it keeps the "no weight decay" requirement while still
training the MLP properly - see `output/mlp_diagnostics/` for evidence this
is real, not an artifact of under-training.

## Adding a new experiment

New mediator: add a class that subclasses `Mediator` in `mediators.py`,
then register it in `build_shared_mediators`.

New head: add a class that subclasses `Head` in `heads.py`, then add it to
`make_head`.

New backbone: add an entry to `BACKBONE_SPECS` in `backbones.py`.

New table: create a file in `experiments/`, then add its name to `main.py`.

None of this requires touching `pipeline.py`.