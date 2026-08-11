# Data and feature files

## Dataset

We use the XPASS-VIS dataset, which records beauty ratings and aesthetic
emotion ratings from 129 participants on 6,526 images across 3 domains,
87,836 interactions in total.

| domain | images | interactions |
|---|---|---|
| art | 2,345 | 31,543 |
| fashion | 2,082 | 28,154 |
| landscape | 2,099 | 28,139 |

Each interaction has an overall beauty rating (1-7 scale) and 9 aesthetic
emotion ratings (1-5 scale). This work uses 7 of the 9 as the mediator,
dropping `like` and `beautiful` because they overlap in meaning with the
dependent variable.

Raw data is not included in this repository and must be requested from the
dataset owner, then placed at `Dataset/maked/ratings.csv`.

### Rating scale conversion

The raw file stores the 9 emotion items as 0-4; the loader adds 1 to report
them on the 1-5 scale used in the rest of this document. The overall beauty
rating (`Aesthetic` in the raw file) is stored as 0-6 but is **already on
the 1-7 scale** in the sense the original dataset paper reports it -
verified by `verify_stats()` in `data.py`, which checks the per-domain mean
against the published reference values (art 3.23, fashion 3.32,
landscape 3.42) and passes without any shift. Shifting `Aesthetic` by +1
would move every mean by about 1.0 and fail that check. This asymmetry (one
column needs +1, the other doesn't) reflects how the two column groups were
originally encoded, not a bug.

## Data split

```
Dataset/split_v4_10group/fold{0..4}/
    train_users.txt         7 groups
    val_users.txt           1 group
    test_users.txt          2 groups
    giaa_train_images.txt   images used to fit the shared mediator
```

This split is included in the repository because it is small and required
to reproduce every result. See `docs/METHODOLOGY.md` section 2 for why it
is designed this way.

### Per-fold composition

All 129 users are partitioned into 10 groups; each fold uses 7 groups as
train, 1 as validation, and 2 as test, rotating so that every user is a
test user in exactly one fold (387 = 129 users x 3 domains total
evaluation units). Verified with `uv run main.py verify --splits`.

| fold | train users | val users | test users | GIAA train images | test images (art / fashion / landscape) |
|---|---|---|---|---|---|
| 0 | 97 | 6 | 26 | 4,561 | 470 / 420 / 420 |
| 1 | 95 | 12 | 22 | 4,564 | 470 / 420 / 419 |
| 2 | 92 | 14 | 23 | 4,563 | 470 / 418 / 420 |
| 3 | 83 | 16 | 30 | 4,561 | 470 / 420 / 420 |
| 4 | 84 | 17 | 28 | 4,582 | 465 / 404 / 420 |

Train, validation, and test users are disjoint within every fold, and the
images seen by train users never overlap with the images evaluated for test
users, in every fold and domain. `uv run main.py verify --splits` checks
both of these directly against the loaded data and prints a pass/fail per
fold; it does not just re-check the split files against each other.

## Feature files

Extracted in advance and stored as `.npz`, each with two keys:
`stimulus_ids` and `features`. Place them under `features/`.

| file | backbone | dim | notes |
|---|---|---|---|
| `clip_features.npz` | CLIP ViT-B/16, frozen | 512 | default for every experiment |
| `clip_ftpf_overall_v4_fold{0..4}.npz` | CLIP fine-tuned per fold | 768 | must be loaded per fold |
| `vlm4b_LT17.npz` | Qwen3-VL 4B, layer 17 | 2560 | |
| `vlm_LT15.npz` | Qwen3-VL 8B, layer 15 | 4096 | |

A fine-tuned backbone must always use the per-fold version. The per-fold
files are fine-tuned only on that fold's train users; a version fine-tuned
on all users leaks test-user information into the features. During
development, using the all-user version inflated Direct SROCC on this
backbone from 0.432 to 0.534 - see `docs/METHODOLOGY.md` section 6.

`clip_ftpf_emotion_v4_fold{0..4}.npz` (CLIP fine-tuned to predict emotions
rather than the overall score) is not used by any experiment currently
reported. It is kept because an earlier line of experiments (trait-based
mediation, not part of this paper) used it.

## Feature extraction

Not normally needed - the `.npz` files above are already provided. Needed
only to reproduce feature extraction from scratch or to extract a new
backbone.

```bash
uv sync --extra extract
```

### Step 1 - CLIP frozen features

```bash
python src/data/extract_clip_features.py \
    --images_dir Dataset/sample --ratings_csv Dataset/maked/ratings.csv \
    --out features/clip_features.npz
```
Reads every unique image referenced in `ratings.csv`, runs frozen CLIP
ViT-B/16, writes `stimulus_ids` + `features` (512-dim). Minutes on a GPU,
still reasonable on CPU. This is also the backbone used to build the v4
split in the first place (see `build_group_split.py`).

### Step 2 - CLIP fine-tuned per fold (leak-free)

```bash
for k in 0 1 2 3 4; do
  python src/data/finetune_clip_perfold.py --fold $k --target overall --v4 \
      --out features/clip_ftpf_overall_v4_fold$k.npz --epochs 8 --batch_size 12
done
```
Fine-tunes CLIP's vision tower to predict the population-mean overall
score, where the population is **that fold's train users only**, trained
on the fold's GIAA image set. Always pass `--v4` -- without it, the script
falls back to the old pre-v4 split, which is not what any reported number
uses. Run once per fold, five separate runs. Needs a GPU in practice (8
epochs over the GIAA image set per fold).

### Step 3 - Qwen3-VL raw features (all layers)

```bash
python src/data/extract_vlm_features.py \
    --images_dir Dataset/sample --ratings_csv Dataset/maked/ratings.csv \
    --model Qwen/Qwen3-VL-4B-Instruct --out features/vlm4b_features.npz
# repeat with --model Qwen/Qwen3-VL-8B-Instruct --out features/vlm_features.npz
```
Runs every image through the VLM once and keeps the hidden state at every
layer (`--save_layers all`), both the text-branch (LT) and vision-branch
(LV) token pooling. This is the expensive step: ~30h on one GPU for the 8B
model. Output is multi-GB per backbone and is **not committed** -
`.gitignore` excludes `features/vlm4b_features.npz` and `features/*_raw.npz`.

### Step 4 - pick one layer

```bash
python src/data/select_vlm_layer.py --vlm features/vlm4b_features.npz \
    --type LT --layer 17 --out features/vlm4b_LT17.npz
python src/data/select_vlm_layer.py --vlm features/vlm_features.npz \
    --type LT --layer 15 --out features/vlm_LT15.npz
```
Pulls one (layer, token type) slice out of the raw file, L2-normalizes it,
and writes it in the same `stimulus_ids`/`features` format as
`clip_features.npz`. This is what every experiment actually loads; the raw
multi-layer file is only an intermediate. Seconds to run.

Which layer to pick for a new backbone: run `layer_sweep_legacy.py` first
(next section) and read off the best-scoring layer, or use the same layer
already chosen for a similar-size model if consistency with prior results
matters more than the marginal gain of a fresh sweep.

### Why layer 17 (4B) and layer 15 (8B)

`src/data/layer_sweep_legacy.py` sweeps every (layer, token type) on Direct
CCC to pick a layer (CCC, not SROCC, is the criterion the script was
written around); results are in
`output/layer_sweep/qwen{4b,8b}_layer_sweep.csv`. This script uses the
project's original (pre-v4) split as a fast Direct-only screen, not the
leak-free v4 split used for reported results - it is a feature-selection
step, run once, upstream of the actual experiments.

**8B, layer 15**: this matches Ryu & Yanaka (arXiv:2604.11374), who report
LT15 (language-decoder layer 15, text tokens, average pooling) as their
config for Qwen3-VL-8B. Our own sweep supports it too, on the criterion the
script uses: Direct CCC is flat at ~0.376 from layer 15 through layer 35
(15: 0.3756, 35: 0.3769, a 0.001 spread), so layer 15 is within noise of the
sweep's best layer under CCC - it isn't the top layer if you instead sort by
SROCC (which would put layer 35 first, 0.4206 vs. 0.4126), but that's a
difference in which metric ranks the sweep, not evidence layer 15 was
picked without justification.

**4B, layer 17**: not carried over from a paper (Ryu & Yanaka don't specify
a 4B config), and deliberately swept from scratch rather than assuming the
8B choice transfers - the project notes are explicit that a paper's best
layer, validated on a different dataset (AADB), might not hold on
XPASS-Vis. Layer 17 came out on top for both CCC and SROCC on our sweep, no
ambiguity there.

Net: both choices are grounded, not arbitrary - 8B follows the published
config and isn't contradicted by our own CCC sweep; 4B is our own sweep's
top layer outright. The SROCC-sorted view naturally looks like it prefers
layer 35 for 8B, but the gap is inside the flat region of the curve, and
the current backbone comparison uses whatever the pipeline is configured
with (layer 15) without needing to be revisited. See
`output/layer_sweep/` for the full sweep.
