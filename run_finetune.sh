#!/usr/bin/env bash
# Fine-tune CLIP per fold, on any machine with uv installed.
#
#   ./run_finetune.sh emotion      # predict the 7 emotions  -> clip_ft_emo
#   ./run_finetune.sh overall      # predict the score       -> clip_ft
#   EPOCHS=4 ./run_finetune.sh emotion
#
# Picks CUDA, then Apple-silicon MPS, then CPU. Folds already finished are
# skipped, so re-running after an interruption resumes instead of redoing.
set -euo pipefail

TARGET="${1:-emotion}"
EPOCHS="${EPOCHS:-8}"
BATCH="${BATCH:-32}"
LR="${LR:-1e-5}"
SEED="${SEED:-42}"
FOLDS="${FOLDS:-0 1 2 3 4}"
IMAGES_DIR="${IMAGES_DIR:-Dataset/sample}"

cd "$(dirname "$0")"
OUTDIR="features/clip_ftpf_${TARGET}_v4_results"
mkdir -p "$OUTDIR"

echo "================================================================"
echo "  target : $TARGET"
echo "  folds  : $FOLDS"
echo "  epochs : $EPOCHS   batch: $BATCH   lr: $LR"
echo "  images : $IMAGES_DIR"
echo "================================================================"

# --- preflight: the checks that are cheap now and expensive later ------
uv run python - "$IMAGES_DIR" <<'PY'
import sys, os
import pandas as pd

images_dir = sys.argv[1]
df = pd.read_csv("Dataset/maked/ratings.csv")

broken = int((df["sample_file"].astype(str) == "#NAME?").sum())
print(f"'#NAME?' rows : {broken}")
if broken:
    raise SystemExit(
        "STOP: ratings.csv is damaged. 36 scenery filenames read as the\n"
        "spreadsheet error '#NAME?', and every image resolved through that\n"
        "column gets silently skipped. Repair it first:\n"
        "  uv run python -m src.data.repair_ratings_csv --source <XPASS-VIS>/maked/ratings.csv")

def resolve(sf):
    sf = str(sf)
    return sf[:-4] + ".jpg" if sf.lower().endswith(".mp4") else sf

have = {f for _, _, fs in os.walk(images_dir) for f in fs}
want = {resolve(s) for s in df["sample_file"].unique()}
missing = want - have
print(f"stimuli       : {len(want)}")
print(f"resolvable    : {len(want & have)}")
print(f"missing       : {len(missing)}")
if missing:
    print("  e.g.", sorted(missing)[:5])
    raise SystemExit(f"STOP: {len(missing)} images not found under {images_dir}")
print("preflight OK")
PY

for k in $FOLDS; do
    OUT="$OUTDIR/clip_ftpf_${TARGET}_v4_fold${k}.npz"
    if [ -f "$OUT" ]; then
        echo "-- fold $k already done, skipping"
        continue
    fi
    echo "================================================================"
    echo "-- fold $k"
    echo "================================================================"
    START=$(date +%s)
    uv run python -u src/data/finetune_clip_perfold.py \
        --fold "$k" --target "$TARGET" --v4 \
        --data_dir Dataset/maked \
        --images_dir "$IMAGES_DIR" \
        --v4_split_dir Dataset/split_v4_10group \
        --out "$OUT" \
        --epochs "$EPOCHS" --batch_size "$BATCH" --lr "$LR" --seed "$SEED"
    echo "-- fold $k done in $(( ($(date +%s) - START) / 60 )) min"
done

# --- verify before anyone uses these features -------------------------
uv run python - "$OUTDIR" "$TARGET" <<'PY'
import sys, os, glob
import numpy as np

outdir, target = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(outdir, f"clip_ftpf_{target}_v4_fold*.npz")))
sets = {}
for f in files:
    z = np.load(f, allow_pickle=True)
    ids, feats = z["stimulus_ids"], z["features"]
    sets[f] = set(map(str, ids))
    ok = "OK " if feats.shape[1] == 512 else "BAD"
    print(f"[{ok}] {os.path.basename(f)}: {feats.shape}")

if sets:
    base = next(iter(sets.values()))
    same = all(s == base for s in sets.values())
    print(("[OK ] " if same else "[BAD] ") + "all folds cover the same stimuli")
    print(("[OK ] " if len(base) == 6526 else "[BAD] ")
          + f"stimulus count: {len(base)} (expected 6526)")
PY

echo
echo "features written to $OUTDIR"
