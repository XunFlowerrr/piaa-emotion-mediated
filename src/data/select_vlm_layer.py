"""Select one layer and token-type from vlm_features.npz and save as a 2D features file.
=====================================================================================
vlm_features.npz (extracted by extract_vlm_features.py) contains all layers:
  LT [N,n_layers,H] (text tokens) / LV [N,n_layers,H] (image tokens)

This script extracts the selected layer and type, optionally applies L2-normalization,
and saves it as a 2D features file (keys: stimulus_ids, features [N,H]), matching the format
of clip_features.npz so that it can be fed directly to the training pipeline.

Usage:
  python src/data/select_vlm_layer.py --vlm vlm_features.npz \
    --type LT --layer 15 --out vlm_LT15.npz
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--type", choices=["LT", "LV"], default="LT")
    ap.add_argument("--layer", type=int, required=True,
                    help="Index of the hidden state layer (0 = embedding layer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_norm", action="store_true",
                    help="Disable L2-normalization (default: normalized)")
    args = ap.parse_args()

    z = np.load(args.vlm, allow_pickle=True)
    layers = list(z["layers"])
    if args.layer not in layers:
        raise SystemExit(f"Layer {args.layer} not found in file; available layers are: {layers}")
    li = layers.index(args.layer)
    feats = z[args.type][:, li, :].astype(np.float32)   # [N, H]
    if not args.no_norm:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.clip(norms, 1e-8, None)

    np.savez_compressed(args.out,
                        stimulus_ids=z["stimulus_ids"],
                        features=feats)
    print(f"Saved {args.out} | {args.type} L{args.layer} | Shape: {feats.shape}")


if __name__ == "__main__":
    main()
