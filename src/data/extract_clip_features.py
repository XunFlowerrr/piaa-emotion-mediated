"""Extract CLIP ViT-B/16 features of all images in XPASS-Vis.
================================================================================
Uses the same backbone as the paper: CLIP ViT-B/16, OpenAI pretrained weights, frozen.

Usage:
  python extract_clip_features.py --images_dir <images_folder> --ratings_csv Dataset/maked/ratings.csv

Output: clip_features.npz (keys: stimulus_ids, features [N,512])

Note:
  Scenery video files (.mp4) are mapped to pre-extracted .jpg frames of the same name.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def build_path_map(images_dir: Path) -> dict[str, Path]:
    """Map filename to absolute path (recursive search, ignoring __MACOSX)."""
    m = {}
    for p in images_dir.rglob("*"):
        if "__MACOSX" in p.parts:
            continue
        if p.suffix.lower() in IMG_EXTS:
            m[p.name] = p
    return m


def resolve_sample_file(sample_file: str) -> str:
    """Map .mp4 video files to pre-extracted .jpg frames."""
    if sample_file.lower().endswith(".mp4"):
        return sample_file[:-4] + ".jpg"
    return sample_file


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--ratings_csv", required=True)
    ap.add_argument("--out", default="clip_features.npz")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    df = pd.read_csv(args.ratings_csv)
    id_col = "stimulus_id" if "stimulus_id" in df.columns else "sample_id"
    sub = df[[id_col, "sample_file"]].drop_duplicates(subset=[id_col]).copy()
    sub[id_col] = sub[id_col].astype(str)
    print(f"Stimuli in CSV: {len(sub)}")

    path_map = build_path_map(Path(args.images_dir))
    sub["resolved_file"] = sub["sample_file"].map(resolve_sample_file)
    sub["path"] = sub["resolved_file"].map(path_map.get)

    missing = sub[sub["path"].isna()]
    if len(missing):
        print(f"Warning: {len(missing)} images not found, e.g., "
              f"{missing['resolved_file'].head(5).tolist()}")
        print("  Please check images_dir and path mappings.")
        sub = sub[sub["path"].notna()]
    stim_ids = sub[id_col].tolist()
    paths = sub["path"].tolist()
    print(f"Extracting features for {len(stim_ids)} images...")

    feats = []
    for i in tqdm(range(0, len(stim_ids), args.batch_size)):
        batch_paths = paths[i:i + args.batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        out = model.get_image_features(**inputs)
        emb = out.pooler_output if hasattr(out, "pooler_output") else out  # [B, 512]
        emb = emb / emb.norm(dim=-1, keepdim=True)          # L2-normalize
        feats.append(emb.cpu().numpy().astype(np.float32))

    features = np.concatenate(feats, axis=0)
    np.savez_compressed(args.out,
                        stimulus_ids=np.array(stim_ids),
                        features=features)
    print(f"Saved {features.shape} to: {args.out}")


if __name__ == "__main__":
    main()
