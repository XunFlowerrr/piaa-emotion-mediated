"""Extract VLM hidden features following Ryu & Yanaka (2026, arXiv:2604.11374).
==================================================================
"What Do Vision-Language Models Encode for Personalized Image Aesthetics?"

Model configurations:
  - Model: Qwen3-VL (2B/4B/8B)
  - Prompt: "Assess the aesthetics of this image."
  - Pooling: Average pooling over tokens
  - Features: LT (text tokens) and LV (vision tokens) across all decoder layers.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROMPT = "Assess the aesthetics of this image."


def build_path_map(images_dir: Path) -> dict:
    """Map filename to absolute path (recursive search, ignoring __MACOSX)."""
    m = {}
    for p in images_dir.rglob("*"):
        if "__MACOSX" in p.parts:
            continue
        if p.suffix.lower() in IMG_EXTS:
            m[p.name] = p
    return m


def resolve_sample_file(sample_file: str) -> str:
    """Map scenery video files (.mp4) to pre-extracted .jpg frames."""
    if sample_file.lower().endswith(".mp4"):
        return sample_file[:-4] + ".jpg"
    return sample_file


def find_image_token_id(model, processor) -> int:
    """Find image_token_id from configuration or processor tokenizer."""
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        tid = getattr(cfg, "image_token_id", None)
        if tid is not None:
            return int(tid)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        for name in ("<|image_pad|>", "<image>"):
            tid = tok.convert_tokens_to_ids(name)
            if tid is not None and tid >= 0:
                return int(tid)
    raise RuntimeError("image_token_id not found in model or processor config")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--ratings_csv", required=True)
    ap.add_argument("--out", default="vlm_features.npz")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--save_layers", default="all",
                    help='"all" or list like "10,12,15,18,20"')
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cpu":
        print("WARNING: GPU not found - running VLM on CPU is extremely slow.")
    print(f"Device: {device} | Model: {args.model} | Prompt: {args.prompt!r}")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("Loading model (downloading on first run)...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    img_token_id = find_image_token_id(model, processor)
    print(f"Model loaded. image_token_id={img_token_id}")

    # Stimulus-to-path lookup
    df = pd.read_csv(args.ratings_csv)
    id_col = "stimulus_id" if "stimulus_id" in df.columns else "sample_id"
    sub = df[[id_col, "sample_file"]].drop_duplicates(subset=[id_col]).copy()
    sub[id_col] = sub[id_col].astype(str)
    path_map = build_path_map(Path(args.images_dir))
    sub["resolved"] = sub["sample_file"].map(resolve_sample_file)
    sub["path"] = sub["resolved"].map(path_map.get)
    miss = sub[sub["path"].isna()]
    if len(miss):
        print(f"WARNING: {len(miss)} images not found, e.g., {miss['resolved'].head(3).tolist()}")
        sub = sub[sub["path"].notna()]
    stim_ids = sub[id_col].tolist()
    paths = sub["path"].tolist()
    print(f"Extracting features for {len(stim_ids)} images...")

    # Select layers to save
    want_layers = None if args.save_layers.strip() == "all" \
        else [int(x) for x in args.save_layers.split(",")]

    LT_all, LV_all = [], []
    layers_idx = None
    for p in tqdm(paths):
        img = Image.open(p).convert("RGB")
        msg = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": args.prompt}]}]
        text = processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[text], images=[img],
                        return_tensors="pt").to(device)
        out = model(**inp, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states                       # tuple of length n_layers + 1

        if layers_idx is None:
            layers_idx = list(range(len(hs))) if want_layers is None \
                else [L for L in want_layers if 0 <= L < len(hs)]

        ids = inp["input_ids"][0]
        img_mask = ids == img_token_id
        txt_mask = ~img_mask
        # Fallback: if no image tokens found, treat all tokens as text
        if img_mask.sum() == 0:
            img_mask = txt_mask

        lt_layers, lv_layers = [], []
        for L in layers_idx:
            h = hs[L][0].float()                     # [seq, H]
            lt_layers.append(h[txt_mask].mean(0).cpu().numpy())
            lv_layers.append(h[img_mask].mean(0).cpu().numpy())
        LT_all.append(np.stack(lt_layers))           # [n_layers, H]
        LV_all.append(np.stack(lv_layers))

    LT = np.stack(LT_all).astype(np.float16)          # [N, n_layers, H]
    LV = np.stack(LV_all).astype(np.float16)
    np.savez_compressed(args.out,
                        stimulus_ids=np.array(stim_ids),
                        layers=np.array(layers_idx),
                        LT=LT, LV=LV)
    size_gb = (LT.nbytes + LV.nbytes) / 1e9
    print(f"Saved {args.out}  LT{LT.shape} LV{LV.shape} (~{size_gb:.2f} GB)")
    print(f"Saved layers: {layers_idx}")


if __name__ == "__main__":
    main()
