"""Fine-tune CLIP per fold in a leak-free manner.
==============================================================================
Fine-tunes CLIP's vision tower to predict the population-mean target
(emotions or overall score), then extracts features for all images to
be used for personalized prediction. Run once per fold; produces
clip_ftpf_{target}_v4_fold{k}.npz.

*** features must come out of the same place as the frozen baseline ***
extract_clip_features.py uses CLIPModel.get_image_features(), which is
visual_projection(vision_model(px).pooler_output) and gives 512 dims. So the
projection is kept and fine-tuned here too, and features are read after it.
Reading pooler_output directly instead would give 768 dims, and then the
backbone comparison would be varying the feature width at the same time as
the fine-tuning, with no way to tell which one moved the result.

Usage:
  python src/data/finetune_clip_perfold.py --fold 0 --target overall --v4 \
    --out clip_ftpf_overall_v4_fold0.npz --epochs 8 --batch_size 12
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm as _tqdm
from transformers import CLIPImageProcessor

from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPModel

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def tqdm(it, **kw):
    """Progress bar only on a real terminal.

    Captured output turns the per-batch bar into thousands of log lines,
    and once the host's log buffer fills, the next write blocks and the
    run appears to hang. TQDM_DISABLE=1 forces it off anywhere."""
    import os as _os
    off = _os.environ.get("TQDM_DISABLE") == "1" or not sys.stderr.isatty()
    return _tqdm(it, disable=off, **kw)

#: the 7 emotions the mediator actually uses, in ratings.csv's own spelling.
#: "Like" and "Beautiful" are deliberately NOT here: both are near-restatements
#: of the Aesthetic score, so fine-tuning on them would train the backbone
#: toward the target under an emotion label, and the resulting features would
#: carry the same objection as the score fine-tuned ones. Keep this list in
#: step with CORE7 in src/data/data.py.
CORE7_RAW = ["Distasteful", "Impressed", "Intellectually",
             "Motivated", "Nostalgic", "Sad", "Amused"]


def build_path_map(images_dir: Path) -> dict:
    """Map filename to absolute path (recursive search, ignoring __MACOSX)."""
    m = {}
    for p in images_dir.rglob("*"):
        if "__MACOSX" in p.parts:
            continue
        if p.suffix.lower() in IMG_EXTS:
            m[p.name] = p
    return m


def resolve(sf: str) -> str:
    """Map scenery video files (.mp4) to pre-extracted .jpg frames."""
    return sf[:-4] + ".jpg" if sf.lower().endswith(".mp4") else sf


class ImgDS(Dataset):
    """Simple image Dataset for CLIP feature extraction and fine-tuning."""
    def __init__(self, ids, paths, targets, proc):
        self.ids, self.paths, self.y, self.proc = ids, paths, targets, proc

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        px = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        return px, torch.tensor(self.y[i], dtype=torch.float32), i


class CLIPRegressor(nn.Module):
    """CLIP vision tower + its projection, with a linear head for regression.

    feat is taken after visual_projection, so it is the same 512-dim space
    CLIPModel.get_image_features() returns and the frozen baseline uses.
    """
    def __init__(self, out_dim):
        super().__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.vision = clip.vision_model
        self.visual_projection = clip.visual_projection      # 768 -> 512
        self.feat_dim = clip.config.projection_dim           # 512
        self.head = nn.Linear(self.feat_dim, out_dim)

    def forward(self, px, return_feat=False):
        out = self.vision(pixel_values=px)
        feat = self.visual_projection(out.pooler_output)
        if return_feat:
            return feat
        return self.head(feat)


def read_ids(path, sf2sid):
    """Read sample IDs from file path."""
    out = []
    for l in open(path, encoding="utf-8"):
        if l.strip():
            sf = l.rstrip("\n").split("\t")[-1]
            if sf in sf2sid:
                out.append(sf2sid[sf])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--target", choices=["emotion", "overall"], required=True)
    ap.add_argument("--data_dir", default="Dataset/maked")
    ap.add_argument("--images_dir", default="Dataset/sample")
    ap.add_argument("--split_dir", default="Dataset/split")
    ap.add_argument("--v4", action="store_true",
                    help="use the v4 group split (leak-free); --fold is 0-based")
    ap.add_argument("--v4_split_dir", default="Dataset/split_v4_10group",
                    help="where the v4 fold directories live; set this when "
                         "running outside the repo root (e.g. on Kaggle)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    # DataLoader workers are respawned at every epoch boundary unless they
    # are persistent, and spawning several of them on a small-CPU box (a
    # Kaggle kernel has 4 vCPU) can deadlock there -- the symptom is a run
    # that stops dead between two epochs with no error. 2 persistent
    # workers keeps the loader fed without that risk; 0 disables
    # multiprocessing entirely if it ever happens again.
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    # mps is Apple silicon; it has no autocast/GradScaler support here, so the
    # mixed-precision path below stays off for it and it trains in fp32
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    torch.manual_seed(args.seed)
    tgt_cols = CORE7_RAW if args.target == "emotion" else ["Aesthetic"]
    missing = [c for c in tgt_cols if c not in pd.read_csv(
        f"{args.data_dir}/ratings.csv", nrows=1).columns]
    if missing:
        raise SystemExit(f"ratings.csv is missing target columns: {missing}")
    print(f"Device: {device} | Fold: {args.fold} | Target: {args.target}", flush=True)

    df = pd.read_csv(f"{args.data_dir}/ratings.csv")
    sf2sid = dict(zip(df["sample_file"].astype(str), df["sample_id"].astype(str)))

    if args.v4:
        # v4 leak-free: train users + GIAA images
        fdir = Path(args.v4_split_dir) / f"fold{args.fold}"
        gen = {int(x) for x in (fdir / "train_users.txt").read_text().split()}
        giaa = {str(x) for x in (fdir / "giaa_train_images.txt").read_text().split()}
    else:
        fdir = Path(args.split_dir) / f"v3_fold{args.fold}"
        # general users (same across genres within a fold), read from art
        gen = set()
        for fn in ("train_users_GIAA.txt", "val_users_GIAA.txt"):
            for l in open(fdir / "art" / fn, encoding="utf-8"):
                if l.strip():
                    gen.add(int(l.strip()))
        # GIAA train images union across 3 genres
        giaa = set()
        for g in ("art", "fashion", "scenery"):
            giaa.update(read_ids(fdir / g / "train_images_GIAA.txt", sf2sid))
    print(f"General users: {len(gen)} | GIAA images: {len(giaa)}", flush=True)

    # Calculate per-image target = mean over general users only
    dgen = df[df["user_id"].isin(gen)].copy()
    dgen["sid"] = dgen["sample_id"].astype(str)
    per = dgen.groupby("sid")[tgt_cols].mean()
    per = per.loc[[s for s in per.index if s in giaa]]   # GIAA images only
    train_ids = per.index.tolist()

    # Map image paths
    path_map = build_path_map(Path(args.images_dir))
    sf_of = dict(zip(df["sample_id"].astype(str), df["sample_file"].astype(str)))
    def pth(sid):
        return path_map.get(resolve(sf_of[sid]))
    keep = [s for s in train_ids if pth(s) is not None]
    Y = per.loc[keep, tgt_cols].to_numpy(np.float32)
    Ymean, Ystd = Y.mean(0), Y.std(0) + 1e-6
    Yn = (Y - Ymean) / Ystd
    paths = [pth(s) for s in keep]
    print(f"Train images (with file): {len(keep)} | Target dim: {len(tgt_cols)}", flush=True)

    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    full = ImgDS(keep, paths, Yn, proc)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(keep))
    n_val = int(len(keep) * args.val_frac)
    va, tr = perm[:n_val].tolist(), perm[n_val:].tolist()
    nw = args.num_workers
    dl = dict(num_workers=nw, persistent_workers=nw > 0)
    tr_loader = DataLoader(Subset(full, tr), batch_size=args.batch_size,
                           shuffle=True, **dl)
    va_loader = DataLoader(Subset(full, va), batch_size=args.batch_size,
                           shuffle=False, **dl)

    model = CLIPRegressor(len(tgt_cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    amp = (device == "cuda")          # fp16 only on CUDA; mps/cpu run fp32
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    lossf = nn.MSELoss()

    # Keep the curve, not just the printed line: without it the only record
    # of how training went is whatever console log happens to survive, and a
    # log cannot be matched back to the .npz it produced.
    history = []
    best, best_state, best_ep = 1e9, None, -1
    for ep in range(args.epochs):
        model.train(); tl = 0.0
        for px, y, _ in tqdm(tr_loader, desc=f"f{args.fold} ep{ep}"):
            px, y = px.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                loss = lossf(model(px), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(px)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for px, y, _ in va_loader:
                px, y = px.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=amp):
                    vl += lossf(model(px), y).item() * len(px)
        tl /= len(tr); vl /= max(1, n_val)
        print(f"Fold {args.fold} Ep {ep}: train_loss={tl:.4f} val_loss={vl:.4f}", flush=True)
        history.append((ep, tl, vl))
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    # Extract features for all 6526 images (inference only)
    allsid = df["sample_id"].astype(str).drop_duplicates().tolist()
    allsid = [s for s in allsid if pth(s) is not None]
    allpaths = [pth(s) for s in allsid]
    ds = ImgDS(allsid, allpaths, np.zeros((len(allsid), len(tgt_cols)), np.float32), proc)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, **dl)
    feats = np.zeros((len(allsid), model.feat_dim), np.float32)
    model.eval()
    with torch.no_grad():
        for px, _, idxb in tqdm(loader, desc=f"f{args.fold} extract"):
            with torch.amp.autocast("cuda", enabled=amp):
                f = model(px.to(device), return_feat=True)
            feats[idxb.numpy()] = f.float().cpu().numpy()
    # history/best_epoch travel with the features, so any later claim about
    # the training run can be checked against the file it came from.
    np.savez_compressed(args.out, stimulus_ids=np.array(allsid), features=feats,
                        history=np.array(history, np.float32),
                        best_epoch=np.int32(best_ep))
    print(f"Saved {args.out} {feats.shape} (best val MSE={best:.4f})", flush=True)


if __name__ == "__main__":
    main()
