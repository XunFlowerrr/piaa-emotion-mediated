"""Legacy script to sweep VLM layers and token-types using Direct Ridge.
==================================================================
Reads vlm_features.npz (all layers of LT/LV), evaluates each (type, layer)
combination, and reports average SROCC and CCC.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.xpass_loader import (load_interactions, load_official_folds,
                               load_piaa_image_splits)

ALPHAS = np.logspace(-2, 3, 11)


def ccc(y, p):
    """Calculate Concordance Correlation Coefficient."""
    my, mp = y.mean(), p.mean()
    cov = np.mean((y - my) * (p - mp))
    d = y.var() + p.var() + (my - mp) ** 2
    return float(2 * cov / d) if d > 0 else 0.0


def scc(y, p):
    """Calculate Spearman Rank Correlation."""
    s = spearmanr(y, p).statistic
    return float(s) if np.isfinite(s) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--n_train", type=int, default=100)
    ap.add_argument("--types", default="LT,LV")
    ap.add_argument("--layers", default="all", help='"all" or list like "8,12,15,18"')
    ap.add_argument("--out", default="vlm_layer_sweep.csv")
    args = ap.parse_args()

    z = np.load(args.vlm, allow_pickle=True)
    all_layers = list(int(x) for x in z["layers"])
    sids = [str(s) for s in z["stimulus_ids"]]
    sid_pos = {s: i for i, s in enumerate(sids)}
    types = args.types.split(",")
    layers = all_layers if args.layers == "all" \
        else [int(x) for x in args.layers.split(",")]

    df = load_interactions(args.data_dir).dropna(subset=["overall"])
    df = df[df["stimulus_id"].astype(str).isin(sid_pos)]
    dataset_root = Path(args.data_dir).resolve().parent
    folds = load_official_folds(dataset_root)
    sf2sid = dict(zip(df["sample_file"].astype(str),
                      df["stimulus_id"].astype(str)))
    piaa = load_piaa_image_splits(dataset_root, sf2sid)

    def eval_layer(feat_matrix):
        """Evaluate a specific feature matrix (StandardScaler + RidgeCV)."""
        rows = []
        for fi, fold in enumerate(folds):
            for dom in ["art", "fashion", "landscape"]:
                sub = df[df["domain"] == dom]
                sd = piaa[fi].get(dom, {})
                for uid in fold["target"]:
                    su = sd.get(uid)
                    if su is None:
                        continue
                    dfi = sub[sub["user_id"] == uid]
                    yb = (dfi.groupby(dfi["stimulus_id"].astype(str))["overall"]
                             .mean())
                    idx = set(yb.index)
                    tr = [s for s in su["train"] if s in idx][:args.n_train]
                    te = [s for s in su["test"] if s in idx]
                    if len(tr) < args.n_train or len(te) < 20:
                        continue
                    Xtr = feat_matrix[[sid_pos[s] for s in tr]]
                    Xte = feat_matrix[[sid_pos[s] for s in te]]
                    ytr = yb.loc[tr].to_numpy(float)
                    yte = yb.loc[te].to_numpy(float)
                    m = make_pipeline(StandardScaler(),
                                      RidgeCV(alphas=ALPHAS)).fit(Xtr, ytr)
                    pr = m.predict(Xte)
                    rows.append((dom, ccc(yte, pr), scc(yte, pr)))
        r = pd.DataFrame(rows, columns=["domain", "ccc", "scc"])
        g = r.groupby("domain").mean(numeric_only=True)
        return g["ccc"].mean(), g["scc"].mean(), g

    results = []
    for t in types:
        arr = z[t]
        for L in layers:
            li = all_layers.index(L)
            feat = arr[:, li, :].astype(np.float32)
            avg_ccc, avg_scc, _ = eval_layer(feat)
            results.append({"type": t, "layer": L,
                            "CCC": avg_ccc, "SROCC": avg_scc})
            print(f"{t} L{L:>2}: CCC={avg_ccc:.4f}  SROCC={avg_scc:.4f}", flush=True)

    out = pd.DataFrame(results).sort_values("CCC", ascending=False)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parents[2] / "output" / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise SystemExit(f"Error: {out_path.name} already exists. Please use a different --out filename.")
    out.to_csv(out_path, index=False)
    print("\n=== TOP 10 (by Direct CCC, n_train={}) ===".format(args.n_train))
    print(out.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
