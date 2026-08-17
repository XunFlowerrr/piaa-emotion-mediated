"""Mediator = the 7-dimensional layer sitting between image features and a
personal score.

Every mediator is fit on **train-user images only**, then frozen and
shared across users -- all the personalization lives in the head.

Mediators used in the paper:

  identity  no mediator (Direct) -- head runs on raw 512-dim features
  emotion   predicts 7 emotions (our proposal) -- a mediator with meaning
  pca       unsupervised 7-dim compression -- controls for "is the gain
            just dimensionality reduction?"
  random    random linear projection to 7 dims -- controls for "does any
            7-dim mediator work?"
  shuffled  emotion predictions shuffled across images -- keeps the
            distribution, destroys the meaning

*** reproducibility note ***
random and shuffled draw from the same per-fold generator, in order: R
first, then the permutation. Reorder or split the generator and the
numbers change even though nothing about the method did.
build_shared_mediators() always draws both, regardless of which mediators
were actually requested.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _shared_ridge(Xg, Yg, alphas, val=None):
    """Ridge for a shared (population-level) mediator.

    The mediator is a shared component, so its penalty is chosen on the
    held-out validation user group when one is available -- those users are
    disjoint from both the train users it is fit on and the test users it is
    scored on. Falls back to RidgeCV's internal generalized CV only if no
    validation data was passed (used by ad-hoc scripts, never by the paper).
    """
    from src.modeling.heads import select_alpha_on_val

    alphas = np.asarray(alphas, float)
    if val is None:
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    else:
        m = make_pipeline(StandardScaler(),
                          Ridge(alpha=select_alpha_on_val(Xg, Yg, val, alphas)))
    m.fit(Xg, Yg)
    return m


class Mediator(ABC):
    name: str = "mediator"
    label: str = "Mediator"

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map image features to the mediator's output."""


class IdentityMediator(Mediator):
    """Direct -- no mediator, raw features go straight to the head."""
    name, label = "identity", "Direct"

    def transform(self, X):
        return np.asarray(X, float)


class EmotionMediator(Mediator):
    """Predicts 7 emotions from an image, using a model shared across users."""
    name, label = "emotion", "Hybrid (ours)"

    def __init__(self, model):
        self.model = model

    def transform(self, X):
        return self.model.predict(X)


class PCAMediator(Mediator):
    name, label = "pca", "PCA"

    def __init__(self, pca: PCA):
        self.pca = pca

    def transform(self, X):
        return self.pca.transform(X)


class RandomMediator(Mediator):
    name, label = "random", "Random"

    def __init__(self, R: np.ndarray):
        self.R = R

    def transform(self, X):
        return np.asarray(X, float) @ self.R


class ShuffledMediator(Mediator):
    """Emotion predictions shuffled across images -- realistic values, wrong image."""
    name, label = "shuffled", "Shuffled"

    def __init__(self, model):
        self.model = model

    def transform(self, X):
        return self.model.predict(X)


def build_shared_mediators(Xg: np.ndarray, Eg: np.ndarray, cfg, fold_index: int,
                           want: list[str] | None = None,
                           seed: int = 0,
                           val: tuple | None = None,
                           yg: np.ndarray | None = None,
                           val8: tuple | None = None,
                           Dg: dict | None = None,
                           val_dist: dict | None = None) -> dict[str, Mediator]:
    """Build every mediator from train-user data (population-level images).

    Every mediator is fit by ridge (Stage-1 is always linear, regardless of
    which Stage-2 head is later tested against it) so the ridge-vs-mlp rows
    in Table 1 differ only in the head, not in how the mediator was fit.

    Xg  features of images train users rated (n_img, d)
    Eg  population-mean emotion ratings for those images (n_img, 7)
    seed  run-level seed for multi-seed averaging (random/shuffled are
          stochastic); seed=0 reproduces the original RNG exactly.
    val  (X_val, E_val) from the held-out validation user group; the ridge
         penalty of every fitted mediator is selected on it. The emotion and
         shuffled mediators get the exact same treatment, so the control
         differs from the real thing only in the labels it saw.

    Random-draw order is fixed for reproducibility, see module docstring.
    """
    K = cfg.mediator_width
    want = want or ["identity", "emotion", "pca", "random", "shuffled"]

    # fixed RNG order: R first, then the permutation -- don't reorder.
    # The width-matched controls need a K+1-wide projection, but drawing one
    # here would shift every value after the first row (a (d,K+1) draw sliced
    # to K is NOT a (d,K) draw -- the generator fills row-major), which would
    # silently move the published random and shuffled numbers. The extra
    # column is therefore drawn last, after everything that already existed.
    rng_seed = fold_index if seed == 0 else fold_index + seed * 1_000_003
    rng = np.random.default_rng(rng_seed)
    R = rng.standard_normal((Xg.shape[1], K)) / np.sqrt(Xg.shape[1])
    perm = rng.permutation(len(Eg))
    R_extra = rng.standard_normal((Xg.shape[1], 1)) / np.sqrt(Xg.shape[1])
    R_full = np.hstack([R, R_extra])

    out: dict[str, Mediator] = {}
    if "identity" in want:
        out["identity"] = IdentityMediator()
    if "emotion" in want:
        out["emotion"] = EmotionMediator(
            _shared_ridge(Xg, Eg, cfg.ridge_alphas, val))
    if "pca" in want:
        out["pca"] = PCAMediator(PCA(n_components=K, random_state=0).fit(Xg))
    if "random" in want:
        out["random"] = RandomMediator(R)
    if "shuffled" in want:
        out["shuffled"] = ShuffledMediator(
            _shared_ridge(Xg, Eg[perm], cfg.ridge_alphas, val))

    # --- distribution-valued Stage-1 ------------------------------------
    # Same seven named concepts, but Stage-1 predicts how the raters were
    # spread over the scale instead of only where they landed on average.
    # The bottleneck is still "the 7 emotions", so Stage-2 stays readable:
    #   emotion_sd    7 means + 7 standard deviations          (14 wide)
    #   emotion_hist  7 emotions x 5 rating bins               (35 wide)
    # Dg is supplied by the caller because it has to be built from the raw
    # per-rater rows, which this module never sees.
    for key in ("emotion_sd", "emotion_hist"):
        if key in want:
            if Dg is None or key not in Dg:
                raise ValueError(f"{key} needs Dg['{key}'] (per-image targets)")
            out[key] = EmotionMediator(
                _shared_ridge(Xg, Dg[key], cfg.ridge_alphas,
                              (val_dist or {}).get(key)))

    # --- width-matched controls -------------------------------------------
    # Variant A gives Hybrid an 8th feature (the GIAA prediction), so a K-wide
    # control is no longer the same shape as the thing it is controlling for.
    # These are K+1 wide and carry no population prediction, which separates
    # "the extra width helped" from "the GIAA prediction helped".
    if "pca8" in want:
        out["pca8"] = PCAMediator(PCA(n_components=K + 1, random_state=0).fit(Xg))
    if "random8" in want:
        out["random8"] = RandomMediator(R_full)
    if "shuffled8" in want:
        # shuffle the [emotions, mean score] block together, so all K+1
        # outputs are real-looking values attached to the wrong images
        if yg is None:
            raise ValueError("shuffled8 needs yg (population mean score)")
        target = np.column_stack([Eg, np.asarray(yg, float)])
        out["shuffled8"] = ShuffledMediator(
            _shared_ridge(Xg, target[perm], cfg.ridge_alphas, val8))

    return out
