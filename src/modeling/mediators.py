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
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _shared_ridge(alphas):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.asarray(alphas, float)))


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
                           emotion_mlp=None) -> dict[str, Mediator]:
    """Build every mediator from train-user data (population-level images).

    Xg  features of images train users rated (n_img, d)
    Eg  population-mean emotion ratings for those images (n_img, 7)
    emotion_mlp  if given, adds an "emotion_mlp" mediator (used with head=mlp)

    Random-draw order is fixed for reproducibility, see module docstring.
    """
    K = cfg.mediator_width
    want = want or ["identity", "emotion", "pca", "random", "shuffled"]

    # fixed RNG order: R first, then the permutation -- don't reorder
    rng = np.random.default_rng(fold_index)
    R = rng.standard_normal((Xg.shape[1], K)) / np.sqrt(Xg.shape[1])
    perm = rng.permutation(len(Eg))

    out: dict[str, Mediator] = {}
    if "identity" in want:
        out["identity"] = IdentityMediator()
    if "emotion" in want:
        m = _shared_ridge(cfg.ridge_alphas); m.fit(Xg, Eg)
        out["emotion"] = EmotionMediator(m)
    if "pca" in want:
        out["pca"] = PCAMediator(PCA(n_components=K, random_state=0).fit(Xg))
    if "random" in want:
        out["random"] = RandomMediator(R)
    if "shuffled" in want:
        m = _shared_ridge(cfg.ridge_alphas); m.fit(Xg, Eg[perm])
        out["shuffled"] = ShuffledMediator(m)
    if emotion_mlp is not None:
        out["emotion_mlp"] = EmotionMediator(emotion_mlp)
    return out


def fit_emotion_mlp(Xg: np.ndarray, Eg: np.ndarray, cfg, seed: int):
    """Nonlinear emotion mediator (used when the whole pipeline is MLP).

    Reuses MLPHead so the training recipe matches the head exactly (128
    hidden, MSE, no L2, lr picked on a validation split).
    """
    from src.modeling.heads import MLPHead
    return MLPHead(cfg, seed=seed).fit(Xg, Eg)
