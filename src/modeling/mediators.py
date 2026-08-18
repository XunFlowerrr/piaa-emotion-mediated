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


class JointMediator(Mediator):
    """Stage-1 trained *jointly* with a population score head.

    Koh et al.'s joint bottleneck backpropagates one loss through g (x->c) and
    f (c->y) together. Here f is personal -- one head per user, fit on ~100 of
    that user's ratings -- so a literally joint fit would either make Stage-1
    personal too (512x7 weights per user, from 100 samples, and the paper's
    "7 parameters per user" claim gone) or train Stage-1 on the test users'
    own scores, which is the leak the v4 split exists to prevent.

    What is trained jointly here is therefore Stage-1 with a *population*
    score head, on the training group only:

        loss = MSE(g(x), c_pop) + joint_score_weight * MSE(h(g(x)), y_pop)

    The score term shapes the seven concepts to be useful for predicting a
    score, which is the point of joint training; the personal head is then fit
    on the frozen output exactly as in every other row, so the comparison and
    the parameter count are unchanged.
    """
    name, label = "emotion_joint", "Hybrid (joint Stage-1)"

    def __init__(self, model):
        self.model = model

    def transform(self, X):
        return self.model.predict(X)


class _JointBottleneckNet:
    """512 -> hidden -> 7 concepts -> 1 score, trained with one combined loss.

        loss = MSE(c, c_pop) + w * MSE(v.c + b, y_pop)

    The score head reads the **seven concepts**, not the hidden layer, so the
    score gradient is forced to flow through the bottleneck -- that is what
    makes this joint in Koh et al.'s sense. Routing the score head off the
    hidden layer instead would shape the trunk while leaving the seven
    concepts under emotion supervision only, which is sequential training
    wearing a joint label.

    Plain numpy + Adam rather than torch: torch is not a dependency of the
    analysis environment, and a 512->128->7->1 net on ~4500 rows does not
    need one. Weights are seeded, so runs reproduce.

    Trained on the training group only and then frozen, exactly like every
    other Stage-1. It is never fit on a test user's own ratings: making
    Stage-1 personal is the Delta model that overfits ~100 images, and the
    "seven parameters per user" claim depends on Stage-1 staying shared.
    """

    def __init__(self, d_in, d_hidden, n_concept, lr, w_score, max_iter, seed):
        rng = np.random.default_rng(int(seed))
        self.W1 = rng.standard_normal((d_in, d_hidden)) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.standard_normal((d_hidden, n_concept)) * np.sqrt(2.0 / d_hidden)
        self.b2 = np.zeros(n_concept)
        self.v = rng.standard_normal(n_concept) * np.sqrt(2.0 / n_concept)
        self.c0 = 0.0
        self.lr, self.w_score, self.max_iter = float(lr), float(w_score), int(max_iter)
        self.mu = self.sd = None

    def _params(self):
        return ["W1", "b1", "W2", "b2", "v", "c0"]

    def _forward(self, X):
        H = np.maximum(X @ self.W1 + self.b1, 0.0)      # relu
        C = H @ self.W2 + self.b2                        # the bottleneck
        y = C @ self.v + self.c0
        return H, C, y

    def fit(self, X, Cp, yp):
        X = np.asarray(X, float)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = (X - self.mu) / self.sd
        Cp, yp = np.asarray(Cp, float), np.asarray(yp, float).ravel()
        n = len(Z)

        m = {k: np.zeros_like(getattr(self, k), dtype=float) for k in self._params()}
        v = {k: np.zeros_like(getattr(self, k), dtype=float) for k in self._params()}
        b1_, b2_, eps = 0.9, 0.999, 1e-8

        for t in range(1, self.max_iter + 1):
            H, C, yh = self._forward(Z)
            dC = (2.0 / n) * (C - Cp)                       # emotion term
            dy = (2.0 * self.w_score / n) * (yh - yp)       # score term
            dC = dC + np.outer(dy, self.v)                  # through the bottleneck

            g = {"v": C.T @ dy, "c0": dy.sum(),
                 "W2": H.T @ dC, "b2": dC.sum(0)}
            dH = (dC @ self.W2.T) * (H > 0)
            g["W1"] = Z.T @ dH
            g["b1"] = dH.sum(0)

            for k in self._params():
                m[k] = b1_ * m[k] + (1 - b1_) * g[k]
                v[k] = b2_ * v[k] + (1 - b2_) * g[k] ** 2
                mh = m[k] / (1 - b1_ ** t)
                vh = v[k] / (1 - b2_ ** t)
                setattr(self, k, getattr(self, k) - self.lr * mh / (np.sqrt(vh) + eps))
        return self

    def loss(self, X, Cp, yp):
        """Combined objective -- the criterion the lr is selected on."""
        _, C, yh = self._forward((np.asarray(X, float) - self.mu) / self.sd)
        return (np.mean((C - np.asarray(Cp, float)) ** 2)
                + self.w_score * np.mean((yh - np.asarray(yp, float).ravel()) ** 2))

    def predict(self, X):
        _, C, _ = self._forward((np.asarray(X, float) - self.mu) / self.sd)
        return C                                    # the bottleneck, 7 wide


def _joint_stage1(Xg, Eg, yg, cfg, seed, val=None, val_y=None):
    """Fit the joint Stage-1 and return something with .predict(X).

    The learning rate is chosen on the validation user group, on this
    network's own combined objective -- the same rule every other shared
    component follows (val users are disjoint from train and test).
    """
    def build(lr):
        return _JointBottleneckNet(Xg.shape[1], cfg.mlp_hidden, Eg.shape[1],
                                   lr, cfg.joint_score_weight,
                                   cfg.stage1_mlp_max_iter, seed)

    grid = np.asarray(cfg.stage1_mlp_lr_grid, float)
    if val is None or val_y is None:
        lr = float(grid[len(grid) // 2])
    else:
        from src.modeling.heads import ALPHA_TIE_RTOL
        Xv, Ev = val
        losses = np.array([build(l).fit(Xg, Eg, yg).loss(Xv, Ev, val_y)
                           for l in grid])
        tied = losses <= losses.min() * (1.0 + ALPHA_TIE_RTOL)
        lr = float(grid[tied][0])          # ascending -> smallest step on a tie
    return build(lr).fit(Xg, Eg, yg)


def _select_mlp_lr(Xg, Yg, cfg, seed, val):
    """Learning rate for a shared MLP Stage-1, chosen on the validation group.

    Same rule the ridge mediators use: a shared component's hyperparameter is
    scored on users disjoint from both train and test. Falls back to the
    middle of the grid when no validation data was passed.
    """
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src.modeling.heads import ALPHA_TIE_RTOL

    grid = np.asarray(cfg.stage1_mlp_lr_grid, float)
    if val is None:
        return float(grid[len(grid) // 2])
    Xv, Yv = val
    Yv = np.asarray(Yv, float)
    if Yv.shape[1] != np.asarray(Yg).shape[1]:      # joint target is 1 wider
        Yv = np.column_stack([Yv, np.zeros(len(Yv))])
    mses = np.empty(len(grid))
    for i, lr in enumerate(grid):
        m = make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(cfg.mlp_hidden,), activation="relu",
            alpha=cfg.mlp_alpha, solver="adam", learning_rate_init=float(lr),
            max_iter=cfg.stage1_mlp_max_iter, early_stopping=False,
            random_state=int(seed)))
        m.fit(Xg, Yg)
        mses[i] = np.mean((m.predict(Xv) - Yv) ** 2)
    tied = mses <= mses.min() * (1.0 + ALPHA_TIE_RTOL)
    return float(grid[tied][0])          # ascending -> smallest step wins a tie


def build_shared_mediators(Xg: np.ndarray, Eg: np.ndarray, cfg, fold_index: int,
                           want: list[str] | None = None,
                           seed: int = 0,
                           val: tuple | None = None,
                           yg: np.ndarray | None = None,
                           val_y: np.ndarray | None = None,
                           Dg: dict | None = None,
                           val_dist: dict | None = None) -> dict[str, Mediator]:
    """Build every mediator from train-user data (population-level images).

    Stage-1 is ridge by default, independent of which Stage-2 head is tested
    against it, so the ridge-vs-mlp rows differ only in the head. Two explicit
    alternatives make Stage-1 its own axis instead of a side effect of the
    head: "emotion_mlp" (nonlinear, still fit on the emotions alone) and
    "emotion_joint" (nonlinear, fit on emotions and score together).

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

    # fixed RNG order: R first, then the permutation -- don't reorder, or the
    # published random and shuffled numbers move even though nothing about
    # the method did.
    rng_seed = fold_index if seed == 0 else fold_index + seed * 1_000_003
    rng = np.random.default_rng(rng_seed)
    R = rng.standard_normal((Xg.shape[1], K)) / np.sqrt(Xg.shape[1])
    perm = rng.permutation(len(Eg))

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

    # --- Stage-1 as its own axis -----------------------------------------
    if "emotion_mlp" in want:
        # same target as "emotion", nonlinear fit: isolates Stage-1 capacity
        from sklearn.neural_network import MLPRegressor
        lr = _select_mlp_lr(Xg, Eg, cfg, seed, val)
        out["emotion_mlp"] = EmotionMediator(make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(cfg.mlp_hidden,), activation="relu",
                         alpha=cfg.mlp_alpha, solver="adam",
                         learning_rate_init=lr,
                         max_iter=cfg.stage1_mlp_max_iter,
                         early_stopping=False, random_state=int(seed))
        ).fit(Xg, Eg))
    if "emotion_joint" in want:
        if yg is None:
            raise ValueError("emotion_joint needs yg (population mean score)")
        out["emotion_joint"] = JointMediator(
            _joint_stage1(Xg, Eg, yg, cfg, seed, val, val_y))

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

    return out
