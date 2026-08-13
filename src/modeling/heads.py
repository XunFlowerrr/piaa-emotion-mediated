"""Head = maps "mediator output" to "this user's beauty score".

The head is the one layer that's personal to a user. Two kinds:

  RidgeHead - linear, interpretable (7 weights = that user's formula).
  MLPHead   - nonlinear, single hidden layer of 128 units, MSE loss, no L2.

Both train **sequentially**: the mediator is fit and frozen first, then the
head is fit on whatever the mediator outputs. Not end-to-end.

*** where hyperparameters come from ***
Both `fit` methods take an optional `val=(X_val, y_val)`, and that decides
which of the two selection rules applies:

  val given (a SHARED component, e.g. the population/GIAA head) -> the
    hyperparameter is scored on the held-out validation *user group*, which
    is disjoint from both train and test users.
  val=None (a PERSONAL component, i.e. a per-user head) -> selected inside
    that user's own support set only. The validation group is other people,
    so it cannot speak to one user's taste; what matters is that the user's
    evaluation images are never touched, and they aren't.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.metrics import effective_dof


class Head(ABC):
    """Common interface for every head."""

    name: str = "head"
    is_linear: bool = False

    @abstractmethod
    def fit(self, M: np.ndarray, y: np.ndarray, val=None) -> "Head":
        """M = mediator output (n_samples, width), y = user's scores.

        val = (M_val, y_val) from the held-out validation user group, for
        shared components only. See module docstring.
        """

    @abstractmethod
    def predict(self, M: np.ndarray) -> np.ndarray:
        ...

    def effective_dof(self) -> float:
        """Effective degrees of freedom -- only defined for linear heads."""
        return np.nan

    def weights(self) -> np.ndarray | None:
        """Coefficients, if linear -- used for interpretability."""
        return None


#: two alphas whose validation MSE differs by less than this (relative) are
#: treated as tied. Floating-point arithmetic is not bit-identical across
#: BLAS builds, so a strict `<` could hand the win to a different alpha on
#: another machine and silently produce a different model. Anything inside
#: this margin is genuinely indistinguishable, so we break the tie by rule
#: instead of by rounding noise.
ALPHA_TIE_RTOL = 1e-9


def select_alpha_on_val(X, Y, val, alphas) -> float:
    """Pick the ridge penalty that does best on the held-out validation group.

    Fit on the train-group data, score MSE on the validation group, take the
    winner. Works for multi-output Y (the 7-emotion mediator) as well.

    Ties are broken toward the *strongest* penalty: among alphas that are
    statistically indistinguishable on the validation group, the most
    regularized one is the conservative choice, and picking it by rule makes
    the result reproducible rather than dependent on the last few bits.
    """
    Xv, Yv = val
    alphas = np.asarray(alphas, float)
    mses = np.empty(len(alphas))
    for i, a in enumerate(alphas):
        p = make_pipeline(StandardScaler(), Ridge(alpha=float(a)))
        p.fit(X, Y)
        mses[i] = np.mean((p.predict(Xv) - Yv) ** 2)
    tied = mses <= mses.min() * (1.0 + ALPHA_TIE_RTOL)
    return float(alphas[tied][-1])          # alphas is ascending -> strongest


class RidgeHead(Head):
    name = "ridge"
    is_linear = True

    def __init__(self, alphas):
        self.alphas = np.asarray(alphas, float)
        self._pipe = None
        self._M_train = None

    def fit(self, M, y, val=None):
        if val is None:
            self._pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=self.alphas))
            self._pipe.fit(M, y)
            self._alpha = float(self._pipe[-1].alpha_)
        else:
            self._alpha = select_alpha_on_val(M, y, val, self.alphas)
            self._pipe = make_pipeline(StandardScaler(), Ridge(alpha=self._alpha))
            self._pipe.fit(M, y)
        self._M_train = np.asarray(M, float)
        return self

    def predict(self, M):
        return self._pipe.predict(M)

    @property
    def alpha_(self) -> float:
        return float(self._alpha)

    def effective_dof(self) -> float:
        return effective_dof(self._M_train, self.alpha_)

    def weights(self):
        return self._pipe[-1].coef_.ravel().copy()


class MLPHead(Head):
    """Single hidden layer of 128 units, MSE loss, no L2.

    early_stopping=True is a stopping rule, not weight regularization --
    needed because on 100 samples, training to convergence memorizes the
    data (train loss ~0.003), and capping max_iter low instead just means
    it never converges. Checked both failure modes, see docs/METHODOLOGY.md.
    """

    name = "mlp"
    is_linear = False

    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        self.seed = int(seed)
        self._pipe = None
        self.best_lr_: float | None = None

    def _make(self, lr: float):
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(self.cfg.mlp_hidden,),
            activation="relu",
            alpha=self.cfg.mlp_alpha,
            solver="adam",
            learning_rate_init=lr,
            max_iter=self.cfg.mlp_max_iter,
            early_stopping=self.cfg.mlp_early_stopping,
            validation_fraction=self.cfg.mlp_validation_fraction,
            n_iter_no_change=self.cfg.mlp_n_iter_no_change,
            random_state=self.seed,
        ))

    def fit(self, M, y, val=None):
        M = np.asarray(M, float)
        y = np.asarray(y, float)

        if val is None:
            # personal head: hold out part of this user's own support set
            n = len(M)
            rng = np.random.RandomState(self.seed)
            idx = rng.permutation(n)
            nv = max(10, int(self.cfg.mlp_search_val_frac * n))
            va, tr = idx[:nv], idx[nv:]
            M_tr, y_tr, M_va, y_va = M[tr], y[tr], M[va], y[va]
        else:
            # shared component: score on the held-out validation user group
            M_va, y_va = np.asarray(val[0], float), np.asarray(val[1], float)
            M_tr, y_tr = M, y

        best, best_lr = np.inf, self.cfg.mlp_lr_grid[0]
        for lr in self.cfg.mlp_lr_grid:
            m = self._make(lr)
            m.fit(M_tr, y_tr)
            mse = float(np.mean((m.predict(M_va) - y_va) ** 2))
            if mse < best:
                best, best_lr = mse, lr
        self.best_lr_ = float(best_lr)
        self._pipe = self._make(best_lr)
        self._pipe.fit(M, y)
        return self

    def predict(self, M):
        return self._pipe.predict(M)


def make_head(kind: str, cfg, seed: int = 0) -> Head:
    if kind == "ridge":
        return RidgeHead(cfg.ridge_alphas)
    if kind == "mlp":
        return MLPHead(cfg, seed=seed)
    raise KeyError(f"unknown head '{kind}' (have: ridge, mlp)")
