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
#: instead of by rounding noise. This narrows the cross-platform gap but does
#: not close it: see "What reproducible does and does not promise" in
#: docs/METHODOLOGY.md for what actually holds across machines.
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

    def fit(self, M, y, val=None, frozen_alpha=None):
        """frozen_alpha, if given, skips selection entirely -- used when the
        penalty was already chosen once for the whole (fold, domain, n_train)
        by mirroring the test protocol inside the validation group, and is
        then applied identically to every test user (see
        Pipeline.select_personal_hyperparam)."""
        if frozen_alpha is not None:
            self._alpha = float(frozen_alpha)
            self._pipe = make_pipeline(StandardScaler(), Ridge(alpha=self._alpha))
            self._pipe.fit(M, y)
        elif val is None:
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

    Trains for a fixed cfg.mlp_max_iter epochs; early_stopping is off.
    early_stopping=True carves an internal validation_fraction split out of
    whatever M it's given, and at a support size of 10 that split is 1-2
    samples -- not a usable stopping signal, and it silently shrinks the data
    the fit actually sees. A fixed epoch count is a stated, identical budget
    for every unit instead. (Previously used early_stopping to stop before
    100-sample support sets memorized the data; that overfitting risk is now
    compensated by choosing lr on the validation group rather than by
    per-unit early stopping. See docs/METHODOLOGY.md.)
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

    def fit(self, M, y, val=None, frozen_lr=None):
        """frozen_lr, if given, skips the lr search entirely -- same role as
        RidgeHead's frozen_alpha (see Pipeline.select_personal_hyperparam)."""
        M = np.asarray(M, float)
        y = np.asarray(y, float)

        if frozen_lr is not None:
            best_lr = float(frozen_lr)
        elif val is None:
            # personal head: hold out part of this user's own support set
            n = len(M)
            rng = np.random.RandomState(self.seed)
            idx = rng.permutation(n)
            nv = max(10, int(self.cfg.mlp_search_val_frac * n))
            va, tr = idx[:nv], idx[nv:]
            M_tr, y_tr, M_va, y_va = M[tr], y[tr], M[va], y[va]
            best_lr = self._search(M_tr, y_tr, M_va, y_va)
        else:
            # shared component: score on the held-out validation user group
            M_va, y_va = np.asarray(val[0], float), np.asarray(val[1], float)
            best_lr = self._search(M, y, M_va, y_va)

        self.best_lr_ = float(best_lr)
        self._pipe = self._make(best_lr)
        self._pipe.fit(M, y)
        return self

    def _search(self, M_tr, y_tr, M_va, y_va) -> float:
        # same tie-break spirit as select_alpha_on_val: don't let the last few
        # bits of a platform-dependent MSE pick the learning rate. Ties within
        # ALPHA_TIE_RTOL go to the smallest lr (mlp_lr_grid is ascending),
        # since a smaller step is the more conservative, less divergence-prone
        # choice -- the MLP analogue of "the stronger penalty" for ridge.
        grid = np.asarray(self.cfg.mlp_lr_grid, float)
        mses = np.empty(len(grid))
        for i, lr in enumerate(grid):
            m = self._make(float(lr))
            m.fit(M_tr, y_tr)
            mses[i] = np.mean((m.predict(M_va) - y_va) ** 2)
        tied = mses <= mses.min() * (1.0 + ALPHA_TIE_RTOL)
        return float(grid[tied][0])

    def predict(self, M):
        return self._pipe.predict(M)


def make_head(kind: str, cfg, seed: int = 0) -> Head:
    if kind == "ridge":
        return RidgeHead(cfg.ridge_alphas)
    if kind == "mlp":
        return MLPHead(cfg, seed=seed)
    raise KeyError(f"unknown head '{kind}' (have: ridge, mlp)")
