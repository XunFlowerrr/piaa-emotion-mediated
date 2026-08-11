"""Head = maps "mediator output" to "this user's beauty score".

The head is the one layer that's personal to a user. Two kinds:

  RidgeHead - linear, interpretable (7 weights = that user's formula).
              alpha picked by RidgeCV (generalized CV) from 11 values.
  MLPHead   - nonlinear, single hidden layer of 128 units, MSE loss, no L2.
              picks a learning rate on a validation split, then refits.

Both train **sequentially**: the mediator is fit and frozen first, then the
head is fit on whatever the mediator outputs. Not end-to-end.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.metrics import effective_dof


class Head(ABC):
    """Common interface for every head."""

    name: str = "head"
    is_linear: bool = False

    @abstractmethod
    def fit(self, M: np.ndarray, y: np.ndarray) -> "Head":
        """M = mediator output (n_samples, width), y = user's scores."""

    @abstractmethod
    def predict(self, M: np.ndarray) -> np.ndarray:
        ...

    def effective_dof(self) -> float:
        """Effective degrees of freedom -- only defined for linear heads."""
        return np.nan

    def weights(self) -> np.ndarray | None:
        """Coefficients, if linear -- used for interpretability."""
        return None


class RidgeHead(Head):
    name = "ridge"
    is_linear = True

    def __init__(self, alphas):
        self.alphas = np.asarray(alphas, float)
        self._pipe = None
        self._M_train = None

    def fit(self, M, y):
        self._pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=self.alphas))
        self._pipe.fit(M, y)
        self._M_train = np.asarray(M, float)
        return self

    def predict(self, M):
        return self._pipe.predict(M)

    @property
    def alpha_(self) -> float:
        return float(self._pipe.named_steps["ridgecv"].alpha_)

    def effective_dof(self) -> float:
        return effective_dof(self._M_train, self.alpha_)

    def weights(self):
        return self._pipe.named_steps["ridgecv"].coef_.ravel().copy()


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

    def fit(self, M, y):
        M = np.asarray(M, float)
        y = np.asarray(y, float)
        n = len(M)
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(n)
        nv = max(10, int(self.cfg.mlp_search_val_frac * n))
        va, tr = idx[:nv], idx[nv:]

        best, best_lr = np.inf, self.cfg.mlp_lr_grid[0]
        for lr in self.cfg.mlp_lr_grid:
            m = self._make(lr)
            m.fit(M[tr], y[tr])
            mse = float(np.mean((m.predict(M[va]) - y[va]) ** 2))
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
