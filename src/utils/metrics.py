# Every metric used here -- CCC / PLCC / SROCC / Wilcoxon / effective DoF.
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon


def ccc(y_true, y_pred) -> float:
    # Concordance correlation coefficient
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mt, mp = y_true.mean(), y_pred.mean()
    cov = np.mean((y_true - mt) * (y_pred - mp))
    denom = y_true.var() + y_pred.var() + (mt - mp) ** 2
    return float(2 * cov / denom) if denom > 0 else 0.0


def srocc(y_true, y_pred) -> float:
    # Spearman rank-order correlation
    s = spearmanr(y_true, y_pred).statistic
    return float(s) if np.isfinite(s) else np.nan


def plcc(y_true, y_pred) -> float:
    # Pearson linear correlation
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if y_true.std() == 0 or y_pred.std() == 0:
        return np.nan
    r = pearsonr(y_true, y_pred)[0]
    return float(r) if np.isfinite(r) else np.nan


METRICS = {"ccc": ccc, "srocc": srocc, "plcc": plcc}

def evaluate(y_true, y_pred) -> dict[str, float]:
    return {k: f(y_true, y_pred) for k, f in METRICS.items()}


def wilcoxon_paired(a, b) -> float:
    """Paired Wilcoxon signed-rank -> p-value.

    Two models compared on the same units (user x domain), so this is
    always paired. Returns nan if all differences are zero or too few
    units remain.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 5 or np.sum(np.abs(a - b)) == 0:
        return np.nan
    return float(wilcoxon(a, b, method="approx").pvalue)


def mean_sd(values) -> tuple[float, float]:
    """Mean and sd (ddof=1) across units, ignoring nan."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), np.nan
    return float(v.mean()), float(v.std(ddof=1))


def sem(values) -> float:
    """Standard error of the mean (sd/sqrt(n)), ignoring nan.

    n is counted from `values` itself, never assumed -- tables differ in what
    one unit is (387 user-domain units for the model tables, 5 fold-domain
    estimates for Stage-1 accuracy)
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan
    return float(v.std(ddof=1) / np.sqrt(len(v)))


def effective_dof(X, alpha: float) -> float:
    """Ridge effective degrees of freedom: tr(Z(Z'Z + alpha*I)^-1 Z').

    Computed on standardized features (the pipeline always standardizes
    before RidgeCV). Only defined for linear heads.
    """
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(np.asarray(X, float))
    G = Z.T @ Z
    p = G.shape[0]
    return float(np.trace(Z @ np.linalg.solve(G + alpha * np.eye(p), Z.T)))
