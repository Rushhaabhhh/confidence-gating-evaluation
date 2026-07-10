"""
Post-hoc calibration methods and honest k-fold out-of-sample evaluation.

We compare three standard recalibration maps:
  - Temperature scaling  (Guo et al. 2017): single scalar, monotonic
  - Platt scaling        (Platt 1999): logistic a*x + b, monotonic
  - Isotonic regression  (Zadrozny & Elkan 2002): non-parametric monotonic

All three are fit on training folds and evaluated on held-out folds, so reported
ECE/Brier are out-of-sample -- they cannot be inflated by overfitting the map to
the same points it is scored on. With n ~ 150 a single split leaves too few test
points to trust one number, so we use K-fold and pool the out-of-fold predictions.

Falls back to numpy/scipy implementations if `netcal` is unavailable, so the
pipeline runs in a bare environment; when netcal is present we use it and say so.
"""
from __future__ import annotations

import numpy as np

from .metrics import expected_calibration_error, brier_score

try:
    from netcal.scaling import TemperatureScaling, LogisticCalibration
    from netcal.binning import IsotonicRegression as _NetcalIso
    _HAS_NETCAL = True
except Exception:  # pragma: no cover
    _HAS_NETCAL = False


# ---------------------------------------------------------------------------
# Fallback implementations (used only if netcal missing)
# ---------------------------------------------------------------------------

def _fit_temperature_np(conf, correct):
    from scipy.optimize import minimize_scalar
    eps = 1e-6
    conf = np.clip(conf, eps, 1 - eps)
    logits = np.log(conf / (1 - conf))
    y = correct.astype(float)

    def nll(T):
        p = 1 / (1 + np.exp(-logits / T))
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    T = minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x

    def apply(c):
        c = np.clip(c, eps, 1 - eps)
        lg = np.log(c / (1 - c))
        return 1 / (1 + np.exp(-lg / T))
    return apply


def _fit_platt_np(conf, correct):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(conf.reshape(-1, 1), correct)
    return lambda c: lr.predict_proba(np.asarray(c).reshape(-1, 1))[:, 1]


def _fit_isotonic_np(conf, correct):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    ir.fit(conf, correct)
    return lambda c: ir.predict(np.asarray(c))


# ---------------------------------------------------------------------------
# Unified fit: returns a function mapping raw confidence -> calibrated prob
# ---------------------------------------------------------------------------

def _fit_method(method: str, conf: np.ndarray, correct: np.ndarray):
    if _HAS_NETCAL:
        conf2d = np.stack([1 - conf, conf], axis=1)
        y = correct.astype(int)
        if method == "temperature":
            m = TemperatureScaling(); m.fit(conf2d, y)
        elif method == "platt":
            m = LogisticCalibration(); m.fit(conf2d, y)
        elif method == "isotonic":
            m = _NetcalIso(); m.fit(conf2d, y)
        else:
            raise ValueError(method)

        def apply(c):
            c = np.asarray(c, dtype=float)
            c2d = np.stack([1 - c, c], axis=1)
            return np.asarray(m.transform(c2d)).ravel()
        return apply
    else:
        return {
            "temperature": _fit_temperature_np,
            "platt": _fit_platt_np,
            "isotonic": _fit_isotonic_np,
        }[method](conf, correct)


def kfold_calibration(confidence_pos, correct, k: int = 5, seed: int = 42):
    """
    Return {method: {'ece','brier','oos_pred'}} evaluated out-of-sample via K-fold.
    `oos_pred` is the pooled out-of-fold calibrated probability for every item,
    aligned to the input order, so it can be re-scored or plotted downstream.
    """
    confidence_pos = np.asarray(confidence_pos, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(confidence_pos)
    methods = ["temperature", "platt", "isotonic"]

    if n < k * 3:
        return {m: {"ece": None, "brier": None, "oos_pred": None,
                    "note": "insufficient data for k-fold"} for m in methods}

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    oos = {m: np.zeros(n) for m in methods}

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        ctr, ytr = confidence_pos[train_idx], correct[train_idx]
        for m in methods:
            apply = _fit_method(m, ctr, ytr)
            oos[m][test_idx] = apply(confidence_pos[test_idx])

    out = {}
    for m in methods:
        out[m] = {
            "ece": expected_calibration_error(oos[m], correct),
            "brier": brier_score(oos[m], correct),
            "oos_pred": oos[m],
            "note": f"{k}-fold out-of-sample, n={n}",
        }
    return out


def backend_name() -> str:
    return "netcal" if _HAS_NETCAL else "numpy/scipy/sklearn fallback"
