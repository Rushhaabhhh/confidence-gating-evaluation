"""
Core calibration and selective-prediction metrics.

Everything numeric in the paper flows through this module so that every script
uses one identical, tested definition of each metric. No metric is re-implemented
anywhere else in the codebase.

Definitions follow:
  - ECE:   Guo et al. (2017), "On Calibration of Modern Neural Networks"
  - Brier: Brier (1950); a strictly proper scoring rule (Gneiting & Raftery 2007)
  - Risk-coverage / selective prediction: El-Yaniv & Wiener (2010)
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Calibration error metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    """
    Expected Calibration Error with equal-width bins.

    confidences : array of p(prediction correct), in [0, 1]
    correct     : array of 0/1 outcomes (1 = prediction was correct)
    Returns the weighted average gap between confidence and accuracy per bin.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(confidences)
    if n == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def maximum_calibration_error(confidences, correct, n_bins: int = 10) -> float:
    """Maximum gap between confidence and accuracy across bins (worst-case)."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidences) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        mce = max(mce, abs(correct[mask].mean() - confidences[mask].mean()))
    return float(mce)


def brier_score(confidences, correct) -> float:
    """Mean squared error between stated confidence and outcome. 0 = perfect."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidences) == 0:
        return float("nan")
    return float(np.mean((confidences - correct) ** 2))


def reliability_bins(confidences, correct, n_bins: int = 10):
    """Return per-bin (mean_confidence, accuracy, count) for reliability diagrams."""
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            out.append((float((lo + hi) / 2), None, 0))
        else:
            out.append((float(confidences[mask].mean()), float(correct[mask].mean()), int(mask.sum())))
    return out


# ---------------------------------------------------------------------------
# Confidence distribution descriptors (the "confidence collapse" finding)
# ---------------------------------------------------------------------------

def sharpness(confidences) -> float:
    """
    Mean distance of confidence from 0.5. A model that says 0.8 on everything is
    'sharp' (high) but may still be badly calibrated -- sharpness and calibration
    are independent, which is exactly the point the paper makes.
    """
    confidences = np.asarray(confidences, dtype=float)
    return float(np.mean(np.abs(confidences - 0.5)))


def confidence_collapse_stats(confidence_in_own_answer):
    """
    Descriptors that quantify 'confidence collapse': a model whose stated
    confidence occupies a very narrow band, so a fixed threshold cannot separate
    confident from unconfident items.
    """
    c = np.asarray(confidence_in_own_answer, dtype=float)
    return {
        "mean": float(np.mean(c)),
        "std": float(np.std(c)),
        "min": float(np.min(c)),
        "max": float(np.max(c)),
        "iqr": float(np.percentile(c, 75) - np.percentile(c, 25)),
        "collapsed": bool(np.std(c) < 0.10),  # named failure mode threshold
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(values, statistic_fn, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    """
    Generic percentile bootstrap 95% CI for any statistic of a 1-D array.

    values       : array the statistic is computed over
    statistic_fn : callable(resampled_array) -> float
    """
    values = np.asarray(values)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(values))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[rng.choice(idx, size=len(idx), replace=True)]
        boots[b] = statistic_fn(sample)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (lo, hi)


def bootstrap_ci_paired(confidences, correct, statistic_fn, n_boot: int = 2000, seed: int = 42):
    """
    Bootstrap CI for metrics that need paired (confidence, correct) arrays,
    e.g. ECE and Brier. Resamples item indices, keeping pairs intact.
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(confidences)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        boots[b] = statistic_fn(confidences[s], correct[s])
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


# ---------------------------------------------------------------------------
# Selective prediction / risk-coverage
# ---------------------------------------------------------------------------

def risk_coverage_curve(confidence_in_own_answer, correct):
    """
    Sweep a confidence threshold; at each level, 'cover' (act autonomously on)
    the items at or above it and report accuracy on that covered set.

    Returns dict with parallel arrays: coverage, accuracy, threshold.
    This is the governance figure: coverage = fraction handled without a human,
    accuracy = reliability of what was handled autonomously.
    """
    c = np.asarray(confidence_in_own_answer, dtype=float)
    y = np.asarray(correct, dtype=float)
    order = np.argsort(-c)  # most confident first
    c_sorted = c[order]
    y_sorted = y[order]

    coverages, accuracies, thresholds = [], [], []
    for k in range(1, len(c_sorted) + 1):
        coverages.append(k / len(c_sorted))
        accuracies.append(float(y_sorted[:k].mean()))
        thresholds.append(float(c_sorted[k - 1]))
    return {
        "coverage": coverages,
        "accuracy": accuracies,
        "threshold": thresholds,
    }


def auc_risk_coverage(confidence_in_own_answer, correct) -> float:
    """
    Area under the risk-coverage (accuracy-vs-coverage) curve. Higher is better:
    it means the most-confident items really are the most-accurate. A value near
    baseline accuracy means confidence carries no selective signal.
    """
    rc = risk_coverage_curve(confidence_in_own_answer, correct)
    cov = np.asarray(rc["coverage"])
    acc = np.asarray(rc["accuracy"])
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 renamed trapz
    return float(trapz(acc, cov))


def selective_accuracy_gain(confidence_in_own_answer, correct, coverage_level: float = 0.7):
    """
    Accuracy on the most-confident `coverage_level` fraction of items, minus the
    overall baseline accuracy. The headline governance number:
    'covering the top X% by confidence yields +Y accuracy'.
    """
    c = np.asarray(confidence_in_own_answer, dtype=float)
    y = np.asarray(correct, dtype=float)
    baseline = float(y.mean())
    order = np.argsort(-c)
    k = max(1, int(round(len(c) * coverage_level)))
    covered_acc = float(y[order[:k]].mean())
    return {
        "coverage_level": coverage_level,
        "covered_accuracy": covered_acc,
        "baseline_accuracy": baseline,
        "gain": covered_acc - baseline,
        "n_covered": k,
    }


def oversight_precision(confidence_in_own_answer, correct, flag_fraction: float = 0.10):
    """
    If we flag the bottom `flag_fraction` least-confident items for human review,
    what fraction of those flagged items were actually wrong? Compared against the
    base error rate, this says whether low confidence actually finds errors.

    Uses rank-based selection (bottom-k by sorted position), which is robust to
    ties -- models that answer in round numbers (0.8, 0.9) produce heavy ties that
    break naive percentile-threshold selection.
    """
    c = np.asarray(confidence_in_own_answer, dtype=float)
    y = np.asarray(correct, dtype=float)
    n = len(c)
    k = max(1, int(round(n * flag_fraction)))
    order = np.argsort(c)  # least confident first
    flagged = order[:k]
    unflagged = order[k:]

    base_error_rate = float(1 - y.mean())
    flagged_error_rate = float(1 - y[flagged].mean()) if len(flagged) else float("nan")
    unflagged_error_rate = float(1 - y[unflagged].mean()) if len(unflagged) else float("nan")

    return {
        "flag_fraction": flag_fraction,
        "n_flagged": int(k),
        "base_error_rate": base_error_rate,
        "flagged_error_rate": flagged_error_rate,        # want this HIGH (flags find errors)
        "unflagged_error_rate": unflagged_error_rate,    # want this LOW (kept items are reliable)
        "lift": (flagged_error_rate / base_error_rate) if base_error_rate > 0 else float("nan"),
    }
