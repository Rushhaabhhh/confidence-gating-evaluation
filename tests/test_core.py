#!/usr/bin/env python3
"""
Unit tests for the core library. Run with:  python tests/test_core.py
Uses only assert + synthetic data (no pytest dependency required).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibrated_oversight import metrics as M
from calibrated_oversight.audit_log import AuditLog


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_ece_perfect_calibration():
    # If confidence exactly equals accuracy in every bin, ECE = 0.
    conf = np.array([0.0, 0.0, 1.0, 1.0])
    correct = np.array([0, 0, 1, 1])
    assert approx(M.expected_calibration_error(conf, correct), 0.0), "perfect ECE should be 0"


def test_ece_worst_case():
    # 100% confident, always wrong -> ECE = 1.
    conf = np.array([1.0, 1.0, 1.0])
    correct = np.array([0, 0, 0])
    assert approx(M.expected_calibration_error(conf, correct), 1.0), "worst ECE should be 1"


def test_brier_bounds():
    conf = np.array([1.0, 0.0])
    correct = np.array([1, 0])
    assert approx(M.brier_score(conf, correct), 0.0), "perfect Brier should be 0"
    assert approx(M.brier_score(np.array([1.0]), np.array([0])), 1.0), "worst Brier should be 1"


def test_sharpness():
    # all-0.5 confidence -> sharpness 0; all-1.0 -> sharpness 0.5
    assert approx(M.sharpness(np.array([0.5, 0.5])), 0.0)
    assert approx(M.sharpness(np.array([1.0, 1.0])), 0.5)


def test_collapse_detection():
    narrow = np.full(100, 0.8)
    assert M.confidence_collapse_stats(narrow)["collapsed"] is True
    wide = np.linspace(0.5, 1.0, 100)
    assert M.confidence_collapse_stats(wide)["collapsed"] is False


def test_auc_rc_perfect_ranking():
    # If confidence perfectly ranks correctness, most-confident-first accuracy
    # starts at 1.0 -> AUC-RC should be high (> accuracy).
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([1, 1, 1, 0, 0])   # top-3 correct
    auc = M.auc_risk_coverage(conf, correct)
    assert auc > correct.mean(), "good ranking should beat baseline"


def test_oversight_precision_lift():
    # Least-confident items are all wrong -> lift should be > 1.
    conf = np.array([0.5, 0.55, 0.9, 0.95, 0.99] * 4)   # n=20
    correct = np.array([0, 0, 1, 1, 1] * 4)
    op = M.oversight_precision(conf, correct, flag_fraction=0.20)
    assert op["lift"] > 1.0, "low-confidence-wrong pattern should give lift>1"


def test_bootstrap_ci_contains_point():
    vals = np.array([1, 0, 1, 1, 0, 1, 0, 1] * 5, dtype=float)
    lo, hi = M.bootstrap_ci(vals, lambda a: a.mean())
    assert lo <= vals.mean() <= hi, "CI should contain the point estimate"


def test_audit_log_intact_verifies():
    log = AuditLog(b"key", agent_id="t")
    for i in range(10):
        log.append("act", f"input-{i}", i % 2, 0.8, False, timestamp=float(i))
    res = log.verify()
    assert res["valid"], "intact chain must verify"


def test_audit_log_detects_tamper():
    log = AuditLog(b"key", agent_id="t")
    for i in range(10):
        log.append("act", f"input-{i}", i % 2, 0.8, False, timestamp=float(i))
    log.entries[5].confidence = 0.1  # tamper without re-signing
    res = log.verify()
    assert not res["valid"], "tampered chain must fail"
    assert res["broken_at"] == 5, "must localize the tamper to the right entry"


def test_audit_log_roundtrip():
    log = AuditLog(b"key", agent_id="t")
    for i in range(5):
        log.append("act", f"in-{i}", 1, 0.9, False, timestamp=float(i))
    restored = AuditLog.from_json(log.to_json(), b"key", agent_id="t")
    assert restored.verify()["valid"], "serialized+restored chain must verify"


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
