#!/usr/bin/env python3
"""
Significance tests for key cross-domain and cross-model comparisons.

Reporting order for every test: effect size → 95% CI → p-value.
e.g. "Oversight lift increased from 1.03 to 2.86 (Δ = 1.83, 95% CI [1.45, 2.21], p = 0.004)"

Tests:
  1. Permutation test — is lift(GSM8K) > lift(SWE-bench) for Llama-70B?
  2. Permutation test — is lift(MMLU) > lift(SWE-bench)?
  3. Permutation test — is lift(TruthfulQA) > lift(SWE-bench)?
  4. DeLong test      — is AUC-RC(GSM8K) > AUC-RC(SWE-bench)?
  5. McNemar test     — Llama-70B vs. Gemini on matched SWE-bench items.
  6. Bootstrap test   — is ECE reduction (raw→best post-hoc) significant?

Usage:
  python analysis/04_significance_tests.py
  python analysis/04_significance_tests.py --domain-pair gsm8k swebench
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from calibrated_oversight.data import load_results_by_domain, DOMAIN_DISPLAY
from calibrated_oversight import metrics as M

RES = ROOT / "analysis" / "results"

# ── Helpers ───────────────────────────────────────────────────────────────────

def permutation_lift_test(conf_a, correct_a, conf_b, correct_b,
                          n_perm: int = 10_000, seed: int = 42):
    """
    One-sided permutation test: is lift(a) > lift(b)?

    Under H0, the domain labels are interchangeable — we pool items, randomly
    split into two groups of the same sizes, and compute the lift difference.
    The p-value is the fraction of permutations where the null difference
    ≥ the observed difference.
    """
    rng = np.random.default_rng(seed)
    conf_a = np.asarray(conf_a, float)
    correct_a = np.asarray(correct_a, float)
    conf_b = np.asarray(conf_b, float)
    correct_b = np.asarray(correct_b, float)

    def lift(conf, correct):
        return M.oversight_precision(conf, correct, 0.10)["lift"]

    obs_a = lift(conf_a, correct_a)
    obs_b = lift(conf_b, correct_b)
    obs_delta = obs_a - obs_b

    n_a = len(conf_a)
    conf_pool = np.concatenate([conf_a, conf_b])
    correct_pool = np.concatenate([correct_a, correct_b])

    null_deltas = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(conf_pool))
        perm_a = idx[:n_a]
        perm_b = idx[n_a:]
        null_deltas[i] = lift(conf_pool[perm_a], correct_pool[perm_a]) - \
                          lift(conf_pool[perm_b], correct_pool[perm_b])

    p_value = float((null_deltas >= obs_delta).mean())

    # Bootstrap CI for the observed delta
    def delta_fn(idx):
        n = len(idx)
        split = n // 2
        la = lift(conf_pool[idx[:split]], correct_pool[idx[:split]])
        lb = lift(conf_pool[idx[split:]], correct_pool[idx[split:]])
        return la - lb

    boots = np.empty(2000)
    idx_full = np.arange(len(conf_pool))
    for b in range(2000):
        s = rng.choice(idx_full, size=len(idx_full), replace=True)
        boots[b] = delta_fn(s)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    return {
        "lift_a": float(obs_a), "lift_b": float(obs_b),
        "delta": float(obs_delta), "ci95": ci,
        "p_value": p_value, "n_perm": n_perm,
    }


def delong_auc_test(conf_a, correct_a, conf_b, correct_b, seed: int = 42):
    """
    Bootstrap test for AUC-RC difference (one-sided: a > b).
    DeLong (1988) for ROC AUC; we adapt for risk-coverage AUC.
    """
    rng = np.random.default_rng(seed)
    conf_a = np.asarray(conf_a, float)
    correct_a = np.asarray(correct_a, float)
    conf_b = np.asarray(conf_b, float)
    correct_b = np.asarray(correct_b, float)

    obs_a = M.auc_risk_coverage(conf_a, correct_a)
    obs_b = M.auc_risk_coverage(conf_b, correct_b)
    obs_delta = obs_a - obs_b

    # Bootstrap CI for delta
    boots = np.empty(2000)
    na, nb = len(conf_a), len(conf_b)
    for i in range(2000):
        sa = rng.integers(0, na, na)
        sb = rng.integers(0, nb, nb)
        boots[i] = M.auc_risk_coverage(conf_a[sa], correct_a[sa]) - \
                   M.auc_risk_coverage(conf_b[sb], correct_b[sb])

    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    # One-sided p: fraction of bootstrap samples with delta ≤ 0
    p_value = float((boots <= 0).mean())

    return {
        "auc_a": float(obs_a), "auc_b": float(obs_b),
        "delta": float(obs_delta), "ci95": ci,
        "p_value": p_value,
    }


def mcnemar_test(correct_a, correct_b):
    """
    McNemar's test for paired binary outcomes (same items, two models).
    Returns p-value. Small p → accuracy difference is significant.
    """
    correct_a = np.asarray(correct_a, int)
    correct_b = np.asarray(correct_b, int)
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    if b + c == 0:
        return 1.0
    from scipy.stats import chi2
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    return float(1 - chi2.cdf(stat, df=1))


def bootstrap_ece_reduction_test(conf, correct, best_ece, seed: int = 42, n_boot: int = 2000):
    """
    Is the ECE reduction (raw → best post-hoc) statistically significant?
    Uses percentile bootstrap on the raw ECE CI, then checks whether
    best_ece falls outside the CI.
    """
    raw_ece = M.expected_calibration_error(conf, correct)
    ci = M.bootstrap_ci_paired(conf, correct, M.expected_calibration_error,
                                n_boot=n_boot, seed=seed)
    reduction = raw_ece - best_ece
    reduction_pct = reduction / raw_ece * 100 if raw_ece > 0 else float("nan")
    # If best_ece is below the lower CI bound, reduction is significant
    significant = best_ece < ci[0]
    return {
        "raw_ece": float(raw_ece), "best_ece": float(best_ece),
        "reduction": float(reduction), "reduction_pct": float(reduction_pct),
        "raw_ece_ci95": ci, "significant": significant,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def sep(msg):
    print(f"\n{'─'*70}\n  {msg}\n{'─'*70}")


def main():
    RES.mkdir(parents=True, exist_ok=True)

    # Load multi-domain files
    llama_path  = ROOT / "experiments/results/results_llama33_70b_multidomain.json"
    gemini_path = ROOT / "experiments/results/results_gemini25_flash_multidomain.json"

    llama_dom  = load_results_by_domain(llama_path)
    gemini_dom = load_results_by_domain(gemini_path)

    all_results = {}

    # ── Tests 1–3: Llama-70B lift comparisons ─────────────────────────────────
    r_swe   = llama_dom["swebench"]
    r_gsm   = llama_dom["gsm8k"]
    r_mmlu  = llama_dom["mmlu"]
    r_tqa   = llama_dom["truthfulqa"]

    for name, r_domain in [("GSM8K", r_gsm), ("MMLU", r_mmlu), ("TruthfulQA", r_tqa)]:
        sep(f"Lift: Llama-70B {name} vs. SWE-bench (permutation, one-sided)")
        t = permutation_lift_test(
            r_domain.confidence_in_own, r_domain.correct,
            r_swe.confidence_in_own, r_swe.correct
        )
        delta = t["delta"]
        ci = t["ci95"]
        p = t["p_value"]
        print(f"  lift({name})={t['lift_a']:.2f}, lift(SWE)={t['lift_b']:.2f}")
        print(f"  Δ = {delta:+.2f}  95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  p = {p:.4f}")
        print(f"  → {'SIGNIFICANT' if p < 0.05 else 'not significant'} (one-sided, α=0.05)")
        all_results[f"lift_llama70b_{name.lower()}_vs_swebench"] = t

    # ── Test 4: AUC-RC comparison ──────────────────────────────────────────────
    sep("AUC-RC: Llama-70B GSM8K vs. SWE-bench (bootstrap, one-sided)")
    t = delong_auc_test(
        r_gsm.confidence_in_own, r_gsm.correct,
        r_swe.confidence_in_own, r_swe.correct
    )
    print(f"  AUC-RC(GSM8K)={t['auc_a']:.3f}, AUC-RC(SWE)={t['auc_b']:.3f}")
    print(f"  Δ = {t['delta']:+.3f}  95% CI [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]  p = {t['p_value']:.4f}")
    all_results["auc_rc_llama70b_gsm8k_vs_swebench"] = t

    # ── Test 5: McNemar — Llama-70B vs. Gemini on SWE-bench ──────────────────
    sep("McNemar: Llama-70B vs. Gemini accuracy on SWE-bench")
    g_swe = gemini_dom["swebench"]
    # Match on instance_id
    llama_ids  = {r["instance_id"]: r for r in r_swe.records if not r.get("parse_error")}
    gemini_ids = {r["instance_id"]: r for r in g_swe.records if not r.get("parse_error")}
    common = sorted(set(llama_ids) & set(gemini_ids))
    if len(common) >= 10:
        ca = np.array([llama_ids[i]["prediction"] == llama_ids[i]["label"] for i in common], int)
        cb = np.array([gemini_ids[i]["prediction"] == gemini_ids[i]["label"] for i in common], int)
        acc_a, acc_b = ca.mean(), cb.mean()
        p = mcnemar_test(ca, cb)
        print(f"  Matched items: {len(common)}")
        print(f"  Acc(Llama)={acc_a:.3f}, Acc(Gemini)={acc_b:.3f}, Δ={acc_a-acc_b:+.3f}, p={p:.4f}")
        all_results["mcnemar_llama70b_vs_gemini_swebench"] = {
            "n_matched": len(common), "acc_llama": float(acc_a),
            "acc_gemini": float(acc_b), "delta": float(acc_a - acc_b), "p_value": p,
        }
    else:
        print(f"  Only {len(common)} matched items — skipping (need ≥ 10)")

    # ── Test 6: ECE reduction significance ────────────────────────────────────
    sep("ECE reduction significance (bootstrap) — key conditions")
    from calibrated_oversight.calibration import kfold_calibration

    ece_red_results = {}
    for label, path in [("Llama-70B", llama_path), ("Gemini", gemini_path)]:
        dom_data = load_results_by_domain(path)
        for domain, r in sorted(dom_data.items()):
            cal = kfold_calibration(r.confidence_pos, r.correct)
            best_method = min(
                (m for m in cal if cal[m]["ece"] is not None),
                key=lambda m: cal[m]["ece"], default=None
            )
            if best_method is None:
                continue
            best_ece = cal[best_method]["ece"]
            t = bootstrap_ece_reduction_test(r.confidence_pos, r.correct, best_ece)
            key = f"{label}_{domain}"
            ece_red_results[key] = {**t, "method": best_method}
            sig = "✓ significant" if t["significant"] else "— not significant"
            print(f"  {label:8s} / {domain:12s}: "
                  f"raw={t['raw_ece']:.3f} → best={t['best_ece']:.3f} "
                  f"(−{t['reduction_pct']:.0f}%)  {sig}")

    all_results["ece_reduction"] = ece_red_results

    # ── Save ──────────────────────────────────────────────────────────────────
    out = RES / "significance_tests.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n  → {out}")

    # ── Summary for paper ─────────────────────────────────────────────────────
    sep("SUMMARY — paste into paper §5")
    for domain_name, key_suffix in [("GSM8K", "gsm8k"), ("MMLU", "mmlu"), ("TruthfulQA", "truthfulqa")]:
        t = all_results.get(f"lift_llama70b_{key_suffix}_vs_swebench", {})
        if t:
            print(f"  Lift {domain_name} vs. SWE-bench: Δ={t['delta']:+.2f} "
                  f"95%CI [{t['ci95'][0]:+.2f},{t['ci95'][1]:+.2f}] "
                  f"p={t['p_value']:.4f}")


if __name__ == "__main__":
    main()
