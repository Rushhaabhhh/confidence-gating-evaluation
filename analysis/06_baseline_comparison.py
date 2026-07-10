#!/usr/bin/env python3
"""
Baseline comparison: verbal confidence vs. self-consistency.
Compares oversight lift and AUC-RC for each uncertainty estimator per domain.

Usage:
  python analysis/06_baseline_comparison.py
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from calibrated_oversight import metrics as M

FIG = ROOT / "paper" / "figures"
RES = ROOT / "analysis" / "results"
DOMAIN_SHORT = {"swebench":"SWE-bench","gsm8k":"GSM8K","mmlu":"MMLU","truthfulqa":"TruthfulQA"}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    sc_path = ROOT / "experiments/results/results_llama33_70b_selfconsistency.json"
    if not sc_path.exists():
        print("NOTE: self-consistency results not found — run Tier 3 experiment first.")
        print("  Command:")
        print("  python experiments/run_self_consistency.py --backend groq \\")
        print("    --domains swebench gsm8k \\")
        print("    --results-file experiments/results/results_llama33_70b_selfconsistency.json")
        return

    sc_data = json.load(open(sc_path))
    sc_valid = [r for r in sc_data if not r.get("parse_error")]

    # Group by domain
    domains_present = sorted({r["domain"] for r in sc_valid})
    rows = []
    for domain in domains_present:
        items = [r for r in sc_valid if r["domain"] == domain]
        verbal_conf = np.array([r["verbal_confidence_in_own"] for r in items], float)
        sc_conf     = np.array([r["self_consistency_confidence"] for r in items], float)
        correct     = np.array([int(r["prediction"] == r["label"]) for r in items], float)

        verbal_lift = M.oversight_precision(verbal_conf, correct, 0.10)["lift"]
        sc_lift     = M.oversight_precision(sc_conf,     correct, 0.10)["lift"]
        verbal_auc  = M.auc_risk_coverage(verbal_conf, correct)
        sc_auc      = M.auc_risk_coverage(sc_conf,     correct)
        verbal_sigma= M.confidence_collapse_stats(verbal_conf)["std"]
        sc_sigma    = M.confidence_collapse_stats(sc_conf)["std"]

        rows.append({
            "domain": domain, "n": len(items),
            "verbal_lift": verbal_lift, "sc_lift": sc_lift,
            "verbal_auc": verbal_auc,   "sc_auc": sc_auc,
            "verbal_sigma": verbal_sigma,"sc_sigma": sc_sigma,
        })

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Uncertainty Estimator Comparison: Verbal Confidence vs. Self-Consistency")
    print(f"{'='*70}")
    print(f"  {'Domain':<12} {'Verbal Lift':<13} {'SC Lift':<10} {'Verbal AUC':<12} {'SC AUC':<10}")
    print("-"*70)
    for r in rows:
        print(f"  {DOMAIN_SHORT[r['domain']]:<12} {r['verbal_lift']:<13.2f} "
              f"{r['sc_lift']:<10.2f} {r['verbal_auc']:<12.3f} {r['sc_auc']:<10.3f}")
    print(f"{'='*70}")

    # ── Figure ────────────────────────────────────────────────────────────────
    n = len(rows)
    x = np.arange(n)
    w = 0.35
    domain_labels = [DOMAIN_SHORT[r["domain"]] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.bar(x - w/2, [r["verbal_lift"] for r in rows], w,
           color="#1f5c99", alpha=0.85, label="Verbal confidence")
    ax.bar(x + w/2, [r["sc_lift"] for r in rows], w,
           color="#2e8b57", alpha=0.85, label="Self-consistency")
    ax.axhline(1.0, ls="--", color="red", lw=1.5, label="Lift=1 (no signal)")
    ax.set_xticks(x); ax.set_xticklabels(domain_labels, fontsize=10)
    ax.set_ylabel("Oversight Lift", fontsize=10)
    ax.set_title("Oversight Lift\nVerbal Confidence vs. Self-Consistency", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - w/2, [r["verbal_auc"] for r in rows], w,
           color="#1f5c99", alpha=0.85, label="Verbal confidence")
    ax.bar(x + w/2, [r["sc_auc"] for r in rows], w,
           color="#2e8b57", alpha=0.85, label="Self-consistency")
    ax.set_xticks(x); ax.set_xticklabels(domain_labels, fontsize=10)
    ax.set_ylabel("AUC-RC", fontsize=10)
    ax.set_title("AUC Risk-Coverage\nVerbal Confidence vs. Self-Consistency", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Uncertainty Estimator Comparison (Llama-3.3-70B)\n"
                 "Verbal confidence vs. self-consistency (5-sample, T=0.3)", fontsize=11)
    fig.tight_layout()
    out = FIG / "fig7_baseline_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure: {out}")

    out_json = RES / "baseline_comparison_results.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"  Results: {out_json}")


if __name__ == "__main__":
    main()
