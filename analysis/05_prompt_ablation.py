#!/usr/bin/env python3
"""
Prompt ablation analysis: default vs. fewshot elicitation.
Compares confidence_std (sigma), ECE, and oversight lift across prompt variants.

Usage:
  python analysis/05_prompt_ablation.py
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from calibrated_oversight.data import load_results_by_domain, DOMAIN_DISPLAY
from calibrated_oversight import metrics as M

FIG = ROOT / "paper" / "figures"
RES = ROOT / "analysis" / "results"
DOMAINS = ["swebench", "gsm8k", "mmlu", "truthfulqa"]
DOMAIN_SHORT = {"swebench":"SWE-bench","gsm8k":"GSM8K","mmlu":"MMLU","truthfulqa":"TruthfulQA"}


def metrics_for(r):
    coll = M.confidence_collapse_stats(r.confidence_in_own)
    ece  = M.expected_calibration_error(r.confidence_pos, r.correct)
    op   = M.oversight_precision(r.confidence_in_own, r.correct, 0.10)
    return {"sigma": coll["std"], "ece": ece, "lift": op["lift"],
            "collapsed": coll["collapsed"], "n": r.n_valid}


def load_variant(path):
    if not path.exists():
        return None
    return load_results_by_domain(path)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    default_path = ROOT / "experiments/results/results_llama33_70b_multidomain.json"
    fewshot_path = ROOT / "experiments/results/results_llama33_70b_multidomain_fewshot.json"

    default_dom = load_variant(default_path)
    fewshot_dom = load_variant(fewshot_path)

    if default_dom is None:
        print("ERROR: default results file not found."); return
    if fewshot_dom is None:
        print("NOTE: fewshot results file not found — run Tier 2 experiment first.")
        print("  Command:")
        print("  python experiments/run_elicitation.py --backend groq \\")
        print("    --prompt-variant fewshot \\")
        print("    --results-file experiments/results/results_llama33_70b_multidomain_fewshot.json")
        return

    rows = []
    for domain in DOMAINS:
        r_def = default_dom.get(domain)
        r_few = fewshot_dom.get(domain)
        if r_def is None or r_few is None:
            continue
        m_def = metrics_for(r_def)
        m_few = metrics_for(r_few)
        rows.append({
            "domain": domain,
            "default": m_def, "fewshot": m_few,
            "delta_sigma": m_few["sigma"] - m_def["sigma"],
            "delta_lift":  m_few["lift"]  - m_def["lift"],
            "delta_ece":   m_few["ece"]   - m_def["ece"],
        })

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Prompt Ablation: Default vs. Fewshot (Llama-3.3-70B)")
    print(f"{'='*70}")
    print(f"  {'Domain':<12} {'σ (def)':<10} {'σ (few)':<10} {'Δσ':<8}"
          f" {'ECE(d)':<8} {'ECE(f)':<8} {'Lift(d)':<9} {'Lift(f)':<9}")
    print("-"*70)
    for row in rows:
        d = row["domain"]
        md, mf = row["default"], row["fewshot"]
        collapse_flag = " ⚠" if md["collapsed"] and mf["collapsed"] else \
                        " ✓" if not mf["collapsed"] else ""
        print(f"  {DOMAIN_SHORT[d]:<12} {md['sigma']:<10.3f} {mf['sigma']:<10.3f} {row['delta_sigma']:+8.3f}"
              f" {md['ece']:<8.3f} {mf['ece']:<8.3f} {md['lift']:<9.2f} {mf['lift']:<9.2f}{collapse_flag}")
    print(f"{'='*70}")
    print("  ⚠ = collapse persists under both prompts (σ < 0.10)")
    print("  ✓ = collapse resolved under fewshot prompt")

    # ── Figure: sigma and lift side-by-side ──────────────────────────────────
    domain_labels = [DOMAIN_SHORT[r["domain"]] for r in rows]
    x = np.arange(len(rows))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: sigma
    ax = axes[0]
    ax.bar(x - width/2, [r["default"]["sigma"] for r in rows], width,
           color="#1f5c99", alpha=0.85, label="Default prompt")
    ax.bar(x + width/2, [r["fewshot"]["sigma"] for r in rows], width,
           color="#e07b39", alpha=0.85, label="Fewshot prompt")
    ax.axhline(0.10, ls="--", color="red", lw=1.5, label="Collapse threshold (σ=0.10)")
    ax.set_xticks(x); ax.set_xticklabels(domain_labels, fontsize=10)
    ax.set_ylabel("Confidence σ (std)", fontsize=10)
    ax.set_title("Confidence Spread (σ)\nDoes prompt format affect collapse?", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # Panel 2: lift
    ax = axes[1]
    ax.bar(x - width/2, [r["default"]["lift"] for r in rows], width,
           color="#1f5c99", alpha=0.85, label="Default prompt")
    ax.bar(x + width/2, [r["fewshot"]["lift"] for r in rows], width,
           color="#e07b39", alpha=0.85, label="Fewshot prompt")
    ax.axhline(1.0, ls="--", color="red", lw=1.5, label="Lift = 1 (no signal)")
    ax.set_xticks(x); ax.set_xticklabels(domain_labels, fontsize=10)
    ax.set_ylabel("Oversight Lift", fontsize=10)
    ax.set_title("Oversight Lift\nDoes prompt format change oversight value?", fontsize=10)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Prompt Sensitivity Ablation: Default vs. Fewshot Elicitation\n"
                 "(Llama-3.3-70B, 250 items/domain)", fontsize=11)
    fig.tight_layout()
    out = FIG / "fig6_prompt_ablation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure: {out}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_json = RES / "prompt_ablation_results.json"
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"  Results: {out_json}")


if __name__ == "__main__":
    main()
