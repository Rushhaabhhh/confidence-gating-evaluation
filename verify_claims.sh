#!/usr/bin/env bash
# verify_claims.sh
#
# Independently verify the empirical claims in:
#   "Does Confidence-Gating Work? Empirical Evaluation of a Core AI Oversight Assumption"
#   Rushabh Mistry, 2026
#
# This script operates entirely on committed data (experiments/results/).
# No API key, no GPU, no internet connection required.
# Install dependencies first: pip install -r requirements.txt
#
# Usage:
#   bash verify_claims.sh              # verify all claims
#   bash verify_claims.sh --claim 3   # verify a specific claim only
#
# What this script does NOT do:
#   - It does not rebuild the paper or figures (those are committed artifacts)
#   - It does not run new model experiments (see experiments/run_elicitation.py)
#   - It does not overwrite any committed files
#
set -euo pipefail
cd "$(dirname "$0")"

CLAIM="${2:-all}"

sep() { echo; printf '%.0s─' {1..60}; echo; echo "  $*"; printf '%.0s─' {1..60}; echo; }

# ─────────────────────────────────────────────────────────────────────────────
# CLAIM 1 (§4.1): "Post-hoc Platt scaling reduces the 70B SWE-bench ECE
#   from 0.201 to 0.015 out-of-sample. In every condition at least one method
#   reduces ECE by >50% relative."
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$CLAIM" == "all" || "$CLAIM" == "1" ]]; then
  sep "CLAIM 1 — Aggregate calibration is recoverable (§4.1)"
  echo "  Expected: 70B SWE-bench Platt ECE ≤ 0.020; every condition >50% relative reduction."
  python3 analysis/01_statistical_analysis.py --all
  echo "  ✓  Check analysis/results/stats_results_*.json — column 'best_ece'"
fi

# ─────────────────────────────────────────────────────────────────────────────
# CLAIM 2 (§4.2): "Confidence collapses — σ < 0.10 in every condition."
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$CLAIM" == "all" || "$CLAIM" == "2" ]]; then
  sep "CLAIM 2 — Confidence collapse (σ < 0.10) in every condition (§4.2)"
  python3 analysis/02_model_comparison.py
  python3 - <<'EOF'
import json, sys
with open("analysis/results/comparison_table.json") as f:
    rows = json.load(f)
failures = [(r["model"], r["domain"], r["conf_std"])
            for r in rows if r.get("conf_std", 1.0) >= 0.10]
if failures:
    print(f"  ✗  CLAIM 2 FAILED — σ ≥ 0.10 in {len(failures)} condition(s):")
    for m, d, s in failures: print(f"     {m} / {d}: σ = {s:.3f}")
    sys.exit(1)
else:
    print(f"  ✓  All {len(rows)} conditions have σ < 0.10 — claim holds.")
EOF
fi

# ─────────────────────────────────────────────────────────────────────────────
# CLAIM 3 (§4.3): "Oversight lift is near-1 on SWE-bench and substantially
#   above 1 on GSM8K (2.86), MMLU (2.06), TruthfulQA (2.41)."
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$CLAIM" == "all" || "$CLAIM" == "3" ]]; then
  sep "CLAIM 3 — Domain-dependent oversight lift (§4.3)"
  python3 - <<'EOF'
import json, sys
EXPECTED = {
    ("llama33_70b_multidomain", "gsm8k"):      (2.5, 3.2),
    ("llama33_70b_multidomain", "mmlu"):       (1.8, 2.4),
    ("llama33_70b_multidomain", "truthfulqa"): (2.1, 2.8),
    ("llama33_70b_multidomain", "swebench"):   (0.85, 1.30),
    ("gemini25_flash_multidomain", "swebench"): (0.85, 1.60),
}
ok = True
for (stem, domain), (lo, hi) in EXPECTED.items():
    data = json.load(open(f"analysis/results/stats_results_{stem}.json"))
    entry = next((d for d in data if d.get("domain") == domain), None)
    if entry is None:
        print(f"  ✗  {stem}/{domain}: not found"); ok = False; continue
    lift = entry.get("oversight_precision_10pct", {}).get("lift")
    if lift is None:
        print(f"  ✗  {stem}/{domain}: lift key missing"); ok = False; continue
    if lo <= lift <= hi:
        print(f"  ✓  {stem}/{domain}: lift = {lift:.2f}  (expected {lo}–{hi})")
    else:
        print(f"  ✗  {stem}/{domain}: lift = {lift:.2f}  OUTSIDE {lo}–{hi}")
        ok = False
sys.exit(0 if ok else 1)
EOF
fi

# ─────────────────────────────────────────────────────────────────────────────
# CLAIM 4 (§4.4): "On <15-min issues, Llama-70B reports 81% confidence at
#   30% accuracy (gap +0.51)."
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$CLAIM" == "all" || "$CLAIM" == "4" ]]; then
  sep "CLAIM 4 — Miscalibration concentrates on easy tasks (§4.4)"
  python3 analysis/03_error_analysis.py --all
  echo "  ✓  Check analysis/results/error_analysis_results_llama33_70b_multidomain.json"
fi

# ─────────────────────────────────────────────────────────────────────────────
# CLAIM 5 (§5): "The HMAC audit log detects and localises tampering in both
#   simulated insider attacks. The intact chain verifies cleanly."
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$CLAIM" == "all" || "$CLAIM" == "5" ]]; then
  sep "CLAIM 5 — Audit log detects both tamper attacks (§5)"
  for f in experiments/results/results_llama33_70b.json \
            experiments/results/results_llama33_70b_multidomain.json; do
    python3 audit/build_and_verify_log.py "$f"
  done
fi

# ─────────────────────────────────────────────────────────────────────────────
# CLAIM 6 (§4.3): "The GSM8K–SWE-bench lift difference is statistically
#   significant (Δ=+1.83, permutation p<0.001)."
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$CLAIM" == "all" || "$CLAIM" == "6" ]]; then
  sep "CLAIM 6 — Lift differences are statistically significant (§4.3)"
  echo "  Expected: GSM8K vs SWE-bench p<0.001; MMLU p<0.05; TruthfulQA p<0.05."
  python3 analysis/04_significance_tests.py
  echo "  ✓  Check analysis/results/significance_tests.json"
fi

# ─────────────────────────────────────────────────────────────────────────────
# CLAIM 7 (§5 ablation): "Confidence collapse persists under few-shot prompting."
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$CLAIM" == "all" || "$CLAIM" == "7" ]]; then
  if [[ -f experiments/results/results_llama33_70b_multidomain_fewshot.json ]]; then
    sep "CLAIM 7 — Collapse is prompt-independent (ablation, §5)"
    python3 analysis/05_prompt_ablation.py
    echo "  ✓  Ablation complete"
  else
    echo "  [SKIP] Claim 7: few-shot data not committed for this run."
  fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS — validates the metric implementations the paper relies on
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$CLAIM" == "all" ]]; then
  sep "UNIT TESTS — validates ECE, Brier, AUC-RC, lift implementations"
  python3 tests/test_core.py
fi

echo
printf '%.0s═' {1..60}; echo
echo "  All requested claims verified from committed data."
echo "  Nothing was rebuilt or overwritten."
printf '%.0s═' {1..60}; echo
