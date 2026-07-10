# Does Confidence-Gating Work?
### Empirical Evaluation of a Core AI Oversight Assumption

**Paper:** [paper/paper.pdf](paper/paper.pdf) · **Preprint:** Zenodo (DOI pending)

---

## What this paper finds

Three 2026 governance frameworks (Singapore IMDA, U.S. NIST/NCCoE, EU AI Act Arts. 14–15) require that AI agents support meaningful human oversight. A widely assumed implementation is **confidence-gating**: the agent flags any decision below a confidence threshold for human review.

**We test whether that assumption holds across task types.** Two models, four domains, N = 2,004 decisions.

| Finding | Evidence |
|---|---|
| Confidence-gating **works** on structured QA | Lift 1.67–2.86 on GSM8K, MMLU, TruthfulQA |
| Confidence-gating **fails** on open-ended agent tasks | Lift ≈ 1.0 on SWE-bench for both models |
| **Confidence collapses** — σ < 0.06 everywhere | Fixed thresholds cannot separate correct from incorrect |
| Post-hoc calibration fixes aggregate ECE but **not** the gate | Platt: 70B ECE 0.201 → 0.015; lift unchanged |
| Self-consistency (3 samples) raises SWE-bench lift to **2.06** | Preliminary, n = 110 |
| Tamper-evident audit log detects both simulated insider attacks | Python standard library only |

**Implication:** Governance requirements that mandate confidence-gating without specifying task type risk failing on the open-ended judgments where agents most commonly operate.

---

## Verify the paper's claims

```bash
pip install -r requirements.txt
bash verify_claims.sh              # verify all 7 claims
bash verify_claims.sh --claim 3   # verify a specific claim
```

Runs entirely on committed data — no API key, no GPU, ~2 minutes. Does not rebuild the paper or overwrite any files.

---

## Repository structure

```
├── src/calibrated_oversight/       Core metrics and audit library
│   ├── metrics.py                  ECE, Brier, AUC-RC, oversight lift
│   ├── calibration.py              Platt / Isotonic / Temperature (5-fold OOS)
│   ├── data.py                     Multi-domain result loading
│   └── audit_log.py                HMAC-signed tamper-evident log
│
├── experiments/
│   ├── run_elicitation.py          Run new model elicitation (free APIs)
│   ├── run_self_consistency.py     Self-consistency experiment
│   └── results/                    Committed data — N = 2,004 decisions
│
├── analysis/
│   ├── 01_statistical_analysis.py  Per-model/domain stats + CIs
│   ├── 02_model_comparison.py      Cross-model comparison table + figures
│   ├── 03_error_analysis.py        Difficulty / repo / subject stratification
│   ├── 04_significance_tests.py    Permutation tests + effect sizes
│   ├── 05_prompt_ablation.py       Few-shot vs zero-shot ablation
│   └── 06_baseline_comparison.py   Verbal confidence vs self-consistency
│
├── audit/
│   └── build_and_verify_log.py     Signed chain + tamper detection demo
│
├── tests/
│   └── test_core.py                Unit tests (no external dependencies)
│
├── paper/
│   ├── paper.tex                   LaTeX source
│   ├── paper.pdf                   Final PDF
│   ├── references.bib
│   └── figures/                    6 committed figures (fig1–fig6)
│
└── verify_claims.sh                Verify all paper claims from committed data
```

---

## Run new experiments (free APIs)

```bash
pip install -r requirements-experiments.txt

export GROQ_API_KEY="gsk_..."           # 1,000 req/day free
export GEMINI_API_KEY="AIzaSy..."       # ~1,500 req/day free
export OPENROUTER_API_KEY="sk-or-..."   # deepseek-r1, 50/day free

# Smoke test first (10 items per domain)
python experiments/run_elicitation.py --smoke --backend groq

# Full multi-domain run
python experiments/run_elicitation.py \
  --domains swebench gsm8k mmlu truthfulqa \
  --n-per-domain 250 --backend groq \
  --results-file experiments/results/results_new_model.json
```

After running, re-execute the individual analysis scripts in `analysis/` to extend the comparison table and figures. The committed figures in `paper/figures/` are not overwritten by these scripts.

---

## Data provenance

| File | Model | Domains | N valid |
|---|---|---|---|
| results_llama31_8b.json | Llama 3.1 8B | SWE-bench | 150 |
| results_llama33_70b.json | Llama 3.3 70B | SWE-bench | 150 |
| results_llama33_70b_multidomain.json | Llama 3.3 70B | SWE-bench, GSM8K, MMLU, TruthfulQA | 974 |
| results_gemini25_flash_multidomain.json | Gemini 2.5 Flash | SWE-bench, GSM8K, MMLU, TruthfulQA | 978 |
| results_llama33_70b_multidomain_fewshot.json | Llama 3.3 70B | SWE-bench, GSM8K, MMLU | 537 |
| results_llama33_70b_selfconsistency.json | Llama 3.3 70B (3-sample SC) | SWE-bench | 110 |

See [experiments/results/PROVENANCE.md](experiments/results/PROVENANCE.md) for exact run parameters.

---

## Limitations

- Single-turn judgment proxies; not full multi-step agent trajectories
- Two model families (Llama, Gemini); reasoning models not yet tested
- Self-consistency comparison preliminary (n = 110, SWE-bench only)
- Audit log uses HMAC; production requires Ed25519 + managed key rotation
- EU AI Act operative timeline under active revision (Digital Omnibus)

## License

MIT

## Citation

```bibtex
@techreport{mistry2026confidence,
  title  = {Does Confidence-Gating Work? Empirical Evaluation of a Core AI Oversight Assumption},
  author = {Mistry, Rushabh},
  year   = {2026},
  type   = {Technical Report / Preprint},
  url    = {https://github.com/Rushhaabhhh/Calibrated-Oversight}
}
```
