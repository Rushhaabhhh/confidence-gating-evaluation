# Data Provenance

Every result file in this directory was produced by `../run_elicitation.py`
with a fixed random seed (42) and temperature=0.0 for reproducibility.

| File | Backend | Model | Domains | Valid n | Overall acc |
|---|---|---|---|---|---|
| results_llama31_8b.json                | Groq  | llama-3.1-8b-instant     | swebench                                 | 150 | 0.507 |
| results_llama33_70b.json               | Groq  | llama-3.3-70b-versatile  | swebench                                 | 150 | 0.680 |
| results_llama33_70b_multidomain.json   | Groq  | llama-3.3-70b-versatile  | swebench, gsm8k, mmlu, truthfulqa        | 974 | 0.911 (see per-domain) |
| results_gemini25_flash_multidomain.json| Gemini| gemini-2.5-flash         | swebench, gsm8k, mmlu, truthfulqa        | 978 | 0.913 (see per-domain) |

The elicitor's prompt asks for a single verbalised probability (0–100) per item.
Ground-truth labels come from the dataset itself:
- SWE-bench Verified: human difficulty annotations (`>15 min` = hard)
- GSM8K: exact numeric match against gold answer
- MMLU: letter-choice match against answer key
- TruthfulQA: text match against expert-marked correct option

No GPU was used. No model was trained. All runs are pure API inference.

## Per-domain accuracies (from committed data)

| File | swebench | gsm8k | mmlu | truthfulqa |
|---|---:|---:|---:|---:|
| results_llama33_70b_multidomain.json    | 0.652 | 0.692 | 0.789 | 0.884 |
| results_gemini25_flash_multidomain.json | 0.660 | 0.560 | 0.886 | 0.904 |

**Do NOT re-run these with `--fresh`** — they are the paper's committed baseline
data. To add a new model, use a different `--results-file` path.
