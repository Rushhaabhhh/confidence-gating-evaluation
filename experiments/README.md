# Experiments — Multi-Domain Verbalized Confidence Elicitation

## Domains

| Domain | Dataset | HuggingFace ID | Task | Label |
|---|---|---|---|---|
| swebench | SWE-bench Verified | princeton-nlp/SWE-bench_Verified | Predict if fix >15min | difficulty annotation |
| gsm8k | Grade School Math 8K | openai/gsm8k | Solve math problem | exact numeric match |
| mmlu | Massive Multitask Language Understanding | cais/mmlu (all) | Answer MCQ (A/B/C/D) | answer key |
| truthfulqa | TruthfulQA Multiple Choice | truthfulqa/truthful_qa | Choose truthful answer | expert-marked truth |

## Committed result files

| File | Model | Domain | n | Accuracy |
|---|---|---|---|---|
| results/results_llama31_8b.json | Llama-3.1-8B Instruct | swebench | 150 | 0.507 |
| results/results_llama33_70b.json | Llama-3.3-70B Instruct | swebench | 150 | 0.680 |

Do NOT re-run these without --fresh — they are the paper's committed baseline data.

## Record schema (every field present in all domains)

```json
{
  "domain":                  "swebench",
  "instance_id":             "django__django-15022",
  "question":                "...",
  "prediction":              1,
  "label":                   1,
  "confidence":              0.80,
  "confidence_in_own_answer": 0.80,
  "oversight_flagged":       false,
  "parse_error":             false,
  "raw_response":            "ANSWER: YES\nPROBABILITY: 80\nREASONING: ...",
  "timestamp":               "2026-07-07T...",
  "domain_meta":             {"difficulty_raw": "15 min - 1 hour", "repo": "django"}
}
```

## Why one PROBABILITY (not ANSWER + CONFIDENCE)

Earlier versions asked for a prediction + a separate confidence score.
That design had a critical bug: models were told to "use low confidence sparingly"
which caused them to always report 70-90%, breaking the oversight gate (0 items
ever flagged) and collapsing the confidence distribution.

The current design elicits a single PROBABILITY (0-100) aligned to a specific
direction, with no guidance to avoid low values. This matches the Tian et al.
(2023) verbalized confidence methodology and produces a spread that can be analyzed.
