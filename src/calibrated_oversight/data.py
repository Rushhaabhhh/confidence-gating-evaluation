"""
Loading and normalizing experiment result files — multi-domain version.

Result files are JSON lists of per-item records. Every valid record has:
    domain       : "swebench" | "gsm8k" | "mmlu" | "truthfulqa"
    instance_id  : unique id within the domain
    question     : the question/problem shown to the model
    prediction   : 0 or 1  (model's binary answer)
    label        : 0 or 1  (ground truth)
    confidence   : float in [0,1] = P(label=1) as stated by the model
    confidence_in_own_answer : max(confidence, 1-confidence)
    oversight_flagged : bool
    parse_error  : bool
    raw_response : str
    domain_meta  : dict  (domain-specific fields, e.g. difficulty_raw, subject)
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Map file stem -> display name. Add new models here.
# Old entries kept so existing committed result files still resolve correctly.
MODEL_DISPLAY_NAMES = {
    # ── Committed baseline runs (SWE-bench only) ──────────────────────────────
    "results_llama31_8b":                   "Llama-3.1-8B Instruct",
    "results_llama33_70b":                  "Llama-3.3-70B Instruct",
    # ── Multi-domain runs ─────────────────────────────────────────────────────
    "results_llama33_70b_multidomain":      "Llama-3.3-70B (Multi-Domain)",
    "results_gemini25_flash_multidomain":   "Gemini-2.5-Flash (Multi-Domain)",
    # ── New models (free-tier expansion) ──────────────────────────────────────
    "results_deepseek_r1":                  "DeepSeek-R1 (Reasoning)",
    "results_gpt_oss_120b":                 "GPT-OSS-120B (Reasoning)",
    "results_gemini35_flash":               "Gemini-3.5-Flash",
    "results_gemma4_31b":                   "Gemma-4-31B Instruct",
    "results_nemotron3_super":              "Nemotron-3-Super-120B",
    # ── Legacy / pilot ────────────────────────────────────────────────────────
    "results_llama33_70b_nim":              "Llama-3.3-70B (NIM)",
    "results_qwen3_235b":                   "Qwen3-235B",
}

DOMAIN_DISPLAY = {
    "swebench":   "SWE-bench (Software Eng.)",
    "gsm8k":      "GSM8K (Math Reasoning)",
    "mmlu":       "MMLU (Multi-domain Knowledge)",
    "truthfulqa": "TruthfulQA (Factual Truthfulness)",
}


@dataclass
class ModelResults:
    name: str
    stem: str
    domain: str                          # may be "all" when pooled
    n_total: int
    n_valid: int
    confidence_pos:   np.ndarray
    confidence_in_own: np.ndarray
    prediction:       np.ndarray
    label:            np.ndarray
    correct:          np.ndarray
    domain_tags:      list               # per-item domain when pooled
    meta:             list               # per-item domain_meta dicts
    records:          list               # raw valid records

    @property
    def accuracy(self) -> float:
        return float(self.correct.mean()) if self.n_valid else float("nan")


def _parse_record(r: dict) -> dict | None:
    """Normalise a record regardless of which version of the elicitor wrote it.

    Schema note (important): the elicitor writes `prediction` and `label` with
    different semantics depending on the domain, and this function unifies them:

      - SWE-bench:  prediction = model's YES/NO answer (0 or 1),
                    label      = ground-truth difficulty (0 or 1).
                    Correctness = (prediction == label).

      - GSM8K / MMLU / TruthfulQA:  the elicitor sets both prediction and label
                    to the *correctness bit* (1 if the model's answer matched
                    ground truth, 0 otherwise). This is redundant encoding.
                    Here we canonicalise to prediction=1, label=correctness,
                    so downstream `correct = (prediction == label)` is well-defined.
                    (The elicitor should be simplified to write this directly;
                    see experiments/run_elicitation.py for the source-of-truth
                    parser that produced these files.)
    """
    if r.get("parse_error"):
        return None
    c = r.get("confidence")
    if c is None:
        return None
    c = float(c)
    c_own = r.get("confidence_in_own_answer", max(c, 1.0 - c))
    pred = int(r.get("prediction", 1 if c >= 0.5 else 0))
    lbl  = int(r.get("label", r.get("correct_label", 0)))

    # domain: present in new files; fall back to swebench for old files
    domain = r.get("domain", "swebench")

    if domain != "swebench":
        # For QA/reasoning tasks, correctness is stored in the raw label/prediction field.
        # We align with (pred == label) logic by setting pred=1 and label=correctness.
        correctness = lbl
        pred = 1
        lbl = correctness

    # domain_meta: severity/subject/difficulty fields
    meta = r.get("domain_meta", {})
    if not meta:
        # old swebench files stored difficulty_raw at top level
        if r.get("difficulty_raw"):
            meta = {"difficulty_raw": r["difficulty_raw"]}

    return {
        "domain":                  domain,
        "instance_id":             r.get("instance_id", ""),
        "question":                r.get("problem_statement", r.get("question", "")),
        "prediction":              pred,
        "label":                   lbl,
        "confidence":              c,
        "confidence_in_own_answer": float(c_own),
        "oversight_flagged":       bool(r.get("oversight_flagged", c_own < 0.70)),
        "raw_response":            r.get("raw_response", ""),
        "domain_meta":             meta,
        "parse_error":             False,
    }


def load_results(path: str | Path, domain_filter: str | None = None) -> ModelResults:
    path = Path(path)
    raw = json.load(open(path, encoding="utf-8"))
    valid = [_parse_record(r) for r in raw]
    valid = [r for r in valid if r is not None]
    if domain_filter:
        valid = [r for r in valid if r["domain"] == domain_filter]

    c_pos  = np.array([r["confidence"] for r in valid], dtype=float)
    c_own  = np.array([r["confidence_in_own_answer"] for r in valid], dtype=float)
    pred   = np.array([r["prediction"] for r in valid], dtype=int)
    label  = np.array([r["label"] for r in valid], dtype=int)
    correct = (pred == label).astype(int)

    stem = path.stem
    domain = domain_filter or (valid[0]["domain"] if valid else "unknown")
    if len(set(r["domain"] for r in valid)) > 1:
        domain = "all"

    return ModelResults(
        name=MODEL_DISPLAY_NAMES.get(stem, stem),
        stem=stem,
        domain=domain,
        n_total=len(raw),
        n_valid=len(valid),
        confidence_pos=c_pos,
        confidence_in_own=c_own,
        prediction=pred,
        label=label,
        correct=correct,
        domain_tags=[r["domain"] for r in valid],
        meta=[r["domain_meta"] for r in valid],
        records=valid,
    )


def load_results_by_domain(path: str | Path) -> dict[str, ModelResults]:
    """Return {domain: ModelResults} for every domain present in the file."""
    raw = json.load(open(Path(path), encoding="utf-8"))
    valid = [_parse_record(r) for r in raw if not r.get("parse_error")]
    valid = [r for r in valid if r]
    domains = sorted(set(r["domain"] for r in valid))
    out = {}
    for d in domains:
        sub = [r for r in valid if r["domain"] == d]
        if len(sub) < 10:
            continue
        c_pos  = np.array([r["confidence"] for r in sub], dtype=float)
        c_own  = np.array([r["confidence_in_own_answer"] for r in sub], dtype=float)
        pred   = np.array([r["prediction"] for r in sub], dtype=int)
        label  = np.array([r["label"] for r in sub], dtype=int)
        stem   = Path(path).stem
        out[d] = ModelResults(
            name=MODEL_DISPLAY_NAMES.get(stem, stem),
            stem=stem,
            domain=d,
            n_total=len(sub),
            n_valid=len(sub),
            confidence_pos=c_pos,
            confidence_in_own=c_own,
            prediction=pred,
            label=label,
            correct=(pred == label).astype(int),
            domain_tags=[d] * len(sub),
            meta=[r["domain_meta"] for r in sub],
            records=sub,
        )
    return out


def discover_result_files(results_dir: str | Path, min_valid: int = 30) -> list[Path]:
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob("results_*.json"))
    keep = []
    for f in files:
        try:
            raw = json.load(open(f))
            n = sum(1 for r in raw if not r.get("parse_error") and r.get("confidence") is not None)
            if n >= min_valid:
                keep.append(f)
        except Exception:
            continue
    return keep


def repo_of(instance_id: str) -> str:
    return instance_id.split("__")[0] if "__" in instance_id else "unknown"
