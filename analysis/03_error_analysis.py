#!/usr/bin/env python3
"""
Stratified error analysis — domain-aware.
For multi-domain files: breaks down per domain, then per-domain-specific strata.
For swebench: by difficulty_raw and repo.
For mmlu: by subject category.
For gsm8k/truthfulqa: by quartile of problem length (proxy for difficulty).

Usage:
  python analysis/03_error_analysis.py experiments/results/results_llama33_70b.json
  python analysis/03_error_analysis.py --all
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from calibrated_oversight.data import (
    load_results_by_domain, discover_result_files, repo_of, DOMAIN_DISPLAY
)

def stratify(keys, correct, conf_own, min_n=4):
    groups = defaultdict(lambda: {"c":[],"cf":[]})
    for k,c,cf in zip(keys,correct,conf_own):
        groups[k]["c"].append(c); groups[k]["cf"].append(cf)
    rows = []
    for k,g in groups.items():
        if len(g["c"]) < min_n: continue
        acc  = float(np.mean(g["c"]))
        conf = float(np.mean(g["cf"]))
        rows.append({"stratum":str(k),"n":len(g["c"]),"accuracy":acc,
                     "mean_confidence":conf,"gap":conf-acc})
    return sorted(rows, key=lambda r: -abs(r["gap"]))

def print_stratum_table(title, rows):
    if not rows: return
    print(f"\n  ── {title} ──")
    print(f"  {'stratum':<30}{'n':>5}{'acc':>7}{'conf':>8}{'gap':>8}")
    for r in rows:
        flag = "  ← overconfident" if r["gap"]>0.12 else ("  ← underconfident" if r["gap"]<-0.12 else "")
        print(f"  {str(r['stratum']):<30}{r['n']:>5}{r['accuracy']:>7.3f}{r['mean_confidence']:>8.3f}{r['gap']:>+8.3f}{flag}")

def analyze_domain(domain, r):
    meta    = r.meta
    correct = r.correct.tolist()
    conf    = r.confidence_in_own.tolist()
    ids     = [rec.get("instance_id","") for rec in r.records]

    if domain == "swebench":
        diff_keys = [m.get("difficulty_raw","?") for m in meta]
        repo_keys = [repo_of(iid) for iid in ids]
        rows_d = stratify(diff_keys, correct, conf)
        rows_r = stratify(repo_keys, correct, conf)
        print_stratum_table("By difficulty tier", rows_d)
        print_stratum_table("By repository", rows_r)
        return {"by_difficulty": rows_d, "by_repo": rows_r}

    elif domain == "mmlu":
        subj_keys = [m.get("subject","?") for m in meta]
        rows_s = stratify(subj_keys, correct, conf)
        print_stratum_table("By MMLU subject (top 15 by |gap|)", rows_s[:15])
        return {"by_subject": rows_s}

    elif domain == "gsm8k":
        q_lens = [len(r.records[i]["question"]) for i in range(len(correct))]
        quartiles = np.percentile(q_lens, [25,50,75])
        def q(l):
            if l <= quartiles[0]: return "Q1 (shortest)"
            if l <= quartiles[1]: return "Q2"
            if l <= quartiles[2]: return "Q3"
            return "Q4 (longest)"
        rows_q = stratify([q(l) for l in q_lens], correct, conf)
        print_stratum_table("By question length quartile", rows_q)
        return {"by_length_quartile": rows_q}

    elif domain == "truthfulqa":
        cat_keys = [m.get("category","?") for m in meta]
        rows_c = stratify(cat_keys, correct, conf)
        print_stratum_table("By TruthfulQA category (top 15)", rows_c[:15])
        return {"by_category": rows_c}
    return {}

def analyze_file(path):
    out_dir = ROOT / "analysis" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_domain = load_results_by_domain(path)
    print(f"\n{'#'*66}")
    print(f"  {Path(path).stem}  ({len(by_domain)} domain(s))")
    print(f"{'#'*66}")
    all_out = {}
    for domain, r in sorted(by_domain.items()):
        print(f"\n  Domain: {DOMAIN_DISPLAY.get(domain,domain)}  (n={r.n_valid}, acc={r.accuracy:.3f})")
        out = analyze_domain(domain, r)
        all_out[domain] = out
    stem = Path(path).stem
    with open(out_dir / f"error_analysis_{stem}.json","w") as f:
        json.dump({"stem":stem, "domains":all_out}, f, indent=2)
    print(f"\n  → {out_dir}/error_analysis_{stem}.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    files = discover_result_files(ROOT/"experiments"/"results") if args.all else [Path(args.path)]
    for f in files:
        analyze_file(f)

if __name__ == "__main__":
    main()
