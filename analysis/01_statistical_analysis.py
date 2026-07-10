#!/usr/bin/env python3
"""
Per-model statistical analysis with domain breakdown.
Usage:
  python analysis/01_statistical_analysis.py --all
  python analysis/01_statistical_analysis.py experiments/results/results_llama33_70b.json
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from calibrated_oversight.data import load_results, load_results_by_domain, discover_result_files, DOMAIN_DISPLAY
from calibrated_oversight import metrics as M
from calibrated_oversight.calibration import kfold_calibration, backend_name

def analyze_one(r):
    ece     = M.expected_calibration_error(r.confidence_pos, r.correct)
    mce     = M.maximum_calibration_error(r.confidence_pos, r.correct)
    brier   = M.brier_score(r.confidence_pos, r.correct)
    ece_ci  = M.bootstrap_ci_paired(r.confidence_pos, r.correct, M.expected_calibration_error)
    brier_ci= M.bootstrap_ci_paired(r.confidence_pos, r.correct, M.brier_score)
    acc_ci  = M.bootstrap_ci(r.correct, lambda a: a.mean())
    cal     = kfold_calibration(r.confidence_pos, r.correct)
    best    = min((m for m in cal if cal[m]["ece"] is not None), key=lambda m: cal[m]["ece"], default=None)
    coll    = M.confidence_collapse_stats(r.confidence_in_own)
    aucrc   = M.auc_risk_coverage(r.confidence_in_own, r.correct)
    gain70  = M.selective_accuracy_gain(r.confidence_in_own, r.correct, 0.70)
    op10    = M.oversight_precision(r.confidence_in_own, r.correct, 0.10)
    return {
        "model": r.name, "stem": r.stem, "domain": r.domain,
        "n_total": r.n_total, "n_valid": r.n_valid,
        "accuracy": r.accuracy, "accuracy_ci95": acc_ci,
        "raw_ece": ece, "raw_ece_ci95": ece_ci, "raw_mce": mce,
        "raw_brier": brier, "raw_brier_ci95": brier_ci,
        "calibration_methods": {m: {"ece": d["ece"], "brier": d["brier"], "note": d["note"]}
                                 for m, d in cal.items()},
        "best_method": best,
        "best_ece": cal[best]["ece"] if best else None,
        "confidence_distribution": coll, "sharpness": M.sharpness(r.confidence_in_own),
        "auc_rc": aucrc,
        "gain_at_70pct": gain70,
        "oversight_precision_10pct": op10,
        "calibration_backend": backend_name(),
    }

def print_report(s):
    print(f"\n{'='*70}")
    print(f"  {s['model']}  |  domain={s['domain']}  |  n={s['n_valid']}/{s['n_total']}")
    print(f"{'='*70}")
    ci = s["accuracy_ci95"]
    print(f"  Accuracy   {s['accuracy']:.3f}  (95% CI [{ci[0]:.3f}, {ci[1]:.3f}])")
    ci = s["raw_ece_ci95"]
    print(f"  Raw ECE    {s['raw_ece']:.3f}  (95% CI [{ci[0]:.3f}, {ci[1]:.3f}])")
    ci = s["raw_brier_ci95"]
    print(f"  Raw Brier  {s['raw_brier']:.3f}  (95% CI [{ci[0]:.3f}, {ci[1]:.3f}])")
    print(f"  ── Post-hoc calibration (5-fold OOS) ──")
    for m, d in s["calibration_methods"].items():
        star = "  ← best" if m == s["best_method"] else ""
        if d["ece"] is not None:
            print(f"     {m:<12} ECE={d['ece']:.4f}  Brier={d['brier']:.4f}{star}")
    cd = s["confidence_distribution"]
    print(f"  Confidence: mean={cd['mean']:.3f} std={cd['std']:.3f} [{cd['min']:.2f},{cd['max']:.2f}]  collapsed={cd['collapsed']}")
    print(f"  AUC-RC={s['auc_rc']:.3f}  gain@70%={s['gain_at_70pct']['gain']:+.3f}  lift={s['oversight_precision_10pct']['lift']:.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    out_dir = ROOT / "analysis" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_result_files(ROOT / "experiments" / "results") if args.all else [Path(args.path)]

    for f in files:
        # Domain breakdown if multi-domain file
        by_domain = load_results_by_domain(f)
        if len(by_domain) > 1:
            print(f"\n{'#'*70}")
            print(f"  {f.stem}  —  {len(by_domain)} domains")
            print(f"{'#'*70}")
            all_stats = []
            for domain, r in sorted(by_domain.items()):
                s = analyze_one(r)
                print_report(s)
                all_stats.append(s)
            stem = f.stem
            with open(out_dir / f"stats_{stem}.json", "w") as fh:
                json.dump(all_stats, fh, indent=2)
        else:
            r = load_results(f)
            s = analyze_one(r)
            print_report(s)
            with open(out_dir / f"stats_{f.stem}.json", "w") as fh:
                json.dump(s, fh, indent=2)
        print(f"  → {out_dir}/stats_{f.stem}.json\n")

if __name__ == "__main__":
    main()
