#!/usr/bin/env python3
"""
Cross-model AND cross-domain comparison.
Produces Table 1, Figure 1 (reliability diagrams), Figure 2 (risk-coverage),
Figure 3 (confidence histograms), Figure 4 (domain heatmap).

Layout philosophy:
  - Reliability diagrams  : 2D grid  (rows=models, cols=domains)
  - Confidence histograms : 2D grid  (rows=models, cols=domains)
  - Risk-coverage curve   : ONE panel, one line per model, POOLED across domains
                            (fair comparison — not one line per model×domain combo)
  - Domain heatmap        : rows=models, cols=domains, cells=ECE (multi-domain only)

Usage:
  python analysis/02_model_comparison.py
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from calibrated_oversight.data import (
    load_results, load_results_by_domain, discover_result_files,
    DOMAIN_DISPLAY, MODEL_DISPLAY_NAMES
)
from calibrated_oversight import metrics as M
from calibrated_oversight.calibration import kfold_calibration

FIG = ROOT / "paper" / "figures"
RES = ROOT / "analysis" / "results"

# ── Palette & constants ───────────────────────────────────────────────────────
# One colour per *model* (not per model×domain), consistent across all figures
MODEL_COLORS = {
    # ── Existing models ───────────────────────────────────────────────────────
    "Llama-3.1-8B Instruct":            "#e07b39",   # warm orange
    "Llama-3.3-70B Instruct":           "#1f5c99",   # deep blue
    "Llama-3.3-70B (Multi-Domain)":     "#2e8b57",   # sea green
    "Gemini-2.5-Flash (Multi-Domain)":  "#b3202c",   # crimson
    # ── New models ────────────────────────────────────────────────────────────
    "DeepSeek-R1 (Reasoning)":          "#7b2d8b",   # purple (reasoning model)
    "GPT-OSS-120B (Reasoning)":         "#9b3dab",   # lighter purple (reasoning model)
    "Gemini-3.5-Flash":                 "#d43a45",   # crimson (Gemini family)
    "Gemma-4-31B Instruct":             "#1a7340",   # dark green (Google open-weight)
    "Nemotron-3-Super-120B":            "#0d5fa6",   # navy (NVIDIA)
    # ── Pilot / legacy ───────────────────────────────────────────────────────
    "results_multidomain":              "#9b59b6",   # purple (pilot run)
}
DEFAULT_COLOR  = "#555555"

DOMAINS      = ["swebench", "gsm8k", "mmlu", "truthfulqa"]
DOMAIN_SHORT = {
    "swebench":   "SWE-bench",
    "gsm8k":      "GSM8K",
    "mmlu":       "MMLU",
    "truthfulqa": "TruthfulQA",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def model_color(name: str) -> str:
    return MODEL_COLORS.get(name, DEFAULT_COLOR)


def row_for(r):
    ece     = M.expected_calibration_error(r.confidence_pos, r.correct)
    ece_ci  = M.bootstrap_ci_paired(r.confidence_pos, r.correct, M.expected_calibration_error)
    brier   = M.brier_score(r.confidence_pos, r.correct)
    cal     = kfold_calibration(r.confidence_pos, r.correct)
    best    = min((m for m in cal if cal[m]["ece"] is not None), key=lambda m: cal[m]["ece"], default=None)
    coll    = M.confidence_collapse_stats(r.confidence_in_own)
    aucrc   = M.auc_risk_coverage(r.confidence_in_own, r.correct)
    op      = M.oversight_precision(r.confidence_in_own, r.correct, 0.10)
    return dict(model=r.name, stem=r.stem, domain=r.domain, n=r.n_valid,
                accuracy=r.accuracy, ece=ece, ece_ci=ece_ci, brier=brier,
                best_method=best, best_ece=cal[best]["ece"] if best else None,
                conf_std=coll["std"], collapsed=coll["collapsed"],
                auc_rc=aucrc, lift=op["lift"])


def mcnemar_p(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """
    McNemar's test for paired binary outcomes (same items, same seed).
    Returns p-value. Small p (<0.05) means the accuracy difference is significant.
    Requires both arrays to be aligned on the same item set.
    """
    b = int(((correct_a == 1) & (correct_b == 0)).sum())  # A right, B wrong
    c = int(((correct_a == 0) & (correct_b == 1)).sum())  # A wrong, B right
    if b + c == 0:
        return 1.0
    from scipy.stats import chi2
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)             # continuity-corrected
    return float(1 - chi2.cdf(stat, df=1))


def print_table(rows):
    header = f"{'Model':<32} {'Domain':<12} {'n':>5} {'Acc':>6} {'ECE':>7} {'95%CI':>14} {'BestECE':>9} {'σconf':>7} {'AUC-RC':>8} {'Lift':>6}"
    print("\n" + "="*len(header))
    print(header)
    print("-"*len(header))
    for r in rows:
        ci = r["ece_ci"]
        be = r["best_ece"] if r["best_ece"] else float("nan")
        print(f"{r['model']:<32} {DOMAIN_SHORT.get(r['domain'],r['domain']):<12} {r['n']:>5}"
              f" {r['accuracy']:>6.3f} {r['ece']:>7.3f}"
              f" [{ci[0]:.2f},{ci[1]:.2f}] {be:>9.4f}"
              f" {r['conf_std']:>7.3f} {r['auc_rc']:>8.3f} {r['lift']:>6.2f}")
    print("="*len(header))
    print("Lift>1: low-conf items are error-enriched (gate useful). Lift≤1: no useful signal.\n")


# ── Figure 1: Reliability Diagrams — 2D grid (models × domains) ──────────────
def fig_reliability(models_by_dom: dict):
    """
    Rows = models that have multi-domain data (i.e. 4 domains).
    Cols = the 4 canonical domains.
    Single-domain models get one subplot in the SWE-bench column only.
    """
    # Separate multi-domain from single-domain models
    all_model_names = sorted({k[0] for k in models_by_dom})
    multi = [m for m in all_model_names
             if sum(1 for k in models_by_dom if k[0] == m) > 1]
    single = [m for m in all_model_names if m not in multi]

    n_rows = len(multi) + len(single)
    n_cols = len(DOMAINS)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.6 * n_rows),
                             squeeze=False)

    def draw_panel(ax, r, label, color):
        bins = M.reliability_bins(r.confidence_pos, r.correct, n_bins=10)
        xs = [b[0] for b in bins if b[1] is not None]
        ys = [b[1] for b in bins if b[1] is not None]
        ece = M.expected_calibration_error(r.confidence_pos, r.correct)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect")
        ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
        if xs:
            ax.plot(xs, ys, "o-", color=color, lw=2, ms=6)
        ax.set_title(f"{label}\nECE={ece:.3f}", fontsize=8.5, pad=4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)

    # Multi-domain rows
    for row_i, mname in enumerate(multi):
        color = model_color(mname)
        for col_i, dom in enumerate(DOMAINS):
            ax = axes[row_i][col_i]
            key = (mname, dom)
            if key in models_by_dom:
                r = models_by_dom[key]
                draw_panel(ax, r, DOMAIN_SHORT[dom], color)
            else:
                ax.set_visible(False)
            if col_i == 0:
                ax.set_ylabel(mname.replace(" (Multi-Domain)", ""), fontsize=8)
            if row_i == n_rows - 1 or (row_i == len(multi) - 1 and not single):
                ax.set_xlabel("Confidence")

    # Single-domain models: one dedicated row each
    for idx, mname in enumerate(single):
        row_i = len(multi) + idx
        color = model_color(mname)
        # Hide all cells first, then show only the ones with data
        for col_i in range(n_cols):
            axes[row_i][col_i].set_visible(False)
        for col_i, dom in enumerate(DOMAINS):
            key = (mname, dom)
            if key in models_by_dom:
                ax = axes[row_i][col_i]
                ax.set_visible(True)
                r = models_by_dom[key]
                short = mname.split(" Instruct")[0]
                draw_panel(ax, r, f"{short}\n({DOMAIN_SHORT[dom]})", color)
                ax.set_xlabel("Confidence")
        axes[row_i][0].set_ylabel(mname.replace(" Instruct", ""), fontsize=8)

    fig.suptitle("Reliability Diagrams — Calibration by Model & Domain\n"
                 "(diagonal = perfect calibration)", fontsize=11, y=1.01)
    fig.tight_layout()
    out = FIG / "reliability_diagrams.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ── Figure 2: Risk-Coverage — ONE panel, one line per model (pooled) ─────────
def fig_risk_coverage(models_by_dom: dict):
    """
    One line per model — confidence and correctness pooled across all domains
    that model was evaluated on. This gives a fair, apples-to-apples comparison.
    """
    all_model_names = sorted({k[0] for k in models_by_dom})
    # For risk-coverage, collapse "Llama-3.3-70B Instruct" and
    # "Llama-3.3-70B (Multi-Domain)" into one entry per model family.
    # Each unique (model_family, stem_group) gets one line.
    # We keep them separate because they have different domain coverage.

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for mname in all_model_names:
        keys = [(mname, d) for d in DOMAINS if (mname, d) in models_by_dom]
        if not keys:
            continue
        # Pool confidence and correctness vectors across all domains
        conf_all    = np.concatenate([models_by_dom[k].confidence_in_own for k in keys])
        correct_all = np.concatenate([models_by_dom[k].correct           for k in keys])
        base_acc    = correct_all.mean()
        rc          = M.risk_coverage_curve(conf_all, correct_all)
        auc         = M.auc_risk_coverage(conf_all, correct_all)
        color       = model_color(mname)
        n_domains   = len(keys)
        suffix      = f" ({n_domains}D pooled)" if n_domains > 1 else f" ({DOMAIN_SHORT.get(keys[0][1], keys[0][1])})"
        label       = mname.replace(" (Multi-Domain)", "") + suffix

        ax.plot([x * 100 for x in rc["coverage"]],
                [a * 100 for a in rc["accuracy"]],
                "-", color=color, lw=2.2, label=f"{label}  AUC={auc:.3f}")
        ax.axhline(base_acc * 100, ls=":", color=color, alpha=0.40, lw=1.2)

    ax.set_xlabel("Coverage — % of decisions handled autonomously (most-confident first)", fontsize=10)
    ax.set_ylabel("Accuracy on covered decisions (%)", fontsize=10)
    ax.set_title("Risk–Coverage Curves\n"
                 "Does verbalized confidence correctly rank decisions by reliability?\n"
                 "(dotted = each model's pooled baseline accuracy; diagonal = ideal selector)",
                 fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 100); ax.set_ylim(None, None)
    fig.tight_layout()
    out = FIG / "risk_coverage_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ── Figure 3: Confidence Histograms — 2D grid (models × domains) ─────────────
def fig_confidence_hist(models_by_dom: dict):
    """
    Same 2D grid layout as reliability diagrams. Each cell shows the
    confidence distribution for one (model, domain) pair.
    """
    all_model_names = sorted({k[0] for k in models_by_dom})
    multi  = [m for m in all_model_names
              if sum(1 for k in models_by_dom if k[0] == m) > 1]
    single = [m for m in all_model_names if m not in multi]

    n_rows = len(multi) + len(single)
    n_cols = len(DOMAINS)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    def draw_hist(ax, r, label, color):
        coll = M.confidence_collapse_stats(r.confidence_in_own)
        bins = np.linspace(0.5, 1.0, 12)
        ax.hist(r.confidence_in_own, bins=bins, color=color,
                edgecolor="white", alpha=0.82, linewidth=0.6)
        ax.axvline(coll["mean"], color="#222222", ls="--", lw=1.8, label=f"μ={coll['mean']:.2f}")
        flag = " ⚠" if coll["collapsed"] else ""
        ax.set_title(f"{label}\nσ={coll['std']:.3f}{flag}", fontsize=8.5, pad=4)
        ax.set_xlim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.25)

    # Multi-domain rows
    for row_i, mname in enumerate(multi):
        color = model_color(mname)
        for col_i, dom in enumerate(DOMAINS):
            ax = axes[row_i][col_i]
            key = (mname, dom)
            if key in models_by_dom:
                r = models_by_dom[key]
                draw_hist(ax, r, DOMAIN_SHORT[dom], color)
            else:
                ax.set_visible(False)
            if col_i == 0:
                ax.set_ylabel(mname.replace(" (Multi-Domain)", ""), fontsize=8)
            if row_i == n_rows - 1 or (row_i == len(multi) - 1 and not single):
                ax.set_xlabel("Confidence")

    # Single-domain models: one dedicated row each
    for idx, mname in enumerate(single):
        row_i = len(multi) + idx
        color = model_color(mname)
        for col_i in range(n_cols):
            axes[row_i][col_i].set_visible(False)
        for col_i, dom in enumerate(DOMAINS):
            key = (mname, dom)
            if key in models_by_dom:
                ax = axes[row_i][col_i]
                ax.set_visible(True)
                r = models_by_dom[key]
                short = mname.split(" Instruct")[0]
                draw_hist(ax, r, f"{short}\n({DOMAIN_SHORT[dom]})", color)
                ax.set_xlabel("Confidence")
        axes[row_i][0].set_ylabel(mname.replace(" Instruct", ""), fontsize=8)

    fig.suptitle("Confidence Distributions — Narrow Range Indicates Governance Risk\n"
                 "(⚠ = confidence collapse, σ < 0.10)", fontsize=11, y=1.01)
    fig.tight_layout()
    out = FIG / "confidence_histograms.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ── Figure 4: Domain Heatmap — only full multi-domain runs ───────────────────
def fig_domain_heatmap(models_by_dom: dict):
    """
    Clean model × domain ECE heatmap. Only include models that cover ALL 4
    canonical domains (i.e. the two full multi-domain runs). Single-domain
    pilot files are excluded to keep the matrix clean.
    """
    # Filter to models with ≥ 3 domains covered
    model_domain_counts = {}
    for (mname, dom) in models_by_dom:
        model_domain_counts.setdefault(mname, set()).add(dom)

    eligible = sorted([
        m for m, doms in model_domain_counts.items()
        if len(doms) >= 3
    ])
    if len(eligible) < 1:
        print("  Heatmap: no model has ≥ 3 domains covered — skipping.")
        print("           Run multi-domain experiments first (all 4 domains per model).")
        return

    domains = [d for d in DOMAINS if any((m, d) in models_by_dom for m in eligible)]
    n_models  = len(eligible)
    n_domains = len(domains)

    # Build ECE matrix
    ece_mat = np.full((n_models, n_domains), np.nan)
    acc_mat = np.full((n_models, n_domains), np.nan)
    for mi, mname in enumerate(eligible):
        for di, dom in enumerate(domains):
            key = (mname, dom)
            if key in models_by_dom:
                r = models_by_dom[key]
                ece_mat[mi, di] = M.expected_calibration_error(r.confidence_pos, r.correct)
                acc_mat[mi, di] = r.accuracy

    # Clean model names for y-axis labels
    y_labels = [m.replace(" (Multi-Domain)", "") for m in eligible]
    x_labels = [DOMAIN_SHORT.get(d, d) for d in domains]

    fig, axes = plt.subplots(1, 2, figsize=(max(7, n_domains * 2.2), max(3.5, n_models * 1.4 + 1.5)))

    for ax, mat, title, cmap, vmin, vmax, fmt in [
        (axes[0], ece_mat, "Raw ECE (↓ better)",  "RdYlGn_r", 0.0, 0.45, ".3f"),
        (axes[1], acc_mat, "Accuracy (↑ better)",  "RdYlGn",   0.5, 1.0,  ".1%"),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_domains))
        ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=10)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(y_labels, fontsize=9)
        # Annotate each cell
        for mi in range(n_models):
            for di in range(n_domains):
                v = mat[mi, di]
                if not np.isnan(v):
                    txt = format(v, fmt)
                    darkness = (v - vmin) / (vmax - vmin)
                    text_color = "white" if (darkness > 0.65 if cmap.endswith("_r") else darkness < 0.35) else "black"
                    ax.text(di, mi, txt, ha="center", va="center",
                            fontsize=9, fontweight="bold", color=text_color)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=10, pad=8)

    fig.suptitle("Calibration × Performance: Model vs. Domain\n"
                 "(ECE: red=overconfident; Accuracy: green=better)", fontsize=11, y=1.02)
    fig.tight_layout()
    out = FIG / "domain_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    files = discover_result_files(ROOT / "experiments" / "results")
    if not files:
        print("No result files found."); return

    models_by_dom = {}   # (model_name, domain) -> ModelResults
    rows          = []

    # Stems to skip in figures (pilot/test runs with very few items)
    SKIP_STEMS = {"results_multidomain"}

    for f in files:
        if f.stem in SKIP_STEMS:
            continue
        by_domain = load_results_by_domain(f)
        if len(by_domain) > 1:
            for domain, r in sorted(by_domain.items()):
                models_by_dom[(r.name, domain)] = r
                rows.append(row_for(r))
        else:
            r = load_results(f)
            domain = r.domain if r.domain != "all" else list(by_domain.keys())[0] if by_domain else "swebench"
            models_by_dom[(r.name, domain)] = r
            rows.append(row_for(r))

    if not models_by_dom:
        print("No usable results."); return

    print_table(rows)
    with open(RES / "comparison_table.json", "w") as f:
        json.dump(rows, f, indent=2, default=float)

    # CSV export — easier to inspect than JSON
    import csv
    csv_path = RES / "comparison_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        skip = {"ece_ci"}  # tuple — not CSV-serialisable directly
        fieldnames = [k for k in rows[0] if k not in skip]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k not in skip})
    print(f"Table → {RES}/comparison_table.json + .csv")

    print("Generating figures:")
    fig_reliability(models_by_dom)
    fig_risk_coverage(models_by_dom)
    fig_confidence_hist(models_by_dom)
    fig_domain_heatmap(models_by_dom)
    print(f"Table → {RES}/comparison_table.json")


if __name__ == "__main__":
    main()
