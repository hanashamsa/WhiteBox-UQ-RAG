"""
risk_coverage.py — Risk-Coverage Curves
========================================
Run from project root: python notebooks/risk_coverage.py

For each available model:
  - Sort predictions by avg_answer_logprob (most to least confident)
  - At each coverage level c (fraction of samples retained):
      accuracy = fraction of retained samples with F1 >= 0.8
  - Plot accuracy-vs-coverage for all models + random baseline

The key insight: if UQ is useful, accuracy should increase as
we reduce coverage (abstain on low-confidence answers).
Area Under the Risk-Coverage Curve (AURC) is also reported.

Saves: data/risk_coverage.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

OUT_PNG = "data/risk_coverage.png"

# Colour palette per tag
PALETTE = {
    "mistral":     ("#3498db", "Mistral-7B-Instruct-v0.3 (RAG)"),
    "qwen":        ("#2ecc71", "Qwen2.5-0.5B-Instruct (RAG)"),
    "qwen_norag":  ("#e67e22", "Qwen2.5-0.5B-Instruct (no RAG)"),
}
DEFAULT_COLORS = ["#9b59b6", "#1abc9c", "#e74c3c", "#f39c12"]


def detect_tags():
    tags = {}
    if Path("data/phase3_results.jsonl").exists():
        tags["mistral"] = "data/phase3_results.jsonl"
    for p in sorted(Path("data").glob("phase3_results_*.jsonl")):
        tag = p.stem.replace("phase3_results_", "")
        tags[tag] = str(p)
    return tags


def load_data(path: str):
    lp, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            avg = rec.get("avg_answer_logprob")
            if avg is None:
                toks = rec.get("token_logprobs", [])
                avg  = sum(toks) / len(toks) if toks else None
            lab = rec.get("label", 0)
            if avg is not None:
                lp.append(avg)
                labels.append(lab)
    return np.array(lp), np.array(labels, dtype=int)


def risk_coverage_curve(lp: np.ndarray, y: np.ndarray, n_points: int = 100):
    """
    Returns (coverages, accuracies).
    Coverage decreases from 1.0 to near 0 as we keep only the most confident.
    """
    order    = np.argsort(lp)[::-1]   # most confident first
    lp_s     = lp[order]
    y_s      = y[order]
    n        = len(y_s)
    steps    = np.unique(np.linspace(1, n, n_points, dtype=int))
    cov, acc = [], []
    for k in steps:
        cov.append(k / n)
        acc.append(y_s[:k].mean())
    return np.array(cov), np.array(acc)


def aurc(coverages: np.ndarray, accuracies: np.ndarray) -> float:
    """Area under the risk (1-accuracy) vs coverage curve."""
    risks = 1.0 - accuracies
    return float(np.trapz(risks, coverages))


def main():
    print("=" * 60)
    print("RISK-COVERAGE CURVES")
    print("=" * 60)

    tags = detect_tags()
    if not tags:
        print("  No phase3 result files found. Run run_analytics.py first.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    default_i = 0
    aurc_vals  = {}

    for tag, path in tags.items():
        lp, y = load_data(path)
        overall_acc = y.mean()
        cov, acc    = risk_coverage_curve(lp, y)
        auroc       = roc_auc_score(y, lp)
        arc         = aurc(cov, acc)
        aurc_vals[tag] = arc

        color, label = PALETTE.get(tag, (DEFAULT_COLORS[default_i % len(DEFAULT_COLORS)], tag))
        if tag not in PALETTE:
            default_i += 1

        ax.plot(cov, acc, color=color, linewidth=2.2, label=f"{label}\n(AUROC={auroc:.3f}, AURC={arc:.3f})")
        ax.axhline(overall_acc, color=color, linestyle=":", linewidth=1.0, alpha=0.5)

        print(f"\n  [{tag}]  {label}")
        print(f"    Overall accuracy  : {overall_acc:.4f}")
        print(f"    Top-10% accuracy  : {acc[np.searchsorted(cov, 0.10)]:.4f}")
        print(f"    Top-25% accuracy  : {acc[np.searchsorted(cov, 0.25)]:.4f}")
        print(f"    Top-50% accuracy  : {acc[np.searchsorted(cov, 0.50)]:.4f}")
        print(f"    AUROC             : {auroc:.4f}")
        print(f"    AURC (lower=better): {arc:.4f}")

    # ── Random baseline ────────────────────────────────────────────────────────
    # For any ordering, random confidence gives flat accuracy = overall accuracy
    # Use first model's data for the reference line
    first_path = next(iter(tags.values()))
    _, y_ref   = load_data(first_path)
    overall    = y_ref.mean()
    ax.axhline(overall, color="#95a5a6", linestyle="--", linewidth=1.5,
               label=f"Random baseline (accuracy={overall:.3f})")

    # ── Formatting ─────────────────────────────────────────────────────────────
    ax.set_xlabel("Coverage (fraction of questions answered)", fontsize=12)
    ax.set_ylabel("Accuracy (fraction correct among answered)", fontsize=12)
    ax.set_title("Risk-Coverage Curve\n"
                 "← abstain more  |  higher = better confidence discrimination",
                 fontsize=13)
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # Annotation: "abstain zone"
    ax.fill_betweenx([0, 1.05], 0, 0.25, alpha=0.04, color="grey")
    ax.text(0.125, 0.05, "abstain\nzone\n(top 25%)", ha="center",
            fontsize=8, color="grey")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()

    print(f"\n  Plot saved to: {OUT_PNG}")
    print("=" * 60)

    # ── Practical utility summary ─────────────────────────────────────────────
    print("\n  PRACTICAL UTILITY:")
    print("  At 25% abstention rate (answer only 75% of questions):")
    for tag, path in tags.items():
        lp, y = load_data(path)
        cov, acc = risk_coverage_curve(lp, y)
        _, label = PALETTE.get(tag, ("", tag))
        acc_75 = float(acc[np.searchsorted(cov, 0.75)])
        base   = float(y.mean())
        print(f"    {label or tag}: {base:.3f} → {acc_75:.3f}  ({acc_75-base:+.3f})")


if __name__ == "__main__":
    main()
