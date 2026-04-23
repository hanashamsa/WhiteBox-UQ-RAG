"""
compare_models.py — Cross-Model AUROC and Calibration Comparison
=================================================================
Run from project root: python notebooks/compare_models.py

Reads analytics results for all available model tags and prints
a side-by-side comparison table.

Requires run_analytics.py to have been run for each model first.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

LABEL_MAP = {
    "mistral":    "Mistral-7B-Instruct-v0.3",
    "qwen":       "Qwen2.5-0.5B-Instruct (RAG)",
    "qwen_norag": "Qwen2.5-0.5B-Instruct (no RAG)",
}

N_BINS = 10


def detect_models():
    """Auto-detect all available model tags from data/ directory."""
    models = []
    # Legacy Mistral without suffix
    if Path("data/phase3_results.jsonl").exists():
        models.append({"tag": "mistral", "label": LABEL_MAP.get("mistral", "mistral"),
                       "path": "data/phase3_results.jsonl"})
    # Tagged files
    for p in sorted(Path("data").glob("phase3_results_*.jsonl")):
        tag = p.stem.replace("phase3_results_", "")
        if tag == "mistral" and any(m["tag"] == "mistral" for m in models):
            continue  # already added legacy
        models.append({"tag": tag, "label": LABEL_MAP.get(tag, tag), "path": str(p)})
    return models



def sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x))


def compute_ece(y_true, y_prob, n_bins=N_BINS):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return ece


def load_phase3(tag: str):
    """Load phase3 results for a given model tag.
    Falls back to phase1 results with tag if phase3 not present."""
    p3 = Path(f"data/phase3_results_{tag}.jsonl")
    p1 = Path(f"data/phase1_results_{tag}.jsonl")

    # Also handle the original Mistral files without suffix
    if tag == "mistral":
        p3_orig = Path("data/phase3_results.jsonl")
        if not p3.exists() and p3_orig.exists():
            p3 = p3_orig

    target = p3 if p3.exists() else p1
    if not target.exists():
        return None, f"No results file found for tag='{tag}'"

    records = []
    with open(target, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records, None


def analyse(tag: str):
    records, err = load_phase3(tag)
    if err:
        return None, err

    labels, logprobs, ret_scores = [], [], []
    for rec in records:
        avg_lp = rec.get("avg_answer_logprob")
        if avg_lp is None:
            toks   = rec.get("token_logprobs", [])
            avg_lp = sum(toks) / len(toks) if toks else None
        lab  = rec.get("label", 0)
        retrieved = rec.get("retrieved", [])
        rs   = max((r.get("score", 0.0) for r in retrieved), default=0.0)
        if avg_lp is not None:
            labels.append(lab)
            logprobs.append(avg_lp)
            ret_scores.append(rs)

    y  = np.array(labels)
    lp = np.array(logprobs)
    rs = np.array(ret_scores)

    auroc_lp = roc_auc_score(y, lp)
    auroc_rs = roc_auc_score(y, rs)

    # Naïve sigmoid calibrated ECE
    probs_ece = np.array([sigmoid(float(x)) for x in lp])
    ece       = compute_ece(y.astype(float), probs_ece)

    lp_corr = lp[y == 1]
    lp_incr = lp[y == 0]
    sep      = float(lp_corr.mean() - lp_incr.mean()) if len(lp_corr) and len(lp_incr) else float("nan")
    acc      = float(y.mean())
    mean_f1s = [rec.get("f1", 0.0) for rec in records]
    mean_f1  = float(np.mean(mean_f1s))

    return {
        "n":           len(labels),
        "accuracy":    acc,
        "mean_f1":     mean_f1,
        "auroc_lp":    auroc_lp,
        "auroc_rs":    auroc_rs,
        "logprob_sep": sep,
        "ece_naive":   ece,
        "mean_lp_corr": float(lp_corr.mean()) if len(lp_corr) else float("nan"),
        "mean_lp_incr": float(lp_incr.mean()) if len(lp_incr) else float("nan"),
    }, None


def main():
    models  = detect_models()
    results = {}
    for m in models:
        stats, err = analyse(m["tag"])
        if err:
            print(f"  ⚠ {m['label']}: {err}")
        else:
            results[m["tag"]] = (m["label"], stats)

    if not results:
        print("No results available. Run run_analytics.py for each model first.")
        return

    # ── Comparison table ──────────────────────────────────────────────────────
    tags   = list(results.keys())
    labels = [results[t][0] for t in tags]
    stats  = [results[t][1] for t in tags]

    col = 28
    width = col + col * len(tags)
    print("\n" + "=" * width)
    print("CROSS-MODEL COMPARISON")
    print("=" * width)

    def row(name, vals, fmt=".4f"):
        line = f"  {name:<{col}}"
        for v in vals:
            line += f"{v:{fmt}}"[:col].ljust(col)
        print(line)

    header = f"  {'Metric':<{col}}"
    for lbl in labels:
        header += f"{lbl:<{col}}"
    print(header)
    print("  " + "─" * (col * (1 + len(tags))))

    row("Samples",         [s["n"]            for s in stats], fmt="d")
    row("Mean F1",         [s["mean_f1"]      for s in stats])
    row("Acc (F1≥0.8)",    [s["accuracy"]     for s in stats])
    row("AUROC (logprob)", [s["auroc_lp"]     for s in stats])
    row("AUROC (retriev)", [s["auroc_rs"]     for s in stats])
    row("Logprob sep",     [s["logprob_sep"]  for s in stats])
    row("Mean LP correct", [s["mean_lp_corr"] for s in stats])
    row("Mean LP incorr",  [s["mean_lp_incr"] for s in stats])
    row("ECE (naive sig)", [s["ece_naive"]    for s in stats])

    print("\n" + "=" * width)

    # ── RAG vs no-RAG ablation (if qwen_norag present) ────────────────────────
    if "qwen" in results and "qwen_norag" in results:
        rag    = results["qwen"][1]
        norag  = results["qwen_norag"][1]
        print("\n── RAG ABLATION (Qwen) ──")
        print(f"  Accuracy WITH    RAG : {rag['accuracy']:.4f}")
        print(f"  Accuracy WITHOUT RAG : {norag['accuracy']:.4f}")
        print(f"  F1 delta (RAG gain)  : {rag['mean_f1'] - norag['mean_f1']:+.4f}")
        print(f"  AUROC  WITH    RAG   : {rag['auroc_lp']:.4f}")
        print(f"  AUROC  WITHOUT RAG   : {norag['auroc_lp']:.4f}")
        delta_auroc = rag["auroc_lp"] - norag["auroc_lp"]
        print(f"  AUROC delta          : {delta_auroc:+.4f}")
        print()
        if abs(delta_auroc) < 0.03:
            print("  → AUROC stable across RAG/no-RAG: UQ signal is INTRINSIC to LM.")
            print("    Confidence signal does not depend on retrieval quality.")
        elif delta_auroc > 0:
            print(f"  → RAG improves AUROC by {delta_auroc:.4f}.")
            print("    Retrieval context stabilises LM confidence — RAG helps UQ.")
        else:
            print(f"  → No-RAG has higher AUROC by {-delta_auroc:.4f}.")
            print("    RAG context may be adding noise to the confidence signal.")

    # ── Cross-model size insight ───────────────────────────────────────────────
    if "mistral" in results and "qwen" in results:
        a, b   = results["mistral"][1], results["qwen"][1]
        la, lb = results["mistral"][0], results["qwen"][0]
        delta  = a["auroc_lp"] - b["auroc_lp"]
        print(f"\n  Model size AUROC delta ({la} − {lb}): {delta:+.4f}")
        if abs(delta) < 0.05:
            print("  → White-box UQ generalises across model size (Δ < 0.05).")
        elif delta > 0:
            print(f"  → Larger model yields stronger UQ signal by {delta:.4f}.")
        else:
            print(f"  → Smaller model yields stronger UQ signal by {-delta:.4f}.")


if __name__ == "__main__":
    main()

