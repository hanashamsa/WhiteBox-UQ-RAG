"""
run_analytics.py — Model-Agnostic Phases 3–7 Runner
=====================================================
Run from project root:

    python notebooks/run_analytics.py --model-tag mistral
    python notebooks/run_analytics.py --model-tag qwen

Reads: data/phase1_results_{tag}.jsonl
Saves: data/phase3_results_{tag}.jsonl
       data/phase5_results_{tag}.jsonl
       data/phase6_fusion_lr_{tag}.pkl
       data/phase4_logprob_hist_{tag}.png

Runs the same analytics as phases 3-7 but parameterised by model tag,
so you don't have to edit any files when adding a new model.
"""
import argparse
import json
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


# ─── ECE helper ───────────────────────────────────────────────────────────────

def compute_ece(y_true, y_prob, n_bins=10):
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(tag: str):
    in_path   = f"data/phase1_results_{tag}.jsonl"
    ph3_path  = f"data/phase3_results_{tag}.jsonl"
    ph5_path  = f"data/phase5_results_{tag}.jsonl"
    lr_path   = f"data/phase6_fusion_lr_{tag}.pkl"
    hist_path = f"data/phase4_logprob_hist_{tag}.png"

    banner = f"ANALYTICS PIPELINE — {tag.upper()}"
    print("=" * 60)
    print(banner)
    print("=" * 60)

    # ── Load phase1 records ────────────────────────────────────────────────────
    records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"\n  Loaded {len(records)} records from {in_path}")

    # ── Phase 3: Logprob signal ────────────────────────────────────────────────
    print("\n── PHASE 3: LOGPROB AUROC ──")
    lp_correct, lp_incorrect = [], []
    labels, avg_logprobs, retrieval_scores = [], [], []

    if Path(ph3_path).exists():
        Path(ph3_path).unlink()

    for rec in records:
        avg_lp = rec.get("avg_answer_logprob")
        label  = rec.get("label", 0)
        retrieved = rec.get("retrieved", [])
        top_rs = max((r.get("score", 0.0) for r in retrieved), default=0.0)

        if avg_lp is None:
            toks   = rec.get("token_logprobs", [])
            avg_lp = sum(toks) / len(toks) if toks else None

        if avg_lp is not None:
            labels.append(label)
            avg_logprobs.append(avg_lp)
            retrieval_scores.append(top_rs)
            (lp_correct if label == 1 else lp_incorrect).append(avg_lp)

        out_rec = dict(rec)
        out_rec["avg_answer_logprob"]  = avg_lp
        out_rec["top_retrieval_score"] = top_rs
        with open(ph3_path, "a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    auroc_lp = roc_auc_score(labels, avg_logprobs)
    n_correct = sum(labels)
    mean_lp_c = np.mean(lp_correct) if lp_correct else float("nan")
    mean_lp_i = np.mean(lp_incorrect) if lp_incorrect else float("nan")

    print(f"  Samples          : {len(labels)}  |  Correct: {n_correct}  Incorrect: {len(labels)-n_correct}")
    print(f"  Mean logprob CORRECT   : {mean_lp_c:.4f}")
    print(f"  Mean logprob INCORRECT : {mean_lp_i:.4f}")
    print(f"  AUROC (logprob)        : {auroc_lp:.4f}")

    # ── Phase 4: Histogram ─────────────────────────────────────────────────────
    print("\n── PHASE 4: DISTRIBUTION HISTOGRAM ──")
    all_vals = lp_correct + lp_incorrect
    bins     = np.linspace(min(all_vals), max(all_vals), 50)
    fig, ax  = plt.subplots(figsize=(9, 5))
    ax.hist(lp_incorrect, bins=bins, alpha=0.55, color="#e74c3c",
            label=f"Incorrect (n={len(lp_incorrect)})", density=True)
    ax.hist(lp_correct,   bins=bins, alpha=0.55, color="#2ecc71",
            label=f"Correct (n={len(lp_correct)})",   density=True)
    ax.axvline(mean_lp_c, color="#27ae60", linestyle="--",
               linewidth=1.5, label=f"Mean correct ({mean_lp_c:.2f})")
    ax.axvline(mean_lp_i, color="#c0392b", linestyle="--",
               linewidth=1.5, label=f"Mean incorrect ({mean_lp_i:.2f})")
    ax.set_xlabel("avg_answer_logprob", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"avg_answer_logprob Distribution — {tag.upper()}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"  Histogram saved to {hist_path}")

    # ── Phase 5: Retrieval AUROC ───────────────────────────────────────────────
    print("\n── PHASE 5: RETRIEVAL SCORE AUROC ──")
    auroc_rs = roc_auc_score(labels, retrieval_scores)
    print(f"  AUROC (retrieval score) : {auroc_rs:.4f}")
    print(f"  AUROC (logprob)         : {auroc_lp:.4f}")
    print(f"  Delta (log - ret)       : {auroc_lp - auroc_rs:+.4f}")
    if Path(ph5_path).exists():
        Path(ph5_path).unlink()
    with open(ph3_path, "r", encoding="utf-8") as fin, \
         open(ph5_path, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line)

    # ── Phase 6: LR Fusion ────────────────────────────────────────────────────
    print("\n── PHASE 6: LOGISTIC REGRESSION FUSION ──")
    X = np.array([[lp, rs] for lp, rs in zip(avg_logprobs, retrieval_scores)])
    y = np.array(labels)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    (tr, te), = sss.split(X, y)
    scaler     = StandardScaler()
    X_tr_s     = scaler.fit_transform(X[tr])
    X_te_s     = scaler.transform(X[te])
    clf        = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr_s, y[tr])
    y_prob     = clf.predict_proba(X_te_s)[:, 1]
    y_pred     = clf.predict(X_te_s)
    auroc_fus  = roc_auc_score(y[te], y_prob)
    auprc_fus  = average_precision_score(y[te], y_prob)
    auroc_lp_te = roc_auc_score(y[te], X[te][:, 0])
    auroc_rs_te = roc_auc_score(y[te], X[te][:, 1])
    print(f"  AUROC fusion     : {auroc_fus:.4f}")
    print(f"  AUPRC fusion     : {auprc_fus:.4f}")
    print(f"  Baseline logprob : {auroc_lp_te:.4f}")
    print(f"  Baseline retriev : {auroc_rs_te:.4f}")
    print(f"  Fusion delta     : {auroc_fus - max(auroc_lp_te, auroc_rs_te):+.4f}")
    print(f"\n  LR coeff  logprob={clf.coef_[0][0]:.4f}, retrieval={clf.coef_[0][1]:.4f}")
    with open(lr_path, "wb") as fout:
        pickle.dump({"clf": clf, "scaler": scaler}, fout)
    print(f"  Model saved to {lr_path}")

    # ── Phase 7: ECE ──────────────────────────────────────────────────────────
    print("\n── PHASE 7: ECE ──")
    y_prob_all = clf.predict_proba(scaler.transform(X))[:, 1]
    ece = compute_ece(y.astype(float), y_prob_all)
    print(f"  ECE (full set, LR model): {ece:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"SUMMARY — {tag.upper()}")
    print(f"  AUROC logprob    : {auroc_lp:.4f}")
    print(f"  AUROC retrieval  : {auroc_rs:.4f}")
    print(f"  AUROC fusion     : {auroc_fus:.4f}")
    print(f"  ECE (LR model)   : {ece:.4f}")
    print(f"  Logprob sep      : {mean_lp_c - mean_lp_i:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-tag", required=True,
        help="Model tag matching the phase1 results file, e.g. 'mistral' or 'qwen'"
    )
    args = parser.parse_args()
    main(args.model_tag)
