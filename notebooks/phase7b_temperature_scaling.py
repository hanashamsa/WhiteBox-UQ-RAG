"""
PHASE 7b — Temperature Scaling Calibration
============================================
Run from project root: python notebooks/phase7b_temperature_scaling.py

Fits a single temperature T on the training split (70%) to minimise
Negative Log-Likelihood of the calibrated sigmoid probability.

The calibrated confidence for a sample is:
    p = sigmoid(avg_answer_logprob / T)

Evaluates ECE on the held-out test split (30%) before and after scaling.

No model re-loading. Reads data/phase3_results.jsonl only.
"""
import json
import math
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedShuffleSplit

PHASE3_PATH  = "data/phase3_results.jsonl"
N_BINS       = 10


# ─── Helpers ──────────────────────────────────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def calibrated_prob(logprob: float, T: float) -> float:
    """Map avg_answer_logprob through temperature-scaled sigmoid."""
    return sigmoid(logprob / T)


def nll_loss(T: float, logprobs: np.ndarray, labels: np.ndarray) -> float:
    """Average binary cross-entropy with temperature T."""
    eps = 1e-8
    total = 0.0
    for lp, y in zip(logprobs, labels):
        p = calibrated_prob(float(lp), float(T))
        p = max(eps, min(1 - eps, p))
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(labels)


def compute_ece(logprobs: np.ndarray, labels: np.ndarray,
                T: float = 1.0, n_bins: int = 10) -> float:
    probs = np.array([calibrated_prob(float(lp), T) for lp in logprobs])
    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    n     = len(labels)
    for i in range(n_bins):
        lo, hi   = bins[i], bins[i + 1]
        mask     = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        if mask.sum() == 0:
            continue
        count    = mask.sum()
        avg_conf = probs[mask].mean()
        acc      = labels[mask].mean()
        ece     += (count / n) * abs(avg_conf - acc)
    return ece


def print_reliability(logprobs, labels, T, n_bins=10):
    probs = np.array([calibrated_prob(float(lp), T) for lp in logprobs])
    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    n     = len(labels)
    print(f"  {'Bin':^12} {'Count':^6} {'AvgConf':^9} {'Acc':^9} {'|Gap|':^8}")
    print("  " + "─" * 50)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        if mask.sum() == 0:
            continue
        cnt    = mask.sum()
        conf   = probs[mask].mean()
        acc    = labels[mask].mean()
        gap    = abs(conf - acc)
        bar    = "█" * int(gap * 20)
        print(f"  [{lo:.1f},{hi:.1f}){cnt:6d}   {conf:.4f}   {acc:.4f}   {gap:.4f}  {bar}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 7b — TEMPERATURE SCALING")
    print("=" * 60)

    # Load data
    lp_all, y_all = [], []
    with open(PHASE3_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lp  = rec.get("avg_answer_logprob")
            lab = rec.get("label", 0)
            if lp is not None:
                lp_all.append(lp)
                y_all.append(lab)

    lp_all = np.array(lp_all)
    y_all  = np.array(y_all, dtype=int)
    print(f"\n  Samples: {len(y_all)}  |  Positive: {y_all.sum()}")

    # 70/30 stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    (tr_idx, te_idx), = sss.split(lp_all, y_all)
    lp_train, lp_test = lp_all[tr_idx], lp_all[te_idx]
    y_train,  y_test  = y_all[tr_idx],  y_all[te_idx]

    # ── Fit temperature on train set ────────────────────────────────────────────
    result = minimize_scalar(
        lambda T: nll_loss(T, lp_train, y_train),
        bounds=(0.01, 10.0),
        method="bounded",
    )
    T_opt = result.x
    print(f"\n  Optimal temperature T = {T_opt:.4f}")
    print(f"  (T=1 means no scaling; T>1 sharpens, T<1 softens)")

    # ── Evaluate ECE before and after ──────────────────────────────────────────
    ece_before   = compute_ece(lp_test, y_test, T=1.0)
    ece_after    = compute_ece(lp_test, y_test, T=T_opt)

    print(f"\n  ECE BEFORE temperature scaling : {ece_before:.4f}")
    print(f"  ECE AFTER  temperature scaling : {ece_after:.4f}")
    print(f"  ECE improvement                : {ece_before - ece_after:+.4f}")

    print("\n  ── Reliability table (T=1.0, before) ──")
    print_reliability(lp_test, y_test, T=1.0)

    print(f"\n  ── Reliability table (T={T_opt:.3f}, after) ──")
    print_reliability(lp_test, y_test, T=T_opt)

    print("\n" + "=" * 60)
    if ece_after < 0.05:
        print("✓ ECE < 0.05 after scaling — well-calibrated.")
    elif ece_after < 0.10:
        print(f"⚠ ECE {ece_after:.4f} — moderate. Acceptable for research.")
    else:
        print(f"✗ ECE {ece_after:.4f} still high. Try isotonic regression.")
    print("=" * 60)
    print(f"\n  T*={T_opt:.4f} — save this for deployment confidence mapping:")
    print(f"  confidence = sigmoid(avg_answer_logprob / {T_opt:.4f})")


if __name__ == "__main__":
    main()
