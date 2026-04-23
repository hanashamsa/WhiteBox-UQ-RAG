"""
PHASE 7 — Calibration Check (ECE)
===================================
Run from project root: python notebooks/phase7_calibration.py

Uses the logistic regression model from Phase 6 to produce
probability estimates, then computes Expected Calibration Error (ECE).

ECE measures whether predicted confidence matches empirical accuracy.
Lower ECE = better calibrated.

Outputs: prints ECE + reliability diagram to stdout
"""
import json
import pickle

import numpy as np

PHASE3_PATH = "data/phase3_results.jsonl"
MODEL_PATH  = "data/phase6_fusion_lr.pkl"
N_BINS      = 10


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (equal-width bins).
    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    bins     = np.linspace(0.0, 1.0, n_bins + 1)
    ece      = 0.0
    n        = len(y_true)

    print(f"\n  {'Bin':^12} {'Count':^7} {'Avg Conf':^10} {'Acc':^10} {'|Conf-Acc|':^12}")
    print("  " + "-" * 55)

    for i in range(n_bins):
        lo, hi  = bins[i], bins[i + 1]
        # include upper edge in last bin
        mask = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        if mask.sum() == 0:
            continue
        count    = mask.sum()
        avg_conf = y_prob[mask].mean()
        acc      = y_true[mask].mean()
        gap      = abs(avg_conf - acc)
        ece     += (count / n) * gap

        bar = "█" * int(gap * 20)
        print(f"  [{lo:.1f}, {hi:.1f})  {count:5d}   {avg_conf:8.4f}   {acc:8.4f}   {gap:8.4f}  {bar}")

    return ece


def main():
    print("=" * 60)
    print("PHASE 7 — CALIBRATION (ECE)")
    print("=" * 60)

    # ── Load model + scaler ────────────────────────────────────────────────────
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    clf    = bundle["clf"]
    scaler = bundle["scaler"]

    # ── Load features ──────────────────────────────────────────────────────────
    X_list = []
    y_list = []
    with open(PHASE3_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            lp  = rec.get("avg_answer_logprob")
            rs  = rec.get("top_retrieval_score")
            lab = rec.get("label", 0)
            if lp is not None and rs is not None:
                X_list.append([lp, rs])
                y_list.append(lab)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=int)

    X_s    = scaler.transform(X)
    y_prob = clf.predict_proba(X_s)[:, 1]

    print(f"\n  Total samples     : {len(y)}")
    print(f"  Predicted prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
    print(f"  True positive rate  : {y.mean():.4f}")

    ece = compute_ece(y, y_prob, n_bins=N_BINS)

    print(f"\n  Expected Calibration Error (ECE, {N_BINS} bins): {ece:.4f}")
    print("=" * 60)

    print()
    if ece < 0.05:
        print("✓ ECE < 0.05 — System is well-calibrated. Usable as confidence estimator.")
    elif ece < 0.10:
        print("⚠ ECE 0.05–0.10 — Moderate calibration. Acceptable for research.")
    else:
        print(f"✗ ECE = {ece:.4f} — Poor calibration. Probabilities are unreliable.")
        print("  → Consider Platt scaling or isotonic regression for post-hoc calibration.")

    print()
    print("Reliability diagram (bin-level):")
    print("  Confidence vs. Accuracy — see table above.")
    print("  If bars are large, confidence != accuracy → poor calibration.")


if __name__ == "__main__":
    main()
