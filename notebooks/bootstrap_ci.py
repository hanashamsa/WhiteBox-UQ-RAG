"""
bootstrap_ci.py — Bootstrap AUROC Confidence Intervals + Significance Tests
=============================================================================
Run from project root: python notebooks/bootstrap_ci.py

For each available model tag:
  - Loads phase3 results
  - Computes AUROC + 95% bootstrap CI (2000 resamples)

Between any pair of models:
  - Bootstrap permutation test: is the AUROC difference significant?

No model loading required.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

N_BOOTSTRAP = 2000
ALPHA       = 0.05    # 95% CI
SEED        = 42
rng         = np.random.default_rng(SEED)

# Auto-detect available model tags from data/ directory
def detect_tags():
    """Find all phase3_results_{tag}.jsonl files plus legacy phase3_results.jsonl."""
    tags = {}
    # Legacy Mistral file
    if Path("data/phase3_results.jsonl").exists():
        tags["mistral"] = ("data/phase3_results.jsonl", "Mistral-7B-Instruct-v0.3")
    # Tagged files
    for p in sorted(Path("data").glob("phase3_results_*.jsonl")):
        tag = p.stem.replace("phase3_results_", "")
        labels = {
            "qwen":        "Qwen2.5-0.5B-Instruct",
            "qwen_norag":  "Qwen2.5-0.5B (no RAG)",
            "mistral":     "Mistral-7B-Instruct-v0.3",
        }
        tags[tag] = (str(p), labels.get(tag, tag))
    return tags


def load_logprobs_labels(path: str):
    lp, y = [], []
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
                y.append(lab)
    return np.array(lp), np.array(y, dtype=int)


def bootstrap_auroc(lp, y, n_boot=N_BOOTSTRAP):
    """Return (mean_auroc, ci_low, ci_high, all_boot_aurocs)."""
    n     = len(y)
    aucs  = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b  = y[idx]
        lp_b = lp[idx]
        if y_b.sum() == 0 or (1 - y_b).sum() == 0:
            continue  # skip degenerate bootstrap samples
        aucs.append(roc_auc_score(y_b, lp_b))
    aucs = np.array(aucs)
    ci_lo = np.percentile(aucs, 100 * ALPHA / 2)
    ci_hi = np.percentile(aucs, 100 * (1 - ALPHA / 2))
    return aucs.mean(), ci_lo, ci_hi, aucs


def permutation_test_auroc(lp_a, y_a, lp_b, y_b, n_perm=2000):
    """
    Test H0: AUROC(A) == AUROC(B) by permuting labels between groups.
    Returns (observed_delta, p_value).
    """
    obs_a   = roc_auc_score(y_a, lp_a)
    obs_b   = roc_auc_score(y_b, lp_b)
    obs_d   = obs_a - obs_b

    # Pool and permute
    lp_all  = np.concatenate([lp_a, lp_b])
    y_all   = np.concatenate([y_a,  y_b])
    na, nb  = len(y_a), len(y_b)
    count   = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(y_all))
        y_pa = y_all[perm[:na]]
        y_pb = y_all[perm[na:]]
        lp_pa = lp_all[perm[:na]]
        lp_pb = lp_all[perm[na:]]
        if y_pa.sum() in (0, na) or y_pb.sum() in (0, nb):
            continue
        d = roc_auc_score(y_pa, lp_pa) - roc_auc_score(y_pb, lp_pb)
        if abs(d) >= abs(obs_d):
            count += 1
    p_val = count / n_perm
    return obs_d, p_val


def main():
    print("=" * 65)
    print("BOOTSTRAP AUROC CONFIDENCE INTERVALS")
    print(f"  Bootstrap resamples  : {N_BOOTSTRAP}")
    print(f"  Significance level   : α={ALPHA}  (95% CI)")
    print("=" * 65)

    tags = detect_tags()
    if not tags:
        print("  No phase3 result files found. Run run_analytics.py first.")
        return

    results = {}
    for tag, (path, label) in tags.items():
        print(f"\n── {label} ──")
        lp, y = load_logprobs_labels(path)
        true_auroc = roc_auc_score(y, lp)
        mean_auc, ci_lo, ci_hi, boot_aucs = bootstrap_auroc(lp, y)

        print(f"  N={len(y)}  |  Pos={y.sum()}  Neg={(1-y).sum()}")
        print(f"  Observed AUROC        : {true_auroc:.4f}")
        print(f"  Bootstrap mean AUROC  : {mean_auc:.4f}")
        print(f"  95% CI                : [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  CI width              : {ci_hi - ci_lo:.4f}")

        results[tag] = {
            "label":      label,
            "lp":         lp,
            "y":          y,
            "auroc":      true_auroc,
            "ci_lo":      ci_lo,
            "ci_hi":      ci_hi,
            "boot_aucs":  boot_aucs,
        }

    # ── Pairwise significance tests ────────────────────────────────────────────
    tag_list = list(results.keys())
    if len(tag_list) >= 2:
        print("\n" + "=" * 65)
        print("PAIRWISE SIGNIFICANCE TESTS  (bootstrap permutation, 2000 perms)")
        print("=" * 65)
        for i in range(len(tag_list)):
            for j in range(i + 1, len(tag_list)):
                ta, tb = tag_list[i], tag_list[j]
                ra, rb = results[ta], results[tb]
                delta, p = permutation_test_auroc(ra["lp"], ra["y"], rb["lp"], rb["y"])
                sig = "**SIGNIFICANT**" if p < 0.05 else "not significant"
                print(f"\n  {ra['label']}  vs  {rb['label']}")
                print(f"    AUROC:  {ra['auroc']:.4f}  vs  {rb['auroc']:.4f}  (Δ={delta:+.4f})")
                print(f"    p-value: {p:.4f}  →  {sig} at α=0.05")

    # ── Formatted table for paper ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PUBLICATION TABLE (copy-paste ready)")
    print("=" * 65)
    print(f"  {'Model':<38} {'AUROC':>7}  {'95% CI':>18}")
    print("  " + "─" * 62)
    for tag, r in results.items():
        ci_str = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        print(f"  {r['label']:<38} {r['auroc']:>7.4f}  {ci_str:>18}")
    print("=" * 65)


if __name__ == "__main__":
    main()
