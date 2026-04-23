import json
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


BOOTSTRAP_SAMPLES = 1000
RANDOM_SEED = 42


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def normalize(x):
    x = np.array(x, dtype=float)
    if x.max() == x.min():
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


def compute_scores(records, mode):
    retrieval = [max([r["score"] for r in obj["retrieved"]]) for obj in records]

    entropy = [
        obj.get("generation", {}).get("mean_entropy")
        for obj in records
    ]

    perplexity = [
        obj.get("generation", {}).get("poly_perplexity")
        for obj in records
    ]

    if mode == "retrieval_only":
        return normalize(retrieval)

    if mode == "whitebox_only":
        ent = [e if e is not None else max(filter(None, entropy)) for e in entropy]
        return 1 - normalize(ent)

    if mode == "blackbox_only":
        ppl = [p if p is not None else max(filter(None, perplexity)) for p in perplexity]
        return 1 - normalize(ppl)

    if mode == "fused_whitebox":
        ent = [e if e is not None else max(filter(None, entropy)) for e in entropy]
        r = normalize(retrieval)
        g = 1 - normalize(ent)
        return normalize(0.5 * r + 0.5 * g)

    if mode == "fused_blackbox":
        ppl = [p if p is not None else max(filter(None, perplexity)) for p in perplexity]
        r = normalize(retrieval)
        g = 1 - normalize(ppl)
        return normalize(0.5 * r + 0.5 * g)

    raise ValueError("Unknown mode")


def compute_ece(y_true, y_prob, bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bin_edges = np.linspace(0, 1, bins + 1)
    ece = 0.0

    for i in range(bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += abs(acc - conf) * mask.mean()

    return ece


def bootstrap_ci(y_true, y_scores, metric_fn):
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y_true)
    stats = []

    for _ in range(BOOTSTRAP_SAMPLES):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(np.array(y_true)[idx], np.array(y_scores)[idx]))

    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return lower, upper


def evaluate(records, mode):
    y_true = [obj["label"] for obj in records]
    y_scores = compute_scores(records, mode)

    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    ece = compute_ece(y_true, y_scores)

    auroc_ci = bootstrap_ci(y_true, y_scores, roc_auc_score)
    auprc_ci = bootstrap_ci(y_true, y_scores, average_precision_score)

    return auroc, auroc_ci, auprc, auprc_ci, ece


def main(path):
    records = load_jsonl(path)

    print("\nEvaluating:", path)
    print("Total samples:", len(records))
    print("\nMode | AUROC (95% CI) | AUPRC (95% CI) | ECE")
    print("----------------------------------------------------------------")

    modes = [
        "retrieval_only",
        "whitebox_only",
        "blackbox_only",
        "fused_whitebox",
        "fused_blackbox",
    ]

    for m in modes:
        try:
            auroc, auroc_ci, auprc, auprc_ci, ece = evaluate(records, m)
            print(f"{m:15s} | "
                  f"{auroc:.3f} [{auroc_ci[0]:.3f},{auroc_ci[1]:.3f}] | "
                  f"{auprc:.3f} [{auprc_ci[0]:.3f},{auprc_ci[1]:.3f}] | "
                  f"{ece:.3f}")
        except Exception:
            pass


if __name__ == "__main__":
    main(sys.argv[1])