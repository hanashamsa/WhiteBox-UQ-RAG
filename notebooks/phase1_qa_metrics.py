"""
PHASE 1 — QA Metrics Only
=========================
Run from project root: python notebooks/phase1_qa_metrics.py

Measures Exact Match and Mean F1 for Mistral-7B-Instruct-v0.3 on
1000 SQuAD validation samples with retrieval context.

Also stores per-token log-probs of generated tokens so Phase 3 can
compute avg_answer_logprob WITHOUT re-running generation.

Outputs: data/phase1_results.jsonl
"""
import json
import math
import re
import string
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── FROZEN CONFIG ─────────────────────────────────────────────────────────────
MODEL_NAME      = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_NEW_TOKENS  = 64
TOP_K_RETRIEVAL = 3
MAX_SAMPLES     = 1000
F1_THRESHOLD    = 0.8
QUERIES_PATH    = "data/squad_validation_queries.jsonl"
INDEX_PATH      = "data/squad_faiss_index.idx"
META_PATH       = "data/squad_meta.json"
OUT_PATH        = "data/phase1_results.jsonl"
EMBED_MODEL     = "all-MiniLM-L6-v2"
# ───────────────────────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize(pred).split()
    gold_toks = normalize(gold).split()
    common    = Counter(pred_toks) & Counter(gold_toks)
    num_same  = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def build_prompt(query: str, retrieved: list) -> str:
    context = "".join(r["text"] + "\n\n" for r in retrieved)
    return (
        "[INST] Answer the question using ONLY the provided context.\n"
        "Return ONLY the exact answer span. Do not explain.\n\n"
        "Context:\n"
        f"{context}"
        f"Question: {query}\n"
        "Answer: [/INST]"
    )


def retrieve(embedder, index, meta, query: str, k: int = 3):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(q_emb.astype(np.float32), k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append({
            "id":    f"doc_{idx}",
            "text":  meta[idx]["text"],
            "score": float(score),
        })
    return results


def load_queries(path: str, max_n: int) -> list:
    queries = []
    seen_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["query_id"] not in seen_ids:
                seen_ids.add(obj["query_id"])
                queries.append(obj)
            if len(queries) >= max_n:
                break
    return queries


def main():
    print("=" * 60)
    print("PHASE 1 — QA METRICS")
    print("Model  :", MODEL_NAME)
    print("Samples:", MAX_SAMPLES)
    print("=" * 60)

    # ── Load retrieval assets ──────────────────────────────────────────────────
    print("\n[1/4] Loading retrieval assets...")
    embedder = SentenceTransformer(EMBED_MODEL)
    index    = faiss.read_index(INDEX_PATH)
    meta     = json.load(open(META_PATH, "r", encoding="utf-8"))
    print(f"  FAISS index loaded: {index.ntotal} vectors")

    # ── Load model ─────────────────────────────────────────────────────────────
    print("\n[2/4] Loading model...")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    print(f"  Model loaded on device: {device}")

    # ── Load queries ───────────────────────────────────────────────────────────
    print("\n[3/4] Loading queries...")
    queries = load_queries(QUERIES_PATH, MAX_SAMPLES)
    print(f"  {len(queries)} queries loaded.")

    # ── Generation loop ────────────────────────────────────────────────────────
    print("\n[4/4] Running generation...")
    if Path(OUT_PATH).exists():
        Path(OUT_PATH).unlink()

    total    = 0
    n_em     = 0
    all_f1   = []
    n_correct = 0
    t0 = time.time()

    for i, qobj in enumerate(queries, 1):
        query      = qobj["query"]
        gold       = qobj["gold_answer"]
        query_id   = qobj["query_id"]

        retrieved  = retrieve(embedder, index, meta, query, k=TOP_K_RETRIEVAL)
        prompt     = build_prompt(query, retrieved)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        # ── Extract generated tokens ───────────────────────────────────────────
        prompt_len  = inputs["input_ids"].shape[1]
        gen_ids     = out.sequences[0][prompt_len:]        # shape (T,)
        scores_list = out.scores                           # tuple of T tensors (1, V)

        gen_text    = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        # Keep first line only
        gen_text    = gen_text.split("\n")[0].strip()

        # ── Per-token log-prob of the chosen token ─────────────────────────────
        # scores[t] is the raw logits for step t, shape (1, vocab_size)
        # We take log_softmax and index by gen_ids[t]
        token_logprobs = []
        for t, logits_t in enumerate(scores_list):
            lp     = torch.log_softmax(logits_t[0], dim=-1)   # (vocab_size,)
            chosen = gen_ids[t].item()
            token_logprobs.append(float(lp[chosen].item()))

        avg_answer_logprob = (
            sum(token_logprobs) / len(token_logprobs) if token_logprobs else None
        )

        # ── Metrics ────────────────────────────────────────────────────────────
        em   = exact_match(gen_text, gold)
        f1   = f1_score(gen_text, gold)
        corr = 1 if f1 >= F1_THRESHOLD else 0

        n_em     += em
        all_f1.append(f1)
        n_correct += corr
        total     += 1

        # ── Save record ───────────────────────────────────────────────────────
        record = {
            "query_id":           query_id,
            "query":              query,
            "gold_answer":        gold,
            "generated_text":     gen_text,
            "retrieved":          retrieved,
            "token_logprobs":     token_logprobs,          # list[float]
            "avg_answer_logprob": avg_answer_logprob,      # float (mean log-prob)
            "exact_match":        em,
            "f1":                 f1,
            "label":              corr,                    # 1 if F1 >= 0.8
        }
        with open(OUT_PATH, "a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        # ── Progress ──────────────────────────────────────────────────────────
        if i % 50 == 0:
            elapsed = time.time() - t0
            mean_f1 = sum(all_f1) / total
            acc     = n_correct / total
            print(
                f"  [{i:4d}/{MAX_SAMPLES}]  EM={n_em/total:.3f}  "
                f"MeanF1={mean_f1:.3f}  Acc(F1≥0.8)={acc:.3f}  "
                f"elapsed={elapsed/60:.1f}min"
            )

    # ── Final summary ──────────────────────────────────────────────────────────
    mean_f1   = sum(all_f1) / total
    em_rate   = n_em / total
    acc_rate  = n_correct / total

    print("\n" + "=" * 60)
    print("PHASE 1 RESULTS")
    print(f"  Samples evaluated : {total}")
    print(f"  Exact Match       : {em_rate:.4f}  ({em_rate*100:.1f}%)")
    print(f"  Mean F1           : {mean_f1:.4f}  ({mean_f1*100:.1f}%)")
    print(f"  Accuracy (F1≥0.8) : {acc_rate:.4f}  ({acc_rate*100:.1f}%)")
    print(f"  Output saved to   : {OUT_PATH}")
    print("=" * 60)

    if mean_f1 < 0.15:
        print("\n⚠ WARNING: Mean F1 is very low. Check prompt format or generation.")
    else:
        print("\n✓ Phase 1 passed. Proceed to Phase 2.")


if __name__ == "__main__":
    main()
