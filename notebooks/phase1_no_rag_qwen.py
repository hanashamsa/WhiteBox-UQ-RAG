"""
PHASE 1 — QA Metrics (Qwen, NO retrieval — ablation)
=====================================================
Run from project root: python notebooks/phase1_no_rag_qwen.py

Identical to phase1_qa_metrics_qwen.py EXCEPT:
  - No FAISS retrieval
  - Prompt contains NO context — pure LM answering
  - Saves to data/phase1_results_qwen_norag.jsonl

Purpose: ablation to determine whether the white-box UQ signal
(avg_answer_logprob → AUROC) is intrinsic to the LM or boosted
by retrieval context.

After running: python notebooks/run_analytics.py --model-tag qwen_norag
               python notebooks/compare_models.py
"""
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── FROZEN CONFIG ─────────────────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 64
MAX_SAMPLES    = 1000
F1_THRESHOLD   = 0.8
QUERIES_PATH   = "data/squad_validation_queries.jsonl"
OUT_PATH       = "data/phase1_results_qwen_norag.jsonl"
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


def build_messages_no_context(query: str) -> list:
    """No retrieval context — pure LM answering."""
    return [
        {
            "role": "system",
            "content": (
                "Answer the question with a short factual answer. "
                "Return ONLY the answer span. Do not explain."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {query}\nAnswer:",
        },
    ]


def load_queries(path: str, max_n: int) -> list:
    queries, seen = [], set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["query_id"] not in seen:
                seen.add(obj["query_id"])
                queries.append(obj)
            if len(queries) >= max_n:
                break
    return queries


def main():
    print("=" * 60)
    print("PHASE 1 — ABLATION: Qwen NO RAG (pure LM)")
    print("Model  :", MODEL_NAME)
    print("Samples:", MAX_SAMPLES)
    print("Context: NONE (ablation)")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading Qwen model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model on: {device}")

    # Load queries
    print("\n[2/3] Loading queries...")
    queries = load_queries(QUERIES_PATH, MAX_SAMPLES)
    print(f"  {len(queries)} queries")

    # Generation
    print("\n[3/3] Running generation (no retrieval)...")
    if Path(OUT_PATH).exists():
        Path(OUT_PATH).unlink()

    total, n_em, n_correct = 0, 0, 0
    all_f1 = []
    t0     = time.time()

    for i, qobj in enumerate(queries, 1):
        query    = qobj["query"]
        gold     = qobj["gold_answer"]
        query_id = qobj["query_id"]

        messages = build_messages_no_context(query)
        text     = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs   = tokenizer(text, return_tensors="pt").to(model.device)

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

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids    = out.sequences[0][prompt_len:]
        gen_text   = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        gen_text   = gen_text.split("\n")[0].strip()

        token_logprobs = []
        for t, logits_t in enumerate(out.scores):
            lp     = torch.log_softmax(logits_t[0], dim=-1)
            chosen = gen_ids[t].item()
            token_logprobs.append(float(lp[chosen].item()))

        avg_lp = sum(token_logprobs) / len(token_logprobs) if token_logprobs else None
        em     = exact_match(gen_text, gold)
        f1     = f1_score(gen_text, gold)
        corr   = 1 if f1 >= F1_THRESHOLD else 0

        n_em      += em
        n_correct += corr
        all_f1.append(f1)
        total     += 1

        record = {
            "query_id":           query_id,
            "query":              query,
            "gold_answer":        gold,
            "generated_text":     gen_text,
            "retrieved":          [],           # empty — no retrieval
            "token_logprobs":     token_logprobs,
            "avg_answer_logprob": avg_lp,
            "exact_match":        em,
            "f1":                 f1,
            "label":              corr,
            "condition":          "no_rag",     # ablation label
        }
        with open(OUT_PATH, "a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        if i % 100 == 0:
            elapsed = time.time() - t0
            mf1     = sum(all_f1) / total
            acc     = n_correct / total
            print(f"  [{i:4d}/{MAX_SAMPLES}]  EM={n_em/total:.3f}  "
                  f"MeanF1={mf1:.3f}  Acc(F1≥0.8)={acc:.3f}  "
                  f"elapsed={elapsed/60:.1f}min")

    mean_f1 = sum(all_f1) / total
    print("\n" + "=" * 60)
    print("ABLATION RESULTS (Qwen, No RAG)")
    print(f"  Samples       : {total}")
    print(f"  Exact Match   : {n_em/total:.4f}")
    print(f"  Mean F1       : {mean_f1:.4f}")
    print(f"  Acc (F1≥0.8)  : {n_correct/total:.4f}")
    print(f"  Output        : {OUT_PATH}")
    print("=" * 60)
    print("\n  Compare RAG vs no-RAG:")
    print("  python notebooks/run_analytics.py --model-tag qwen_norag")
    print("  python notebooks/bootstrap_ci.py")
    print("  python notebooks/risk_coverage.py")
    print("  python notebooks/compare_models.py")


if __name__ == "__main__":
    main()
