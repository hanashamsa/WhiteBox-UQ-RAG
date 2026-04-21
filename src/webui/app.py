

import gradio as gr
import os, json, math
import numpy as np
import torch
from collections import Counter

from src.retrieval.retriever import Retriever
from src.models.whitebox_adapter import generate_and_extract
from src.mitigation.policy import mitigation_action

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


STATS = {
    "r_min": 0.0,
    "r_max": 1.0,
    "entropy_min": 0.0,
    "entropy_max": 3.0
}

if os.path.exists("data/stats.json"):
    STATS.update(json.load(open("data/stats.json","r")))

def normalize(x, lo, hi):
    if hi <= lo:
        return 0.5
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

retriever = Retriever()


def detect_repetition(text):
    words = text.lower().split()
    return any(words.count(w) >= 5 for w in set(words))


def handle_query(query):

    # -------- Retrieval --------
    retrieved = retriever.retrieve(query, k=3)
    R_raw = max([r["score"] for r in retrieved]) if retrieved else 0.0
    R_norm = normalize(R_raw, STATS["r_min"], STATS["r_max"])

    # -------- Prompt --------
    prompt = "You are an assistant that explains if a route is impacted by notices.\nContext:\n"
    for r in retrieved:
        prompt += f"[DOC:{r['id']} | score={r['score']:.3f}] {r['text']}\n"
    prompt += f"\nQuestion:\n{query}\nAnswer concisely:"

    # -------- Generation --------
    gen = generate_and_extract(MODEL_NAME, prompt, device=DEVICE, max_new_tokens=120)
    text = gen.get("text","").strip()
    mean_entropy = gen.get("mean_entropy")

    entropy_norm = normalize(
        mean_entropy if mean_entropy is not None else STATS["entropy_max"],
        STATS["entropy_min"],
        STATS["entropy_max"]
    )

    # generation confidence 
    gen_conf = math.exp(-entropy_norm)  

    # -------- Fusion --------
    fusion = 0.65 * R_norm + 0.35 * gen_conf
    trust = sigmoid(5 * (fusion - 0.5))

    # -------- Decision --------
    if trust >= 0.75:
        decision = "green"
        note = "confident"
    elif trust >= 0.55:
        decision = "amber"
        note = "moderate confidence"
    else:
        decision = "red"
        note = "low confidence"

    action = mitigation_action(decision)

    # -------- Diagnostics --------
    diag = "\n".join([
        f"Retrieval confidence (R_norm): {R_norm:.3f}",
        f"Mean token entropy: {mean_entropy}",
        f"Generation confidence: {gen_conf:.3f}",
        f"Fusion score: {fusion:.3f}",
        f"Final trust score: {trust:.3f}"
    ])

    return (
        text,
        f"{trust:.3f}",
        decision,
        note,
        action.get("recommendation",""),
        diag
    )


iface = gr.Interface(
    fn=handle_query,
    inputs=gr.Textbox(lines=2, placeholder="Ask about road closures…"),
    outputs=[
        gr.Textbox(label="Generated answer", lines=8),
        gr.Textbox(label="Trust score"),
        gr.Textbox(label="Decision"),
        gr.Textbox(label="Action note"),
        gr.Textbox(label="Recommendation"),
        gr.Textbox(label="Uncertainty & diagnostics", lines=10),
    ],
    title="RAG + White-Box Uncertainty",
    description="Retrieval aware generation with calibrated uncertainty and risk aware decisions."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", share=False, inbrowser=True)
