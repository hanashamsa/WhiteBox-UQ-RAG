import torch
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


# Global model cache 


_MODEL_CACHE = {}


# Load model once (GPU optimized)


def load_hf_model(model_name: str, device="cuda"):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    print(f"Loading model: {model_name}")

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    _MODEL_CACHE[model_name] = (model, tokenizer)
    return model, tokenizer


# Softmax


def softmax(x):
    e = torch.exp(x - x.max(-1, keepdim=True).values)
    return e / e.sum(-1, keepdim=True)


# Token entropy


def compute_token_entropy(logits):
    p = softmax(logits)
    return float(-(p * (p + 1e-20).log()).sum().item())


# Main generation + extraction


def generate_and_extract(model_name: str, prompt: str, device="cuda", max_new_tokens=20):

    model, tokenizer = load_hf_model(model_name, device)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    sequences = out.sequences
    scores = out.scores

    gen_ids = sequences[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)



    gen_text = gen_text.strip()

    # keep only first line
    if "\n" in gen_text:
        gen_text = gen_text.split("\n")[0]

    # remove "Exact answer:" if model adds it
    if "Exact answer:" in gen_text:
        gen_text = gen_text.split("Exact answer:")[-1].strip()

    # remove trailing explanation markers
    if "Explanation:" in gen_text:
        gen_text = gen_text.split("Explanation:")[0].strip()

    entropies = []
    top1_probs = []

    for logits in scores:
        logits = logits[0]
        p = softmax(logits)
        entropies.append(compute_token_entropy(logits))
        top1_probs.append(float(p.max().item()))

    if top1_probs:
        log_probs = [math.log(max(1e-12, p)) for p in top1_probs]
        mean_log_prob = float(sum(log_probs) / len(log_probs))
        geom_mean_top1 = float(math.exp(mean_log_prob))
        mean_entropy = float(sum(entropies) / len(entropies))
        perplexity = float(math.exp(-mean_log_prob))
    else:
        geom_mean_top1 = None
        mean_entropy = None
        mean_log_prob = None
        perplexity = None

    return {
        "text": gen_text,
        "entropies": entropies,
        "top1_probs": top1_probs,
        "geom_mean_top1": geom_mean_top1,
        "mean_entropy": mean_entropy,
        "poly_mean_log_prob": mean_log_prob,
        "poly_perplexity": perplexity
    }
