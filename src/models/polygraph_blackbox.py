import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM


def polygraph_generate(model_name, prompt, device="cpu", max_new_tokens=120):

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Separate generated tokens from prompt
    full_sequence = output.sequences
    generated_tokens = full_sequence[:, input_length:]

    # Decode only generated part
    generated_text = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True
    ).strip()

    # Compute log-probs for generated tokens
    log_probs = []

    if output.scores is not None and len(output.scores) > 0:
        for i, step_scores in enumerate(output.scores):
            probs = torch.log_softmax(step_scores, dim=-1)
            token_id = generated_tokens[0, i]
            log_probs.append(probs[0, token_id].item())

    if len(log_probs) > 0:
        mean_log_prob = sum(log_probs) / len(log_probs)
        perplexity = math.exp(-mean_log_prob)
        variance = float(torch.tensor(log_probs).var().item()) if len(log_probs) > 1 else 0.0
    else:
        mean_log_prob = None
        perplexity = None
        variance = None

    return {
        "text": generated_text,
        "poly_mean_log_prob": mean_log_prob,
        "poly_perplexity": perplexity,
        "poly_sample_variance": variance
    }