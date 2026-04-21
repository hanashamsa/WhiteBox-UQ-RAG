import torch
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators.perplexity import PerplexityEstimator
from lm_polygraph.estimators.entropy import EntropyEstimator
from lm_polygraph.estimators.semantic import SemanticDiversityEstimator
from lm_polygraph.ue_manager import UEManager  

# 1) wrap the HF model and tokenizer
def make_whitebox_model(model_path="Qwen/Qwen2.5-0.5B-Instruct", device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tok = AutoTokenizer.from_pretrained(model_path)
    wb = WhiteboxModel(base, tok, model_path=model_path)
    return wb

# 2) instantiate a set of estimators
def make_estimators():
    return {
      "perplexity": PerplexityEstimator(),
      "entropy": EntropyEstimator(),
      "semantic_div": SemanticDiversityEstimator()
    }

# 3) compute estimators for a generation object produced by WhiteboxModel.generate()
def compute_estimators_for_generation(wb_model, prompt, gen_kwargs=None):
    gen_kwargs = gen_kwargs or {"max_new_tokens":120}
    generation = wb_model.generate_text_with_scores(prompt, **gen_kwargs)
    
    estimators = make_estimators()
    results = {}
    for name, est in estimators.items():
        results[name] = est.estimate(generation)   
 
    ue = UEManager(estimators=list(estimators.values()))
    fused_score = ue.fuse(generation)  
    results["fused"] = fused_score
    return generation, results
