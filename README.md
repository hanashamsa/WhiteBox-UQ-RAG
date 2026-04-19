
# WhiteBox-UQ-RAG

White-box uncertainty quantification for retrieval-augmented question answering systems using token-level log-probabilities, retrieval signals, calibration, and cross-model evaluation.

---

## Overview

Large Language Models can generate fluent answers even when incorrect.  
This project studies whether internal model confidence signals can reliably predict answer correctness in Retrieval-Augmented Generation (RAG) systems.

The core focus is **white-box uncertainty quantification**:

- Access model logits during generation
- Compute answer-level confidence
- Compare confidence against actual correctness
- Evaluate whether confidence generalizes across models and retrieval settings

---

## Motivation

Reliable AI systems require more than accuracy.

They also need to know:

- When they are likely correct
- When they are uncertain
- When they should abstain
- Whether confidence remains useful across model sizes

This repository investigates whether average token log-probability is a practical confidence signal for QA systems.

---

## Research Hypothesis

Average answer token log-probability (`avg_answer_logprob`) is a reliable predictor of answer correctness in retrieval-augmented QA systems.

If true:

- Higher confidence answers should be correct more often
- Lower confidence answers should fail more often
- Confidence ranking should improve selective answering
- Signal should generalize across model sizes

---

## Why SQuAD QA Was Chosen

SQuAD provides:

- Standard extractive QA benchmark
- High-quality human-labeled answers
- Clear EM / F1 evaluation
- Strong baseline for uncertainty experiments
- Repeatable comparison across models

This makes it suitable for controlled confidence evaluation.

---

## Experimental Pipeline

```text
SQuAD Corpus
   ↓
SentenceTransformer Embeddings
   ↓
FAISS Retrieval Index
   ↓
Top-k Retrieved Context
   ↓
LLM Answer Generation
   ↓
Token Log-Probability Extraction
   ↓
Confidence Scoring
   ↓
Correctness Evaluation (EM / F1)
   ↓
AUROC / Calibration / Risk-Coverage
````


## Models Evaluated

### RAG Conditions

* `mistralai/Mistral-7B-Instruct-v0.3`
* `Qwen/Qwen2.5-0.5B-Instruct`

### No-RAG Ablation

* `Qwen/Qwen2.5-0.5B-Instruct`

---

## Dataset

### Stanford Question Answering Dataset (SQuAD)

Used validation split with generated retrieval corpus from contexts.

Approximate project artifacts:

* Unique corpus passages: 2067
* Validation QA pairs: 10k+

---

## Metrics

### QA Quality

* Exact Match (EM)
* Token F1
* Accuracy using `F1 ≥ 0.8`

### Confidence Quality

* AUROC
* AUPRC
* Expected Calibration Error (ECE)
* Risk-Coverage Curve
* AURC

---

## Key Results

| Condition        | Mean F1 | Accuracy (F1≥0.8) | AUROC (logprob) |
| ---------------- | ------- | ----------------- | --------------- |
| Mistral-7B + RAG | 0.641   | 0.564             | 0.773           |
| Qwen-0.5B + RAG  | 0.423   | 0.360             | 0.747           |
| Qwen-0.5B no-RAG | 0.068   | 0.023             | 0.732           |

---

## Main Findings

### 1. Internal Confidence Is Predictive

Average token log-probability strongly predicts correctness.

### 2. Signal Generalizes Across Model Sizes

A 0.5B model retains similar uncertainty ranking behavior relative to a 7B model.

### 3. Retrieval Helps Accuracy More Than Confidence

RAG greatly improves QA accuracy, while AUROC remains relatively stable.

### 4. Selective Answering Improves Reliability

High-confidence subsets achieve substantially higher accuracy than full-set answering.

---

## Evidence Supporting Hypothesis

Observed AUROC:

* Mistral-7B: **0.773**
* Qwen-0.5B: **0.747**
* Qwen no-RAG: **0.732**

This supports the claim that model-internal likelihood contains usable correctness information.

---

## Repository Structure

```text
WhiteBox-UQ-RAG/
│── data/
│── notebooks/
│── src/
│── scripts/
│── outputs/
│── README.md
```

Suggested contents:

```text
data/        datasets, FAISS indexes, outputs
notebooks/   experiments and evaluation phases
src/         reusable modules
scripts/     utility scripts
outputs/     figures, tables, plots
```

---

## Installation

```bash
git clone https://github.com/yourname/WhiteBox-UQ-RAG.git
cd WhiteBox-UQ-RAG

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run the Project

### Build Retrieval Index

```bash
python notebooks/build_squad_index.py
```

### Run QA Experiments

```bash
python notebooks/phase1_qa_metrics.py
python notebooks/phase1_qa_metrics_qwen.py
python notebooks/phase1_no_rag_qwen.py
```

### Run Analytics

```bash
python notebooks/run_analytics.py --model-tag qwen
python notebooks/bootstrap_ci.py
python notebooks/risk_coverage.py
python notebooks/compare_models.py
```

---

## Sample Outputs

```text
AUROC (Mistral): 0.773
AUROC (Qwen):    0.747

Top-10% confidence accuracy:
Mistral: 0.918
Qwen:    0.743
```

---

## Limitations

* Limited to SQuAD-style QA
* Greedy decoding only
* Retrieval confidence uses simple similarity features
* Some experiments use partial Mistral sample counts due compute cost
* Calibration can vary by model family

---

## Future Improvements

* Multi-dataset validation (Natural Questions, TriviaQA)
* Larger model families
* Sampling-based uncertainty
* Better retrieval confidence features
* Conformal prediction / abstention policies
* Production API deployment

---

## Acknowledgments

* Hugging Face Transformers
* SentenceTransformers
* FAISS
* scikit-learn
* Stanford SQuAD Dataset

````

---

## Section 5: Structured Report Add-on

```markdown
# Research Report

## Abstract

This project evaluates whether token-level generation probabilities can serve as reliable uncertainty signals in retrieval-augmented question answering systems. Using SQuAD and two open-source LLM families, we study correctness prediction, calibration, and selective answering.

## Objective

Determine whether average token log-probability predicts answer correctness across models and retrieval settings.

## Method

1. Build FAISS retrieval index from SQuAD contexts  
2. Retrieve relevant passages  
3. Generate answers with open-source LLMs  
4. Extract token log-probabilities  
5. Compare confidence against correctness labels

## Experiments

- Mistral-7B with RAG
- Qwen-0.5B with RAG
- Qwen-0.5B without RAG
- Cross-model AUROC comparison
- Calibration and risk-coverage evaluation

## Findings

- Confidence AUROC remained strong across settings
- RAG significantly improved QA accuracy
- Confidence ranking remained stable even without retrieval
- High-confidence filtering increased practical reliability

## Conclusion

Model-internal token likelihood is a useful and transferable uncertainty signal for QA systems. This supports lightweight white-box trust mechanisms for deployable RAG pipelines.
````
