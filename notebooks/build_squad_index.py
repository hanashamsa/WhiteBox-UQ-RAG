"""
Build a FAISS index from squad_corpus.jsonl.
Run from project root: python notebooks/build_squad_index.py

Outputs:
  data/squad_faiss_index.idx
  data/squad_meta.json

Does NOT overwrite the existing faiss_index.idx / meta.json.
"""
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_PATH  = "data/squad_corpus.jsonl"
INDEX_PATH   = "data/squad_faiss_index.idx"
META_PATH    = "data/squad_meta.json"
EMBED_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE   = 256

def main():
    print("Loading corpus from", CORPUS_PATH)
    docs = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"  {len(docs)} passages loaded.")

    print("Embedding with", EMBED_MODEL)
    model  = SentenceTransformer(EMBED_MODEL)
    texts  = [d["text"] for d in docs]
    embeds = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True,
                          convert_to_numpy=True, normalize_embeddings=True)

    dim   = embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeds.astype(np.float32))
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index written to {INDEX_PATH}  (ntotal={index.ntotal})")

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    print(f"Metadata written to {META_PATH}")

if __name__ == "__main__":
    main()
