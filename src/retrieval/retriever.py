import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:

    def __init__(
        self,
        index_path="data/faiss_index.idx",
        meta_path="data/meta.json",
        model_name="all-MiniLM-L6-v2"
    ):
        self.index = faiss.read_index(index_path)
        self.meta = json.load(open(meta_path))
        self.embedder = SentenceTransformer(model_name)

    # ---------------------------
    # Lexical Overlap Function
    # ---------------------------
    def lexical_overlap(self, query, doc_text):
        q_tokens = set(query.lower().split())
        d_tokens = set(doc_text.lower().split())

        if len(q_tokens) == 0:
            return 0.0

        return len(q_tokens & d_tokens) / len(q_tokens)

    # ---------------------------
    # Main Retrieve Method
    # ---------------------------
    def retrieve(self, query, k=20):

        # Step 1: Encode query
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        # Step 2: Dense retrieval (fetch large pool)
        TOP_K_FAISS = 100
        scores, indices = self.index.search(q_emb, TOP_K_FAISS)

        candidates = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            doc = self.meta[idx]
            candidates.append({
                "id": f"doc_{idx}",
                "text": doc["text"],
                "dense_score": float(score)
            })

        # Step 3: Hybrid reranking
        reranked = []

        for c in candidates:
            lex_score = self.lexical_overlap(query, c["text"])

            # Weighted fusion
            final_score = 0.7 * c["dense_score"] + 0.3 * lex_score

            reranked.append({
                "id": c["id"],
                "text": c["text"],
                "score": final_score
            })

        # Step 4: Sort by fused score
        reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)

        # Step 5: Return top-k
        return reranked[:k]