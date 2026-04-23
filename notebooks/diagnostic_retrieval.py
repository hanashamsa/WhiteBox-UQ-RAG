import json
import re
import string
from src.retrieval.retriever import Retriever

def normalize(text):
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text

def main():
    retriever = Retriever()
    
    # Use target K=15 for diagnostic as requested
    k = 15 
    
    test_queries = "data/test_queries.jsonl"
    
    total = 0
    covered = 0
    seen = set()
    
    with open(test_queries, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            query = obj["query"]
            gold = obj.get("gold_answer", "")
            
            if not gold:
                continue
                
            if query in seen:
                continue
            seen.add(query)
                
            norm_gold = normalize(gold)
            
            # Retrieve with lexcial reranking
            results = retriever.retrieve(query, k=k, rerank=True)
            
            # Check coverage
            found = False
            for r in results:
                norm_text = normalize(r["text"])
                if norm_gold in norm_text:
                    found = True
                    break
                    
            if found:
                covered += 1
            total += 1
            
            if total % 100 == 0:
                print(f"Processed {total}, Cover: {covered}/{total} ({(covered/total)*100:.1f}%)")
                
    print(f"\nFINAL: K={k}")
    print(f"Total: {total}")
    print(f"Covered: {covered}")
    print(f"Coverage %: {(covered/total)*100:.2f}%")
    
if __name__ == "__main__":
    main()
