from typing import Dict

# thresholds 
TH_GREEN = 0.78
TH_AMBER = 0.65

def decision_from_trust(trust: float) -> str:
    if trust >= TH_GREEN:
        return "green"
    if trust >= TH_AMBER:
        return "amber"
    return "red"

def mitigation_action(decision: str) -> Dict:
    if decision == "green":
        return {"action":"show_answer", "note":"confident"}
    if decision == "amber":
        return {
            "action":"show_answer_with_citation",
            "note":"moderate_confidence",
            "recommendation":"Please verify the cited sources."
        }
    # red
    return {
        "action":"fallback",
        "note":"low_confidence",
        "recommendation":"Provide routing-engine instruction only; ask user for clarification or re-query retrieval."
    }
