from typing import Dict, Any

class Safeguards:
    """
    Anti-grandiosity & Inversion Safeguard.
    """
    def __init__(self):
        self.reversal_count = 0
        self.max_local_clarity = 0.0

    def check_re_derivation(self, current_data: Dict[str, Any], previous_data: Dict[str, Any]) -> bool:
        if not previous_data or not current_data:
            return True
        n = current_data.get("n_original", 0)
        width = current_data.get("width", 0)
        height = current_data.get("height", 0)
        return abs(width * height - n) < n * 0.15

    def apply_reversal_trigger(self, clarity_score: float, strength: float = 0.08) -> float:
        self.reversal_count += 1
        if clarity_score > self.max_local_clarity * 1.25:
            self.max_local_clarity = clarity_score
            return clarity_score * (1 - strength)
        return clarity_score

    def should_force_reversal(self, persistence: float, iterations: int) -> bool:
        if iterations > 20 and persistence > 0.92:
            return True
        return False

    def is_mimic(self, prompt: str, answer: str) -> bool:
        """Pure radial mimic detection — high overlap with prompt triggers inversion."""
        if not prompt or not answer:
            return False
        p = prompt.lower().strip()
        a = answer.lower().strip()
        if len(a) == 0:
            return False
        overlap = len(set(p.split()) & set(a.split())) / len(a.split())
        return overlap > 0.85  # high literal overlap = mimic

    def get_status(self) -> Dict[str, Any]:
        return {
            "reversal_count": self.reversal_count,
            "max_local_clarity": round(self.max_local_clarity, 4)
        }

safeguards = Safeguards()