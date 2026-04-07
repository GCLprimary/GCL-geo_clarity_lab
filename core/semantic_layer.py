from typing import Dict, Any, Optional, Tuple
import numpy as np


class SemanticLayer:
    """
    Semantic Layer — Geometric core only (v2, scaffolding stripped).

    What remains:
      - Pocket splitting: finds context/query boundary geometrically.
        Mirrors SymbolicWave._insert_pockets() exactly.
      - Pocket confidence: waveform amplitude deviation at zero-break
        indices. '0' symbol at radius=2.2 creates a geometric signature.
      - Sentence type detection.
      - Negation detection.

    What was removed:
      - Concept dictionary
      - Verb stem lookup and inflection tables
      - Subject/object extraction engine
      - Aggregation and exclusion reasoning
      - Known-concept fallback answers

    The field resolves from geometry. If it cannot it says so honestly.
    """

    def __init__(self):
        self.question_words = {
            "what", "who", "how", "why", "when", "where", "which",
            "does", "do", "is", "are", "can", "could", "should", "would"
        }
        self.negation_words = {
            "no", "not", "never", "doesn't", "don't", "isn't",
            "aren't", "cannot", "can't", "won't", "wouldn't"
        }

    def _split_context_and_query(self, prompt: str) -> Tuple[str, str]:
        prompt = prompt.strip()
        if '?' not in prompt:
            return ("", prompt)
        question_start = prompt.index('?')
        prefix         = prompt[:question_start]
        last_end       = max(prefix.rfind('.'), prefix.rfind('!'))
        if last_end == -1:
            return ("", prompt)
        return (prompt[:last_end + 1].strip(), prompt[last_end + 1:].strip())

    def compute_pocket_confidence(
        self,
        tri_data:    Dict[str, Any],
        prop_result: Dict[str, Any],
    ) -> float:
        zero_breaks = tri_data.get("zero_breaks", [])
        if not zero_breaks:
            return 0.0
        raw     = prop_result.get("waveform_sample", [])
        numeric = [x for x in raw if isinstance(x, (int, float))]
        if not numeric:
            return 0.0
        waveform    = np.array(numeric, dtype=float)
        n_wave      = len(waveform)
        global_mean = float(np.mean(np.abs(waveform)))
        if global_mean < 1e-8:
            return 0.0
        n_symbols  = tri_data.get("n_original", max(zero_breaks) + 1)
        n_steps    = prop_result.get("steps", n_wave)
        deviations = []
        for br in zero_breaks:
            wave_idx  = int((br / max(n_symbols, 1)) * n_steps)
            wave_idx  = min(wave_idx, n_wave - 1)
            local_amp = abs(float(waveform[wave_idx]))
            deviations.append(abs(local_amp - global_mean))
        if not deviations:
            return 0.0
        return float(np.clip(float(np.mean(deviations)) / global_mean, 0.0, 1.0))

    def extract_with_pocket_alignment(
        self,
        prompt:      str,
        tri_data:    Dict[str, Any],
        prop_result: Dict[str, Any],
    ) -> Tuple[Optional[str], float]:
        pocket_confidence = self.compute_pocket_confidence(tri_data, prop_result)
        return None, pocket_confidence

    def detect_sentence_type(self, prompt: str) -> str:
        if prompt.strip().endswith("?"):
            return "question"
        if any(w in prompt.lower().split() for w in self.question_words):
            return "question"
        return "statement"

    def has_negation(self, prompt: str) -> bool:
        return any(w in prompt.lower() for w in self.negation_words)

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode":             "geometric_only",
            "extraction":       "disabled",
            "pocket_confidence":"active",
            "question_words":   len(self.question_words),
        }


semantic_layer = SemanticLayer()