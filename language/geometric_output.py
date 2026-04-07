"""
language/geometric_output.py
==============================
Geometric Output Pipeline — Inverse Input, Parity-Locked Generation

THE CORE PRINCIPLE
──────────────────
Input:  text → symbols → radial positions → tension → load-bearers → carry → field
Output: field → load-bearers → radial positions → symbols → words → text

The same invariants govern both directions. A generated output is
geometrically valid when running it back through the input pipeline
produces a field state that aligns with the field that generated it.
That alignment IS the parity lock — the system verifying its own output
by the same standard it used to process the input.

THE PIPELINE (six stages)
──────────────────────────

STAGE 1 — FIELD READING
  Read the current field configuration:
    - Mersenne band differential (positive vs negative string tension)
    - Active load-bearers from the last processed sentence
    - Net carry direction and magnitude
    - Resolution score from fold line

  This gives us the target geometric state the output must express.

STAGE 2 — TARGET REGION IDENTIFICATION
  Map the field state to a target region in the dual-13 space.

  High positive differential → target the positive arc [0, π)
  High negative differential → target the negative arc (π, 2π)
  Near-zero differential     → target the boundary zone (near π)

  The target region is expressed as a signed value range in dual-13
  space — e.g. "prefer symbols with net_signed between +2 and +8".

  The resolution score gates confidence: low resolution means the
  target region is wide (uncertain), high resolution narrows it.

STAGE 3 — VOCABULARY SAMPLING
  Walk the session vocabulary to find candidate words whose symbol
  stream maps to the target radial region.

  A word "fits" the target when:
    - Its net_signed value falls in the target range
    - Its dominant symbol group is geometrically compatible
    - Named invariants get priority — they're the most verified

  Priority order:
    1. Named invariants in the target range (highest trust)
    2. Stable vocabulary words in the target range
    3. Any word in the target range (lowest trust, flagged uncertain)

STAGE 4 — SEQUENCE ASSEMBLY
  Assemble candidate words into a coherent sequence.

  Rules derived from the same geometry:
    - Start with the highest-magnitude load-bearer in the target range
      (the word the field is most strongly expressing)
    - Follow with words that are geometrically compatible
      (their symbol streams don't introduce opposing tension)
    - Insert structural anchors (the, is, and) where the symbol
      stream needs a zero-boundary separator — mirrors how 0 works
      as a pocket in the input pipeline
    - Sequence length is determined by field persistence:
      persistence ≈ 1.0 → full sentence
      persistence ≈ 0.5 → fragment
      persistence < 0.3 → single word only

STAGE 5 — PARITY VERIFICATION
  Run the assembled sequence back through the input pipeline
  (SymbolicWave → tension profile → load-bearers → carry direction).

  Compute alignment between:
    - The carry direction of the generated sequence
    - The carry direction that the input sentence produced

  If alignment >= _PARITY_THRESHOLD: output is parity-locked (verified)
  If alignment < _PARITY_THRESHOLD:  output is flagged as approximate

STAGE 6 — OUTPUT EMISSION
  Emit the verified sequence with parity status attached.
  The answer header shows: recognition/reconstruction + parity status.
  Parity-locked outputs are emitted cleanly.
  Approximate outputs carry a geometric confidence note.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from core.invariants import invariants
from utils.symbol_grouping import symbol_to_signed, symbol_grouping
from utils.bipolar_lattice import bipolar_lattice
from utils.fold_line_resonance import fold_line_resonance
from wave.symbolic_wave import SymbolicWave

# ── Constants ─────────────────────────────────────────────────────────────────
_PARITY_THRESHOLD    = 0.35   # minimum alignment for parity lock
_CONTENT_THRESHOLD   = 1.5    # |net_signed| for content word classification
_STRUCTURAL_ANCHORS  = {
    "the", "and", "is", "in", "of", "a", "to", "it", "as",
    "that", "this", "was", "be", "are", "for", "on", "or",
    "from", "by", "at", "an", "not", "but", "so", "if",
    "its", "has", "had", "have", "with", "how", "what",
}

# Sentence structural templates — minimal scaffolding that lets content
# words carry meaning without imposing external semantic structure.
# Templates are expressed as (slot_type, ...) tuples:
#   'anchor'  = structural anchor word
#   'content' = content word from target range
#   'high'    = highest-magnitude content word (primary load-bearer)
_TEMPLATES = {
    "statement":   ["high", "anchor", "content"],
    "elaboration": ["high", "anchor", "content", "anchor", "content"],
    "fragment":    ["high", "content"],
    "single":      ["high"],
}


class GeometricOutput:
    """
    Inverse input pipeline — reads field state and generates parity-locked output.
    """

    def __init__(self):
        self._sw = SymbolicWave()

    # ── Stage 1: Field Reading ────────────────────────────────────────────────

    def _read_field(self) -> Dict[str, Any]:
        """
        Read the current field configuration from active components.

        Returns a unified field state dict with:
          polarity      : float [-1, +1] — Mersenne differential
          resolution    : float [0, 1]   — fold line resolution score
          carry         : float          — net carry magnitude
          carry_sign    : +1 / -1 / 0   — carry direction
          persistence   : float          — last waveform persistence
        """
        # Mersenne band differential — content polarity in the lattice
        pos_tension = sum(
            s.tension for s in bipolar_lattice.strings
            if s.active and s.polarity > 0
        )
        neg_tension = sum(
            abs(s.tension) for s in bipolar_lattice.strings
            if s.active and s.polarity < 0
        )
        n_active    = max(1, sum(1 for s in bipolar_lattice.strings if s.active))
        differential = (pos_tension - neg_tension) / (n_active * 0.5)
        polarity     = float(np.clip(differential, -1.0, 1.0))

        resolution  = fold_line_resonance.get_resolution_score()
        field_state = fold_line_resonance._field_persistence

        return {
            "polarity":    polarity,
            "resolution":  resolution,
            "persistence": field_state,
            "carry":       fold_line_resonance._field_carry,
            "carry_sign":  int(math.copysign(1, fold_line_resonance._field_carry))
                           if fold_line_resonance._field_carry != 0.0 else 0,
        }

    # ── Stage 2: Target Region Identification ────────────────────────────────

    def _identify_target_region(
        self,
        field: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Map field state to a target range in the dual-13 space.

        Resolution gates the width of the target window:
          resolution = 1.0 → tight window (±2 around centre)
          resolution = 0.5 → medium window (±4)
          resolution < 0.3 → wide window (full half-space)
        """
        polarity   = field["polarity"]
        resolution = field["resolution"]

        # Window width — tighter when more resolved
        window = int(np.clip(8 * (1.0 - resolution) + 2, 2, 8))

        if polarity > 0.1:
            # Positive field → target positive arc
            centre    = int(np.clip(round(polarity * 9), 1, 13))
            low, high = max(1, centre - window), min(13, centre + window)
            side      = "positive"
        elif polarity < -0.1:
            # Negative field → target negative arc
            centre    = int(np.clip(round(abs(polarity) * 9), 1, 13))
            low, high = -min(13, centre + window), -max(1, centre - window)
            side      = "negative"
        else:
            # Near-zero differential → boundary zone
            low, high = -3, 3
            side      = "boundary"

        return {
            "side":       side,
            "low":        low,
            "high":       high,
            "centre":     polarity * 9,
            "window":     window,
            "polarity":   polarity,
        }

    # ── Stage 3: Vocabulary Sampling ─────────────────────────────────────────

    def _sample_vocabulary(
        self,
        target:          Dict[str, Any],
        vocabulary:      Any,            # SessionVocabulary instance
        invariant_engine: Any,
        fingerprint:     Dict[str, Any],
        n_candidates:    int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Find vocabulary words whose net_signed falls in the target region.

        Returns candidates sorted by: named > stable > net_signed magnitude.
        Includes the active load-bearers from the current fingerprint as
        highest-priority candidates — they're what the field is already
        expressing.
        """
        low, high = target["low"], target["high"]
        candidates = []

        # Priority 1: current sentence load-bearers in target range
        per_word = fingerprint.get("per_word", [])
        for w in per_word:
            ns  = w.get("net_signed", 0.0)
            word = w.get("word", "").rstrip(".!?,;:")
            if not word or word.lower() in _STRUCTURAL_ANCHORS:
                continue
            if low <= ns <= high and abs(ns) >= _CONTENT_THRESHOLD:
                candidates.append({
                    "word":       word,
                    "net_signed": ns,
                    "source":     "load_bearer",
                    "priority":   3,
                    "named":      invariant_engine.is_named(word),
                })

        # Priority 2: named invariants in target range
        for word_key, data in invariant_engine.named_invariants.items():
            word = data.get("word", "")
            if not word or word.lower() in _STRUCTURAL_ANCHORS:
                continue
            # Compute net_signed from word's symbol stream
            stream = [self._sw._token_to_27_symbol(c)
                      for c in word if c and not c.isspace()]
            ns = sum(symbol_to_signed(s) / 13.0 for s in stream if s != '0')
            if low <= ns <= high and abs(ns) >= _CONTENT_THRESHOLD:
                candidates.append({
                    "word":       word,
                    "net_signed": ns,
                    "source":     "named_invariant",
                    "priority":   2,
                    "named":      True,
                })

        # Priority 3: stable vocabulary in target range
        stable = vocabulary.get_stable_words() if hasattr(vocabulary, 'get_stable_words') else []
        for entry in stable:
            word = entry.get("word", "")
            if not word or word.lower() in _STRUCTURAL_ANCHORS:
                continue
            ns = entry.get("net_signed", 0.0)
            if low <= ns <= high and abs(ns) >= _CONTENT_THRESHOLD:
                candidates.append({
                    "word":       word,
                    "net_signed": ns,
                    "source":     "stable_vocab",
                    "priority":   1,
                    "named":      invariant_engine.is_named(word),
                })

        # Deduplicate by word, keep highest priority
        seen     = {}
        for c in candidates:
            w = c["word"].lower()
            if w not in seen or c["priority"] > seen[w]["priority"]:
                seen[w] = c
        candidates = sorted(
            seen.values(),
            key=lambda c: (c["priority"], abs(c["net_signed"])),
            reverse=True
        )

        return candidates[:n_candidates]

    # ── Stage 4: Sequence Assembly ────────────────────────────────────────────

    def _assemble_sequence(
        self,
        candidates:      List[Dict[str, Any]],
        field:           Dict[str, Any],
        target:          Dict[str, Any],
        vocabulary:      Any,
    ) -> Tuple[str, str]:
        """
        Assemble candidates into an output sequence.

        Template selection based on persistence:
          >= 0.8 → elaboration (full sentence)
          >= 0.5 → statement
          >= 0.3 → fragment
          <  0.3 → single word

        Structural anchors are inserted at zero-boundary positions —
        mirroring how '0' pockets work in the input pipeline.

        Returns (assembled_text, template_used).
        """
        persistence = field["persistence"]
        side        = target["side"]

        if persistence >= 0.8:
            template_key = "elaboration"
        elif persistence >= 0.5:
            template_key = "statement"
        elif persistence >= 0.3:
            template_key = "fragment"
        else:
            template_key = "single"

        template = _TEMPLATES[template_key]

        if not candidates:
            return ("Field geometry unresolved — no vocabulary in target region.",
                    "fallback")

        # Select anchor words from structural set that are in vocabulary
        anchors = []
        if hasattr(vocabulary, 'get_stable_words'):
            for entry in vocabulary.get_stable_words():
                if entry.get("word", "").lower() in _STRUCTURAL_ANCHORS:
                    anchors.append(entry["word"])
        if not anchors:
            anchors = ["is", "and", "the"]

        # Fill template slots
        words     = []
        content_i = 0   # index into candidates
        anchor_i  = 0   # index into anchors

        for slot in template:
            if slot == "high":
                # Highest magnitude content word
                if candidates:
                    words.append(candidates[0]["word"])
            elif slot == "content":
                # Next content candidate
                if content_i + 1 < len(candidates):
                    content_i += 1
                    words.append(candidates[content_i]["word"])
                elif candidates:
                    words.append(candidates[0]["word"])
            elif slot == "anchor":
                # Structural anchor — zero boundary separator
                if anchors:
                    words.append(anchors[anchor_i % len(anchors)])
                    anchor_i += 1

        # Join with spaces, add terminal punctuation based on field polarity
        if side == "boundary":
            text = " ".join(words) + "?"
        else:
            text = " ".join(words) + "."

        return text, template_key

    # ── Stage 5: Parity Verification ─────────────────────────────────────────

    def _verify_parity(
        self,
        generated_text:   str,
        input_carry_sign: int,
    ) -> Tuple[float, bool]:
        """
        Run generated text back through the input pipeline and check
        whether its carry direction aligns with the input that produced
        the current field state.

        Returns (alignment_score, is_locked).
        """
        if not generated_text or generated_text.startswith("Field geometry"):
            return 0.0, False

        # Encode through symbolic wave
        tri = self._sw.triangulate(generated_text)
        per_word = tri.get("symbol_stream", [])

        # Compute net_signed sum of content words
        gen_content_sum = 0.0
        for sym in per_word:
            if sym != '0':
                gen_content_sum += symbol_to_signed(sym) / 13.0

        if abs(gen_content_sum) < 1e-4:
            return 0.5, False   # neutral — not confirmed but not opposed

        gen_sign = math.copysign(1.0, gen_content_sum)

        if input_carry_sign == 0:
            return 0.5, False   # no carry to compare against

        alignment = gen_sign * input_carry_sign
        is_locked = alignment >= _PARITY_THRESHOLD

        return float(alignment), is_locked

    # ── Stage 6: Emission ─────────────────────────────────────────────────────

    def generate(
        self,
        fingerprint:      Dict[str, Any],
        vocabulary:       Any,
        invariant_engine: Any,
        consensus:        float,
        persistence:      float,
    ) -> Dict[str, Any]:
        """
        Full six-stage geometric output pipeline.

        Returns a dict with:
          text          : the generated output string
          parity_locked : bool — whether parity verification passed
          alignment     : float — parity alignment score
          template      : str — which template was used
          field_polarity: float — what polarity the field was expressing
          target_region : dict — the dual-13 target window
          confidence    : str — 'high' / 'medium' / 'low'
        """
        # Stage 1
        field = self._read_field()
        field["persistence"] = persistence   # use actual pipeline persistence

        # Stage 2
        target = self._identify_target_region(field)

        # Stage 3
        candidates = self._sample_vocabulary(
            target, vocabulary, invariant_engine, fingerprint
        )

        # Stage 4
        text, template = self._assemble_sequence(
            candidates, field, target, vocabulary
        )

        # Stage 5
        carry_sign = field["carry_sign"]
        alignment, is_locked = self._verify_parity(text, carry_sign)

        # Confidence tier
        if is_locked and field["resolution"] >= 0.7:
            confidence = "high"
        elif alignment >= 0.0:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "text":           text,
            "parity_locked":  is_locked,
            "alignment":      round(alignment, 4),
            "template":       template,
            "field_polarity": round(field["polarity"], 4),
            "target_region":  target,
            "candidates":     [c["word"] for c in candidates],
            "confidence":     confidence,
            "resolution":     field["resolution"],
        }

    def format_output(self, result: Dict[str, Any]) -> str:
        """
        Format the generation result for display.

        Parity-locked outputs are shown cleanly.
        Approximate outputs carry a geometric confidence note.
        """
        text       = result["text"]
        locked     = result["parity_locked"]
        alignment  = result["alignment"]
        confidence = result["confidence"]
        resolution = result["resolution"]
        polarity   = result["field_polarity"]
        candidates = result.get("candidates", [])

        if locked and confidence == "high":
            return text

        if locked:
            return (f"{text} "
                    f"[parity confirmed, alignment {alignment:+.3f}]")

        if confidence == "medium":
            return (f"{text} "
                    f"[geometric approximation — "
                    f"field polarity {polarity:+.3f}, "
                    f"resolution {resolution:.3f}]")

        # Low confidence — surface the geometric state honestly
        candidate_str = ", ".join(candidates[:3]) if candidates else "none"
        return (f"Field geometry active but parity unconfirmed. "
                f"Strongest candidates: {candidate_str}. "
                f"Polarity {polarity:+.3f}, resolution {resolution:.3f}.")


# Singleton
geometric_output = GeometricOutput()
