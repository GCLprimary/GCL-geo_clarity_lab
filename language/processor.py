"""
language/processor.py
=====================
Language Processor — Geometric Fingerprint + Generation + Session Vocabulary

Pipeline for a natural language sentence:

  1. ENCODE
     SymbolicWave maps the sentence to a symbol stream.
     Word boundaries (spaces) become zero breaks automatically.
     Each word is its own radial expansion terminated by a boundary.

  2. FINGERPRINT
     stream_context() produces the tension profile across the stream.
     We extract:
       - per-word tension signatures  (mean tension of each word's symbols)
       - active group IDs             (which groups fired and how strongly)
       - dominant direction           (net positive/negative field lean)
       - boundary count               (how many word-breaks the field crossed)
       - peak pair                    (highest absolute tension adjacency)

  3. SESSION VOCABULARY
     A word is written to session vocab when it produces a stable pattern —
     defined as: same dominant group activation on two or more appearances.
     Stable words get a stored fingerprint. Future appearances check against
     that stored fingerprint and report familiarity score (0–1).

  4. GENERATION
     The fingerprint + familiarity signal feeds the existing generator.
     prop_result is enriched with fingerprint data so generation.py
     can use geometric field state rather than just waveform persistence.

  5. OUTPUT
     {
       sentence:        original string
       words:           per-word breakdown
       fingerprint:     full geometric summary
       vocabulary_hits: words recognised from prior session
       answer:          generated response
     }
"""

import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from wave.symbolic_wave import SymbolicWave
from wave.propagation import WavePropagator
from wave.vibration import VibrationPropagator
from utils.fold_line_resonance import fold_line_resonance
from utils.symbol_grouping import symbol_grouping, symbol_to_signed
from utils.bipolar_lattice import bipolar_lattice
from core.clarity_ratio import clarity_ratio
from core.invariants import invariants
from core.ouroboros_engine import ouroboros_engine
from observer.observer import MultiObserver
from wave.generation import generator
from language.invariant_engine import invariant_engine
from language.relational_tension import relational_tension
from language.geometric_output import geometric_output


# ── Vocabulary stability threshold ───────────────────────────────────────────
# A word needs this many consistent activations before it's considered stable
_VOCAB_STABILITY_THRESHOLD = 2
# Familiarity score threshold above which we report a vocabulary hit
_FAMILIARITY_THRESHOLD     = 0.65


class WordFingerprint:
    """
    Geometric fingerprint for a single word.

    Stores the tension profile, dominant group, net signed value,
    and the symbol stream that produced it.
    """

    def __init__(
        self,
        word:           str,
        symbol_stream:  List[str],
        tensions:       List[float],
        group_ids:      List[int],
        net_signed:     float,
    ):
        self.word          = word.lower()
        self.symbol_stream = symbol_stream
        self.tensions      = tensions
        self.mean_tension  = float(np.mean(tensions)) if tensions else 0.0
        self.group_ids     = group_ids
        self.dominant_group = max(set(group_ids), key=group_ids.count) if group_ids else -1
        self.net_signed    = net_signed
        self.timestamp     = time.time()
        self.appearances   = 1

    def similarity(self, other: "WordFingerprint") -> float:
        """
        Geometric similarity between two fingerprints of the same word.

        Uses three signals:
          1. Symbol stream overlap (same characters → same geometric addresses)
          2. Mean tension proximity (field response magnitude)
          3. Dominant group match (same structural region activated)

        Returns [0, 1] — 1.0 = identical geometric response.
        """
        # Symbol stream overlap
        s1 = set(self.symbol_stream)
        s2 = set(other.symbol_stream)
        sym_overlap = len(s1 & s2) / max(len(s1 | s2), 1)

        # Tension proximity — normalised by max possible tension (2.0)
        tension_diff = abs(self.mean_tension - other.mean_tension)
        tension_sim  = max(0.0, 1.0 - (tension_diff / 2.0))

        # Dominant group match
        group_match  = 1.0 if self.dominant_group == other.dominant_group else 0.0

        return round(0.4 * sym_overlap + 0.4 * tension_sim + 0.2 * group_match, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word":           self.word,
            "mean_tension":   round(self.mean_tension, 4),
            "dominant_group": self.dominant_group,
            "net_signed":     round(self.net_signed, 4),
            "appearances":    self.appearances,
        }


class SessionVocabulary:
    """
    In-session word memory.

    Words are stored after their first appearance. On subsequent appearances,
    the new fingerprint is compared to the stored one. If similarity exceeds
    _FAMILIARITY_THRESHOLD on two or more appearances, the word is marked
    stable and its fingerprint is updated to a rolling mean.
    """

    def __init__(self):
        self._store: Dict[str, WordFingerprint] = {}

    def lookup(self, word: str) -> Optional[WordFingerprint]:
        return self._store.get(word.lower())

    def update(self, fp: WordFingerprint) -> Tuple[float, bool]:
        """
        Add or update a word fingerprint.

        Returns (familiarity_score, is_stable).
          familiarity_score — how similar this appearance is to prior ones
          is_stable         — whether the word has crossed the stability threshold
        """
        word = fp.word
        existing = self._store.get(word)

        if existing is None:
            self._store[word] = fp
            return 0.0, False

        similarity = existing.similarity(fp)
        existing.appearances += 1

        # Update rolling mean tension
        n = existing.appearances
        existing.mean_tension = (existing.mean_tension * (n-1) + fp.mean_tension) / n

        # Update dominant group if new appearance agrees
        if fp.dominant_group == existing.dominant_group:
            pass  # stable — keep it
        elif existing.appearances <= 2:
            existing.dominant_group = fp.dominant_group  # too early to lock in

        is_stable = (existing.appearances >= _VOCAB_STABILITY_THRESHOLD
                     and similarity >= _FAMILIARITY_THRESHOLD)
        return similarity, is_stable

    def get_stable_words(self) -> List[Dict[str, Any]]:
        return [
            fp.to_dict() for fp in self._store.values()
            if fp.appearances >= _VOCAB_STABILITY_THRESHOLD
        ]

    def size(self) -> int:
        return len(self._store)

    def stable_count(self) -> int:
        return sum(1 for fp in self._store.values()
                   if fp.appearances >= _VOCAB_STABILITY_THRESHOLD)


class LanguageProcessor:
    """
    Full language processing pipeline.

    process(sentence) → {fingerprint, vocabulary_hits, answer}
    """

    def __init__(self):
        self.vocabulary     = SessionVocabulary()
        self.sw             = SymbolicWave()
        self._process_count = 0

    # ── Word-level fingerprinting ─────────────────────────────────────────────

    def _fingerprint_word(self, word: str) -> WordFingerprint:
        """
        Produce a geometric fingerprint for a single word.

        Maps word → symbol stream (no zero breaks — word is one unit),
        computes pair tensions across the stream, extracts group activations.
        """
        # Map each character to its symbol
        stream = [self.sw._token_to_27_symbol(c) for c in word if c and not c.isspace()]
        stream = [s for s in stream if s != '0']  # remove internal zeros

        if not stream:
            return WordFingerprint(word, [], [], [], 0.0)

        # Pair tensions across the word's symbol stream
        tensions  = []
        group_ids = []
        net_sv    = 0.0

        for i in range(len(stream) - 1):
            s1, s2 = stream[i], stream[i+1]
            pt = symbol_grouping.pair_tension(s1, s2)
            tensions.append(pt["tension"])
            if pt.get("group_ids"):
                group_ids.extend(pt["group_ids"])

        # Solo tension for single-character words
        if not tensions and stream:
            v = symbol_to_signed(stream[0])
            scale = 0.8 if abs(v) % 2 == 1 else 0.6
            grp   = symbol_grouping.group_for(stream[0])
            c     = grp.tension_centroid if grp else 0.1
            weight = max(0.1, 1.0 - (1.0 - c)**2)
            tensions.append((v / 13.0) * scale * weight)
            if grp:
                group_ids.append(grp.group_id)

        # Net signed value of the word
        for sym in stream:
            net_sv += symbol_to_signed(sym) / 13.0

        return WordFingerprint(word, stream, tensions, group_ids, net_sv)

    # ── Sentence-level fingerprint ────────────────────────────────────────────

    def _fingerprint_sentence(
        self,
        sentence:      str,
        symbol_stream: List[str],
        stream_ctx:    Dict[str, Any],
        word_fps:      List[WordFingerprint],
    ) -> Dict[str, Any]:
        """
        Aggregate word fingerprints into a sentence-level geometric summary.
        """
        tensions = stream_ctx.get("tensions", [])
        profile  = stream_ctx.get("tension_profile", [])

        # Dominant direction — is the field leaning positive or negative?
        net_tension = float(np.sum(tensions)) if tensions else 0.0
        if net_tension > 0.05:
            direction = "positive"
        elif net_tension < -0.05:
            direction = "negative"
        else:
            direction = "boundary"   # near the π boundary — transitional

        # Peak pair — highest absolute tension in the stream
        if tensions:
            peak_idx = int(np.argmax(np.abs(tensions)))
            peak_val = tensions[peak_idx]
            # Find the symbols at that position
            non_zero = [(i, s) for i, s in enumerate(symbol_stream) if s != '0']
            if peak_idx < len(non_zero) - 1:
                peak_pair = (non_zero[peak_idx][1], non_zero[peak_idx+1][1])
            else:
                peak_pair = ("?", "?")
        else:
            peak_val  = 0.0
            peak_pair = ("?", "?")

        # Group activation — which groups fired and how often
        all_group_ids = []
        for wfp in word_fps:
            all_group_ids.extend(wfp.group_ids)
        group_counts: Dict[int, int] = {}
        for gid in all_group_ids:
            group_counts[gid] = group_counts.get(gid, 0) + 1
        top_groups = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Field stress — std of tension profile indicates how much the field
        # is varying. High stress = complex structure. Low stress = uniform.
        field_stress = float(np.std(tensions)) if len(tensions) > 1 else 0.0

        # Boundary crossings — number of zero breaks = number of word boundaries
        boundary_count = len(stream_ctx.get("zero_boundaries", []))

        return {
            "sentence":        sentence,
            "word_count":      len(word_fps),
            "symbol_count":    len(symbol_stream),
            "boundary_count":  boundary_count,
            "mean_tension":    round(stream_ctx.get("mean_tension", 0.0), 4),
            "net_tension":     round(net_tension, 4),
            "direction":       direction,
            "field_stress":    round(field_stress, 4),
            "peak_tension":    round(peak_val, 4),
            "peak_pair":       peak_pair,
            "top_groups":      top_groups,
            "coherence":       stream_ctx.get("coherence_used", 0.0),
            "tension_profile": profile,
            "per_word":        [wfp.to_dict() for wfp in word_fps],
        }

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def process(self, sentence: str) -> Dict[str, Any]:
        """
        Full language processing pipeline.

          sentence → symbol stream → fingerprint → vocabulary → generation → output
        """
        self._process_count += 1
        start_time = time.time()

        # ── 1. Encode ─────────────────────────────────────────────────────────
        tri_data            = self.sw.triangulate(sentence)
        tri_data["prompt"]  = sentence
        symbol_stream       = tri_data.get("symbol_stream", [])

        # ── 1b. Pre-sentence carry check ──────────────────────────────────────
        # The Mersenne bands already carry residual tension from prior
        # sentences. Measure how much carry is active before we process —
        # this sets the context the new sentence will resolve against.
        prior_carry     = relational_tension.get_current_carry()
        carry_direction = relational_tension.get_carry_direction()

        # ── 2. Wave propagation ───────────────────────────────────────────────
        prop        = WavePropagator()
        prop_result = prop.propagate(tri_data, steps=60)

        recall_triggered = len(generator.memory_store) > 0
        if ouroboros_engine.should_go_generative(
                prop_result["persistence"], recall_triggered):
            prop_result = prop.propagate_generative(
                prop_result, tri_data, recall_triggered)

        # Fold line ticks with live wave amplitude
        numeric_wave = [x for x in prop_result.get("waveform_sample", [0.1])
                        if isinstance(x, (int, float))]
        wave_amp = float(np.mean(np.abs(numeric_wave))) if numeric_wave else 0.1
        for _ in range(6):
            fold_line_resonance.tick(external_wave_amp=wave_amp)

        # Bipolar lattice
        bipolar_lattice.react_to_wave(np.array(numeric_wave))
        for _ in range(4):
            bipolar_lattice.apply_tension_cycle(wave_amp)
        linked_wave = bipolar_lattice.band_emit_and_core_propagate(tri_data)
        wave_amp    = float(np.mean(np.abs(linked_wave)))

        # Clarity ratio
        clarity_ratio.measure(
            tri_data["width"], tri_data["height"],
            tri_data["total_triangles"], tri_data["n_original"]
        )

        # ── 3. Stream context (full sentence) ─────────────────────────────────
        stream_ctx = symbol_grouping.stream_context(symbol_stream)

        # ── 4. Per-word fingerprints ──────────────────────────────────────────
        words    = sentence.strip().split()
        word_fps = [self._fingerprint_word(w) for w in words]

        # ── 5. Session vocabulary + naming ───────────────────────────────────
        vocab_hits  = []
        newly_named = []
        for wfp in word_fps:
            familiarity, is_stable = self.vocabulary.update(wfp)
            stored = self.vocabulary.lookup(wfp.word)
            centroid = 0.0
            if stored:
                # Get centroid from dominant group
                grp = symbol_grouping.group_for(
                    wfp.symbol_stream[0] if wfp.symbol_stream else 'A'
                )
                centroid = grp.tension_centroid if grp else 0.0

            # Attempt naming
            if is_stable or familiarity >= _FAMILIARITY_THRESHOLD:
                named = invariant_engine.try_name_word(
                    word          = wfp.word,
                    symbol_stream = wfp.symbol_stream,
                    appearances   = stored.appearances if stored else 1,
                    familiarity   = familiarity,
                    centroid      = centroid,
                )
                if named:
                    newly_named.append(wfp.word)

            if familiarity >= _FAMILIARITY_THRESHOLD:
                vocab_hits.append({
                    "word":        wfp.word,
                    "familiarity": familiarity,
                    "stable":      is_stable,
                    "named":       invariant_engine.is_named(wfp.word),
                    "appearances": stored.appearances if stored else 1,
                })

        # ── 5b. Non-local decay ───────────────────────────────────────────────
        # Apply decay to all groups — named invariants get a stability boost
        # first so they resist decay. Novel words decay toward zero.
        invariant_engine.apply_decay(symbol_grouping.groups)

        # ── 6. Sentence fingerprint ───────────────────────────────────────────
        fingerprint = self._fingerprint_sentence(
            sentence, symbol_stream, stream_ctx, word_fps
        )
        fingerprint["named_hits"] = [
            h["word"] for h in vocab_hits if h.get("named")
        ]
        fingerprint["newly_named"]     = newly_named
        fingerprint["prior_carry"]     = round(prior_carry, 4)
        fingerprint["carry_direction"] = carry_direction

        # Measure how well this sentence aligns with prior context
        alignment = relational_tension.measure_alignment(fingerprint)
        fingerprint["carry_alignment"] = alignment

        # Inject fingerprint into prop_result for generation
        prop_result["stream_mean_tension"] = stream_ctx["mean_tension"]
        prop_result["fold_coherence"]      = fold_line_resonance.get_coherence_signal()
        prop_result["field_direction"]     = fingerprint["direction"]
        prop_result["field_stress"]        = fingerprint["field_stress"]
        prop_result["vocab_hits"]          = len(vocab_hits)
        prop_result["vocab_stable"]        = self.vocabulary.stable_count()

        # ── 7. Observer consensus ─────────────────────────────────────────────
        obs = MultiObserver(num_observers=3)
        vib = VibrationPropagator()
        linked_numeric = [x for x in prop_result.get("waveform_sample", [0.1])
                          if isinstance(x, (int, float))]
        linked_vib = vib.holographic_linkage(np.array(linked_numeric) * 10)
        consensus, _ = obs.interact(
            linked_vib, prompt=sentence, iterations=10,
            prop_result=prop_result
        )

        # ── 8. Generation — base pass ─────────────────────────────────────────
        base_answer = generator.generate(
            prompt=sentence,
            tri_data=tri_data,
            prop_result=prop_result,
            consensus=consensus,
        )

        # ── 8b. Spin-driven modulation ────────────────────────────────────────
        answer = invariant_engine.generate_response(
            fingerprint  = fingerprint,
            base_answer  = base_answer,
            consensus    = consensus,
            persistence  = prop_result.get("persistence", 0.0),
            vocab_hits   = vocab_hits,
        )

        # ── 8c. Geometric output pipeline (inverse input) ─────────────────────
        # Runs the inverse pipeline: field state → target region → vocabulary
        # sampling → sequence assembly → parity verification.
        # This is the geometric generation path — output derived from the
        # same geometry used to process the input, not from semantic rules.
        geo_result = geometric_output.generate(
            fingerprint      = fingerprint,
            vocabulary       = self.vocabulary,
            invariant_engine = invariant_engine,
            consensus        = consensus,
            persistence      = prop_result.get("persistence", 0.0),
        )

        # ── 9. Post-sentence relational tension ───────────────────────────────
        # Inject carry from this sentence into the Mersenne bands so the
        # next sentence processes against a field shaped by this one.
        carry_injected = relational_tension.after_sentence(
            fingerprint      = fingerprint,
            vocab_hits       = vocab_hits,
            invariant_engine = invariant_engine,
        )

        # ── 9b. Feed field resolution back into fold line ─────────────────────
        # Makes spin sign field-aware rather than clock-driven.
        fold_line_resonance.update_field_state(
            persistence  = prop_result.get("persistence", 0.0),
            alignment    = fingerprint.get("carry_alignment", 0.0),
            named_count  = len(invariant_engine.named_invariants),
            carry        = relational_tension.get_current_carry(),
        )

        elapsed = time.time() - start_time

        return {
            "sentence":        sentence,
            "fingerprint":     fingerprint,
            "vocab_hits":      vocab_hits,
            "vocab_size":      self.vocabulary.size(),
            "vocab_stable":    self.vocabulary.stable_count(),
            "named_count":     len(invariant_engine.named_invariants),
            "newly_named":     newly_named,
            "answer":          answer,
            "geo_output":      geo_result,
            "consensus":       round(consensus, 4),
            "persistence":     round(prop_result.get("persistence", 0.0), 4),
            "gen_mode":        invariant_engine.get_generation_mode()["mode"],
            "carry_injected":  round(carry_injected, 4),
            "carry_alignment": alignment,
            "net_carry":       round(relational_tension.get_current_carry(), 4),
            "elapsed":         round(elapsed, 3),
        }
    def get_vocabulary(self) -> List[Dict[str, Any]]:
        return self.vocabulary.get_stable_words()

    def get_status(self) -> Dict[str, Any]:
        inv_status = invariant_engine.get_status()
        rt_status  = relational_tension.get_status()
        return {
            "process_count":    self._process_count,
            "vocab_size":       self.vocabulary.size(),
            "vocab_stable":     self.vocabulary.stable_count(),
            "named_invariants": inv_status["named_invariants"],
            "named_words":      inv_status["named_words"],
            "generation_mode":  inv_status["generation_mode"],
            "spin_description": inv_status["spin_description"],
            "coherence":        fold_line_resonance.get_coherence_signal(),
            "net_carry":        rt_status["net_carry"],
            "carry_direction":  rt_status["carry_direction"],
            "active_carries":   rt_status["active_carries"],
        }


# Singleton
language_processor = LanguageProcessor()
