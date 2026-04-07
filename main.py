"""
GeometricClarityLab - Interactive Test Runner
Fold Line Resonance + Symbol Grouping + Semantic Diagnostic wired.
"""
import sys
from pathlib import Path
import numpy as np
import time

root = Path(__file__).parent.absolute()
sys.path.insert(0, str(root))

# Core + semantic
from core.clarity_ratio import clarity_ratio
from core.invariants import invariants
from core.safeguards import safeguards
from core.semantic_layer import semantic_layer
from core.ouroboros_engine import ouroboros_engine

# Wave & radial layers
from wave.symbolic_wave import SymbolicWave
from wave.propagation import WavePropagator
from wave.vibration import VibrationPropagator
from utils.radial_displacer import radial_displacer
from utils.bipolar_lattice import bipolar_lattice

# Fold line + symbol grouping
from utils.fold_line_resonance import fold_line_resonance
from utils.symbol_grouping import symbol_grouping

# Semantic diagnostic
from diagnostics.semantic_probe import run_semantic_diagnostic

# Language processor
from language.processor import language_processor

# Memory, observer, triad, generator
from memory.geometric_memory import GeometricMemory
from observer.observer import MultiObserver
from observer.triad import Triad
from wave.generation import generator
from language.geometric_output import geometric_output


def print_header():
    print("\n" + "=" * 90)
    print("GeometricClarityLab — Fold Line + Symbol Grouping + Semantic Diagnostic")
    print("=" * 90)
    print(f"Clarity Ratio    : {clarity_ratio.get_status()}")
    print(f"Invariants       : {invariants.get_status()}")
    print(f"Safeguards       : {safeguards.get_status()}")
    print(f"Radial Web       : {radial_displacer.get_status()}")
    print(f"Bipolar Lattice  : {bipolar_lattice.get_status()}")
    print(f"Ouroboros Engine : {ouroboros_engine.get_status()}")
    print(f"Fold Line        : {fold_line_resonance.get_status()}")
    print(f"Symbol Grouping  : {symbol_grouping.get_status()}")


def test_full_pipeline(custom_prompt: str = None):
    print("\n[PIPELINE] Full Pipeline — Fold Line + Symbol Grouping Active")

    prompt = custom_prompt or "Clara loves to paint watercolours. What does Clara love to do?"
    print(f"Prompt: {prompt}")

    start_time = time.time()

    # ── 1. Symbolic encoding ──────────────────────────────────────────────────
    sw       = SymbolicWave()
    tri_data = sw.triangulate(prompt)
    tri_data["prompt"] = prompt
    clarity_ratio.measure(
        tri_data["width"], tri_data["height"],
        tri_data["total_triangles"], tri_data["n_original"]
    )
    print(f"Box: {tri_data['width']}×{tri_data['height']} | "
          f"Triangles: {tri_data['total_triangles']} | "
          f"Symbols: {tri_data['n_original']}")

    # ── 2. Radial web ─────────────────────────────────────────────────────────
    radial = radial_displacer.generate_structure(prompt, tri_data, wave_amplitude=0.0)

    # ── 3. Wave propagation ───────────────────────────────────────────────────
    prop        = WavePropagator()
    prop_result = prop.propagate(tri_data, steps=60)

    recall_triggered = len(generator.memory_store) > 0
    if ouroboros_engine.should_go_generative(prop_result["persistence"], recall_triggered):
        prop_result = prop.propagate_generative(prop_result, tri_data, recall_triggered)

    # ── 4. Vibration + holographic linkage ────────────────────────────────────
    numeric_wave = [x for x in prop_result.get("waveform_sample", [0.1])
                    if isinstance(x, (int, float))]
    vib         = VibrationPropagator()
    linked_wave = vib.holographic_linkage(np.array(numeric_wave) * 10)

    # ── 5. Observer + Triad ───────────────────────────────────────────────────
    obs = MultiObserver(num_observers=3)
    consensus, perturb = obs.interact(
        linked_wave, prompt=prompt, iterations=10, prop_result=prop_result
    )
    triad = Triad()
    _, triad_hist = triad.forward(
        linked_wave.reshape(1, -1) if linked_wave.ndim == 1 else linked_wave,
        prompt=prompt
    )

    # ── 6. Radial + bipolar refinement ────────────────────────────────────────
    wave_amp = float(np.mean(np.abs(linked_wave)))
    radial_displacer.react_to_wave(linked_wave)
    radial = radial_displacer.generate_structure(prompt, tri_data, wave_amplitude=wave_amp)

    bipolar_lattice.react_to_wave(linked_wave)
    for _ in range(6):
        bipolar_lattice.apply_tension_cycle(wave_amp)

    linked_wave = bipolar_lattice.band_emit_and_core_propagate(tri_data)
    wave_amp    = float(np.mean(np.abs(linked_wave)))

    clarity_ratio.measure(
        tri_data["width"], tri_data["height"],
        tri_data["total_triangles"], tri_data["n_original"]
    )
    bipolar_status = bipolar_lattice.get_status()

    # ── 7. Fold line resonance ────────────────────────────────────────────────
    fold_results = []
    for _ in range(8):
        fr = fold_line_resonance.tick(external_wave_amp=wave_amp)
        fold_results.append(fr)

    total_folds      = sum(r["fold_events_this_tick"] for r in fold_results)
    coherence        = fold_line_resonance.get_coherence_signal()
    active_fold_zone = fold_line_resonance.get_active_fold_zone()

    # ── 8. Symbol grouping ────────────────────────────────────────────────────
    symbol_stream   = tri_data.get("symbol_stream", [])
    stream_ctx      = symbol_grouping.stream_context(symbol_stream)
    grouping_status = symbol_grouping.get_status()

    # ── 9. Semantic layer ─────────────────────────────────────────────────────
    semantic_layer.project_to_radial(
        prompt.split()[0] if prompt else "unknown", clarity=1.0
    )

    # ── 10. Memory etching ────────────────────────────────────────────────────
    mem          = GeometricMemory(secret_phrase="resonance eternal")
    displacers   = radial.get("displacers", [{}])
    radial_syms  = displacers[0].get("symbol_sequence", []) if displacers else []
    radial_zeros = displacers[0].get("zero_breaks", [])     if displacers else []
    mem.encode(prompt, radial_syms, radial_zeros)
    key      = mem.generate_key()
    recalled = mem.access(key)

    # ── 11. Answer generation ─────────────────────────────────────────────────
    prop_result["stream_mean_tension"] = stream_ctx["mean_tension"]
    prop_result["fold_coherence"]      = coherence

    answer = generator.generate(
        prompt=prompt,
        tri_data=tri_data,
        prop_result=prop_result,
        consensus=consensus,
        memory_recall=recalled,
    )

    elapsed = time.time() - start_time

    print(f"\n--- Generated Answer ---")
    print(answer)

    print(f"\nPipeline Metrics:")
    print(f" Clarity Score     : {prop_result['clarity_ratio_score']:.4f}"
          f"  (trend: {clarity_ratio.get_trend()['trend']})")
    print(f" Persistence       : {prop_result['persistence']:.4f}")
    print(f" Consensus         : {consensus:.4f}")
    print(f" Radial Convergence: {radial['web_convergence_score']:.4f}"
          f"  ({radial['num_displacers']} displacers)")
    print(f" Memory Recall     : {'Success' if len(recalled) > 20 else 'Partial'}")
    print(f" Bipolar Core      : {bipolar_status['core_id']}"
          f" (score {bipolar_status['core_score']:.4f},"
          f" strings {bipolar_status['active_strings']})")

    print(f"\nFold Line Resonance:")
    print(f" Folds this step   : {total_folds}")
    print(f" Coherence signal  : {coherence:.4f}"
          f"  ({'resolving' if coherence > 0.3 else 'forming'})")
    print(f" Active fold zone  : phase={active_fold_zone['centroid_phase']:.4f}"
          f"  spread={active_fold_zone['spread']:.4f}"
          f"  strength={active_fold_zone['strength']:.4f}")
    print(f" Imprinted points  : {fold_line_resonance.get_status()['imprinted_points']}")

    print(f"\nSymbol Grouping:")
    print(f" Groups            : {grouping_status['total_groups']}"
          f"  (imprinted: {grouping_status['imprinted_groups']})")
    print(f" Stream tension    : mean={stream_ctx['mean_tension']:.4f}"
          f"  zero_boundaries={len(stream_ctx['zero_boundaries'])}")

    mode = prop_result.get("mode", "direct")
    if mode == "generative":
        print(f"\nGenerative Pass    : "
              f"phys={prop_result.get('phys_pers', 0):.3f}"
              f" wave={prop_result.get('wave_pers', 0):.3f}"
              f" data={prop_result.get('data_pers', 0):.3f}"
              f" consensus={prop_result.get('consensus_pers', 0):.3f}")
    else:
        print(f" Mode              : {mode}")
    print(f" Elapsed           : {elapsed:.3f}s")


def main():
    print_header()

    while True:
        print("\nOptions:")
        print(" 1 - Box Scaling Test")
        print(" 2 - Full Pipeline Test (default prompt)")
        print(" 3 - Full Pipeline Test with custom prompt")
        print(" 4 - Inspect symbol groups")
        print(" 5 - Semantic Diagnostic (basis matrix + excitation sequence)")
        print(" 6 - Targeted Diagnostic (excite unimprinted singletons)")
        print(" 7 - Language Processing (fingerprint + generation)")
        print(" q - Quit")

        choice = input("\nChoose [1/2/3/4/5/6/7/q]: ").strip().lower()

        if choice == '1':
            print("\n[1] Box Scaling Test")
            sw    = SymbolicWave()
            sizes = [16, 32, 64, 128, 256]
            for n in sizes:
                seq  = list(range(1, n + 1))
                data = sw.triangulate(seq)
                clarity_ratio.measure(
                    data["width"], data["height"],
                    data["total_triangles"], data["n_original"]
                )
                print(f" n={n:3d} → Box: {data['width']:2d}×{data['height']:2d}"
                      f" | Triangles: {data['total_triangles']:3d}"
                      f" | Clarity: {clarity_ratio.current_ratio:.4f}")

        elif choice == '2':
            test_full_pipeline()

        elif choice == '3':
            prompt = input("Enter prompt: ").strip()
            if prompt:
                test_full_pipeline(prompt)
            else:
                print("Prompt cannot be empty.")

        elif choice == '4':
            groups    = symbol_grouping.get_group_summary()
            imprinted = [g for g in groups if g["tension_centroid"] > 0.05]
            print(f"\nActive symbol groups ({len(imprinted)} imprinted of {len(groups)} total):")
            for g in imprinted:
                print(f"  Group {g['group_id']:2d} | "
                      f"members={g['members']} | "
                      f"tension={g['base_tension']:+.3f} | "
                      f"polarity={g['dominant_polarity']}")
            if not imprinted:
                print("  No imprinted groups yet — run option 5 first.")

        elif choice == '5':
            print("\n[5] Semantic Diagnostic")
            print("  Modes:")
            print("    p - pairs    (2-symbol chains)")
            print("    t - triples  (3-symbol chains)")
            print("    c - chain    (4, 5, or 6-symbol chains)")
            mode_in = input("  Mode [p/t/c]: ").strip().lower()

            chain_length = 2
            if mode_in == 't':
                mode = "triples"
                chain_length = 3
            elif mode_in == 'c':
                mode = "chain"
                len_in = input("  Chain length [4/5/6]: ").strip()
                try:
                    chain_length = int(len_in)
                    if chain_length not in (4, 5, 6):
                        chain_length = 4
                except ValueError:
                    chain_length = 4
                print(f"  Using {chain_length}-symbol chains.")
            else:
                mode = "pairs"

            count_in = input("  How many prompts? [default 54]: ").strip()
            try:
                max_p = int(count_in) if count_in else 54
            except ValueError:
                max_p = 54

            run_semantic_diagnostic(mode=mode, max_prompts=max_p,
                                    verbose=True, chain_length=chain_length)

        elif choice == '6':
            print("\n[6] Targeted Diagnostic — Unimprinted Singleton Excitation")
            # Show current unimprinted singletons
            groups      = symbol_grouping.get_group_summary()
            singletons  = [
                g["members"][0] for g in groups
                if g["size"] == 1
                and g["tension_centroid"] < 0.005
                and g["members"][0] != '0'
            ]
            if not singletons:
                print("  No unimprinted singletons — all symbols are active.")
                print("  Running targeted mode anyway with N, O, P as focus.")
                singletons = ['N', 'O', 'P']
            else:
                print(f"  Unimprinted singletons: {singletons}")

            # Allow override
            override = input(
                f"  Target symbols [{', '.join(singletons)}] or enter custom (comma-separated): "
            ).strip()
            if override:
                custom = [s.strip().upper() for s in override.split(',')
                          if s.strip().upper() in [sym for sym in
                          ['0']+[chr(ord('A')+i) for i in range(26)]]]
                if custom:
                    singletons = custom

            count_in = input("  How many prompts? [default 108]: ").strip()
            try:
                max_p = int(count_in) if count_in else 108
            except ValueError:
                max_p = 108

            run_semantic_diagnostic(
                mode="targeted",
                max_prompts=max_p,
                verbose=True,
                target_symbols=singletons,
            )

        elif choice == '7':
            print("\n[7] Language Processing — Geometric Fingerprint + Generation")

            # ── Auto warm-up ──────────────────────────────────────────────────
            # Run a 6-chain diagnostic pass before opening the language loop
            # so the fold line has live imprints and groups are active.
            # Without this, named invariants from prior sessions can't be
            # re-earned because the centroid threshold is never crossed.
            imprinted = fold_line_resonance.get_status()["imprinted_points"]
            if imprinted < 100:
                print("\n  [warm-up] Seeding fold line for language session...")
                from diagnostics.semantic_probe import (
                    generate_excitation_sequence, probe_prompt
                )
                warmup_prompts = generate_excitation_sequence(
                    mode="chain", max_prompts=27, chain_length=6
                )
                for wp in warmup_prompts:
                    probe_prompt(wp)
                # Additional targeted fold ticks
                wave_amp = invariants.asymmetric_delta * 3
                for _ in range(48):
                    fold_line_resonance.tick(external_wave_amp=wave_amp)
                symbol_grouping._compute_groups()
                imprinted_after = fold_line_resonance.get_status()["imprinted_points"]
                imprinted_grps  = symbol_grouping.get_status()["imprinted_groups"]
                print(f"  [warm-up] Done — "
                      f"imprinted pts: {imprinted_after} | "
                      f"active groups: {imprinted_grps}")
            else:
                print(f"\n  [warm-up] Fold line already seeded "
                      f"({imprinted} imprinted pts) — skipping.")

            print("\n  Enter sentences to process. Type 'vocab' to see session")
            print("  vocabulary, 'carry' to inspect relational tension window,")
            print("  'status' for processor state, 'back' to return.")
            print()

            while True:
                sentence = input("  > ").strip()
                if not sentence or sentence.lower() == 'back':
                    break

                if sentence.lower() == 'carry':
                    from language.relational_tension import relational_tension as rt
                    status = rt.get_status()
                    print(f"\n  Relational tension window:")
                    print(f"    Net carry    : {status['net_carry']:+.4f} "
                          f"({status['carry_direction']})")
                    print(f"    Active carries: {status['active_carries']}")
                    for entry in status["window"]:
                        print(f"    [{entry['age']} ago] \"{entry['sentence']}\" "
                              f"carry={entry['carry_value']:+.4f} "
                              f"anchors={entry['named_anchors']}")
                    print()
                    continue

                if sentence.lower() == 'vocab':
                    stable = language_processor.get_vocabulary()
                    print(f"\n  Session vocabulary ({len(stable)} stable words):")
                    if stable:
                        for entry in sorted(stable, key=lambda x: x["appearances"],
                                            reverse=True):
                            print(f"    {entry['word']:15s} | "
                                  f"appearances={entry['appearances']} | "
                                  f"tension={entry['mean_tension']:+.4f} | "
                                  f"group={entry['dominant_group']} | "
                                  f"net_signed={entry['net_signed']:+.3f}")
                    else:
                        print("    No stable words yet.")
                    print()
                    continue

                if sentence.lower() == 'status':
                    s = language_processor.get_status()
                    print(f"\n  Processor status:")
                    for k, v in s.items():
                        print(f"    {k}: {v}")
                    print()
                    continue

                # Process the sentence
                result = language_processor.process(sentence)
                fp     = result["fingerprint"]

                print(f"\n  ── Fingerprint ──────────────────────────────────────")
                print(f"  Direction    : {fp['direction']}")
                print(f"  Mean tension : {fp['mean_tension']:+.4f}")
                print(f"  Net tension  : {fp['net_tension']:+.4f}")
                print(f"  Field stress : {fp['field_stress']:.4f}")
                print(f"  Boundaries   : {fp['boundary_count']}  "
                      f"(words: {fp['word_count']})")
                print(f"  Peak pair    : {fp['peak_pair'][0]}→{fp['peak_pair'][1]}"
                      f"  ({fp['peak_tension']:+.4f})")

                if fp["top_groups"]:
                    group_str = "  ".join(
                        f"grp{gid}×{cnt}" for gid, cnt in fp["top_groups"]
                    )
                    print(f"  Top groups   : {group_str}")

                # Per-word breakdown
                print(f"\n  ── Per-word ─────────────────────────────────────────")
                for wd in fp["per_word"]:
                    print(f"  {wd['word']:12s} | "
                          f"tension={wd['mean_tension']:+.4f} | "
                          f"group={wd['dominant_group']:2d} | "
                          f"net={wd['net_signed']:+.3f}")

                # Vocabulary hits
                if result["vocab_hits"]:
                    print(f"\n  ── Vocabulary hits ──────────────────────────────────")
                    for hit in result["vocab_hits"]:
                        stable_marker = " [STABLE]" if hit["stable"] else ""
                        print(f"  '{hit['word']}'"
                              f"  familiarity={hit['familiarity']:.3f}"
                              f"  appearances={hit['appearances']}"
                              f"{stable_marker}")

                # Answer
                print(f"\n  ── Answer ({result['gen_mode']}) ──────────────────────────")
                print(f"  {result['answer']}")

                # Geometric output — inverse pipeline
                geo = result.get("geo_output", {})
                if geo:
                    locked_str = "⟳ parity locked" if geo.get("parity_locked") else "~ approximate"
                    print(f"\n  ── Geometric Output ({locked_str}) ──────────────")
                    print(f"  {geometric_output.format_output(geo)}")
                    print(f"  [polarity {geo['field_polarity']:+.3f} | "
                          f"region {geo['target_region']['side']} "
                          f"[{geo['target_region']['low']:.1f},{geo['target_region']['high']:.1f}] | "
                          f"candidates: {', '.join(geo.get('candidates', [])[:4])}]")

                if result.get("newly_named"):
                    print(f"\n  ★ Newly named: {result['newly_named']}")

                # Carry alignment — show how this sentence relates to prior context
                alignment = result.get("carry_alignment", 0.0)
                net_carry = result.get("net_carry", 0.0)
                if abs(net_carry) > 0.001:
                    align_str = (
                        "aligned" if alignment > 0.3 else
                        "opposing" if alignment < -0.3 else
                        "neutral"
                    )
                    print(f"\n  carry: {net_carry:+.4f} | "
                          f"alignment: {alignment:+.4f} ({align_str}) | "
                          f"injected: {result.get('carry_injected', 0.0):+.4f}")

                print(f"\n  consensus={result['consensus']:+.4f}  "
                      f"persistence={result['persistence']:.4f}  "
                      f"resolution={fold_line_resonance.get_resolution_score():.3f}  "
                      f"vocab={result['vocab_size']} words  "
                      f"({result['vocab_stable']} stable, "
                      f"{result['named_count']} named)  "
                      f"elapsed={result['elapsed']}s")
                print()

        elif choice == 'q':
            print("Goodbye. Lattice preserved.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
