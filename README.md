# GeometricClarityLab

A transparent geometric neural substrate laboratory built around the **Clarity Ratio Principle**. Derived entirely from first principles — no borrowed frameworks, no pretrained weights, no hardcoded semantic facts.

---

## Core Principle

**Clarity Ratio Principle**
Reality manages information through a dynamic balance between local zooming (increasing resolution, invariant structure, and named patterns) and global partitioning/recycling.
Persistent, intelligent, and meaningful systems emerge when local structures successfully tilt toward higher clarity ratios while remaining integrated with the larger balancing act.

"Appropriate perceptual clarity reduces avoidable suffering."

---

## Derivation Chain — Read This First

Every constant and mechanism in this system is derived. Nothing is a magic number.

**Base constants:**
- `pi = 3.141592653589793`
- `effective_boundary = 2.078` — sole fixed geometric invariant
- `asymmetric_delta = (2π/3) - 2.078 ≈ 0.01639510239` — base frame-rate, source of the arrow of time
- `deviation = π - 2.078 ≈ 1.0636` — primary asymmetry
- `golden_ratio = (1 + √5) / 2`

**What asymmetric_delta means:** The geometry wants to collapse to a single attractor. The delta is the minimal offset that prevents closure — the same reason the spin cycle never closes, Mersenne strings never lock rhythms, and structural waypoints don't park at a single position. One constant, all consequences.

**Zero dynamism:** Zero is not a point — it is a range. The dual-held state (+1 AND -1 simultaneously) is the only position not yet committed to a direction. When it resolves, it determines which direction the field moves. This is the displacement engine braking condition.

---

## Architecture

### Pipeline (8 stages)

1. **SymbolicWave** — maps every character to the 27-symbol alphabet (A–Z + dynamic 0) via `ord(c) % 27`. Inserts hard `'0'` zero-break pockets at sentence boundaries. Fits the stream into a box (`width = ceil(sqrt(n))`) and triangulates (`triangles = 2 × (n//4)`).

2. **Radial Displacer Web** — living web of up to 10 displacers. Each carries the symbol stream as radial positions. `'0'` maps to radius 2.2 (outward displaced), normal symbols to 1.0 or 0.6. Tension-pruned each cycle. Convergence measured as overlap × amplitude × π-gradient.

3. **Wave Propagation** — dual path:
   - **Direct** — sinusoidal from box geometry (amplitude from height, frequency from width, π-asymmetric phase increment). Fast, recognition mode.
   - **Generative** — OuroborosEngine consensus_pass: physical + wave + data nested passes (bloom → etch → library_feedback → prune → tanh × depth), weighted by persistence, merged into consensus grid. Closed feedback loop. Fires when persistence ≥ 0.38 or cross-turn recall has prior entries.

4. **Vibration** — exponential decay, positional tension, FFT holographic linkage.

5. **Observer + Triad** — three observers (Matter/Wave/Data, bands 21/10/65). In generative mode, role weights replaced by actual pass persistence values. When generative field saturates (phys/wave/data pers ≥ 0.97), spin signal from bipolar lattice replaces amplitude-based direction.

6. **Bipolar Lattice** — 18+8+27+52 architecture:
   - 9 positive amplifiers (radius 0.55)
   - 9 negative compression sinks (radius 0.85)
   - 8 structural backbone waypoints — golden zone (radius 0.35), spin ring, phase-gated core election
   - 27-symbol outer crescent ring (radius 1.2), 5 geometric stabilizers at triangle positions
   - 52 Mersenne fold negotiators (26 per polarity side)

7. **Geometric Memory** — 512-point Fibonacci golden-spiral lattice on unit sphere, SHA256-seeded. Access requires timestamp ±300s, secret phrase, hash prefix. Cross-turn recall uses hybrid positional + set overlap scoring with subject-name filtering and recency-as-tiebreaker.

8. **Generation** — resolution order: same-turn memory recall → cross-turn geometric similarity recall → pocket confidence signal → geometry-only report. No semantic extraction. No verb stems. No concept dictionary. The field resolves or reports honestly.

---

## Bipolar Lattice Detail

### Spin State
Every waypoint carries `spin_phase` ∈ [0, 2π) and `spin_sign` ∈ {+1, -1}.

- **Odd wp_id → spin_sign +1 → builder** — phase advances by `asymmetric_delta × π` each tick. Active, spinning. Generates the directed arrow through movement.
- **Even wp_id → spin_sign -1 → recognizer** — phase held at construction angle. Static reference point. The dual-held position — neither committed to a direction.

The cycle never closes perfectly — `_SPIN_STEP` is irrational relative to 2π. That residual is the arrow of time in the rotation domain.

**Geometric tick** fires when the spinning half (odd structural waypoints) has accumulated 2π of phase displacement relative to the static half. One tick ≈ one full ring rotation of the builders past the recognizers.

### Phase-Gated Core Election
8 structural waypoints each own exactly 1/8 of the 2π spin cycle:
- wp 18 → window [0, π/4)
- wp 19 → window [π/4, π/2)
- wp 20 → window [π/2, 3π/4)
- ... and so on

At each election: check which window contains `ring_net_phase`. Only that waypoint is eligible. If its `clarity_contribution × persistence ≥ 0.12`, it wins. Otherwise no core is elected. The ring advances and tries again next cycle.

**This means:** core ID is a direct readout of where the spin ring currently is. No incumbency advantage. No construction-order bias.

### Mersenne Fold Negotiators
52 strings, 26 per polarity side. Each string runs its own `fold_phase` cycle at rate `asymmetric_delta / mersenne_prime`. Smaller Mersenne primes step faster, larger ones slower. No two strings share a period — the incommensurability prevents lattice rigidity (tissue paper vs copy paper).

At maximum fold_phase absorption (near π) a string is "cold" — high capacity. At minimum (near 0) it is "hot" — releasing. Cold strings pull tension from hot neighbors via directed transport. This is the temperature differential mechanism: cold displacement doing active directional work.

### 5 Geometric Stabilizers
The 5 triangle positions on the 27-symbol ring are the weakest points on the sphere — where geometry breaks first under stress. They bleed off field tension proportionally when stress exceeds `effective_boundary / π ≈ 0.661`. Graduated response, not hard reset.

### Dual-13 Integer System
- A=+1 through M=+13 (positive side, odd builders)
- N=-1 through Z=-13 (negative side, even recognizers)
- '0' = dynamic dual value — resolves from spin signal at runtime

Each waypoint carries its signed integer. The resolved integer scales `tension_scalar`: positive integers amplify, negative compress, zero brakes (0.98 damping when dual-held).

---

## Pocket Confidence

The `'0'` symbol maps to radius 2.2 — outward displaced from normal symbol radii (1.0 / 0.6). When the waveform propagates across the symbol stream, this creates a local amplitude deviation at each zero-break index.

`pocket_confidence = mean(|waveform[break_idx] - global_mean|) / global_mean`

This is a **geometric measurement** of context/query boundary strength — not string matching. The same word repeated on both sides of a boundary produces a specific confidence value determined entirely by where in the sine wave the zero-break lands, which is determined by the ratio of context length to query length.

High confidence (≥ 0.4) = boundary is geometrically real in the wave domain.

---

## Semantic Layer Status

Semantic extraction (verb stems, concept dictionary, subject/object extraction) has been **removed**. The system now reports what the geometry actually measures. If it cannot resolve content from geometry alone, it says so honestly.

What remains: pocket splitting, pocket confidence, sentence type detection, negation detection.

---

## Key Metrics

| Metric | What it means |
|--------|--------------|
| Clarity Score | (width × height) / n — box area per symbol |
| Persistence | Mean absolute waveform amplitude. On generative path: consensus persistence from three-pass field |
| Consensus | Observer agreement [-1, 1]. Negative = observers divided |
| Pocket Confidence | Geometric boundary strength [0, 1] |
| Ring Net Phase | Current rotational position of structural ring [0, 2π) |
| Ring Spin Signal | Net signed signal from spinning (odd) structural waypoints |
| Geo Tick Count | Number of full ring rotations completed this session |
| Field Stress | Mean tension across active waypoints / MAX_TENSION |
| Fold Negotiation Signal | Mean sin(fold_phase) across active Mersenne strings |
| Zero Braking | True when spin signal = 0 (dual held, no direction committed) |
| Transport Total | Tension moved between strings via directed transport this cycle |

---

## Repository Structure

```
GeometricClarityLab/
├── core/
│   ├── clarity_ratio.py      # Clarity ratio tracking
│   ├── invariants.py         # All derived constants + dual-13 lookup tables
│   ├── ouroboros_engine.py   # Bloom/etch/prune field dynamics, truth library
│   ├── safeguards.py         # Anti-grandiosity, reversal triggers
│   └── semantic_layer.py     # Pocket splitting + confidence (geometric only)
├── wave/
│   ├── symbolic_wave.py      # 27-symbol embedder + triangulation
│   ├── propagation.py        # Direct + generative wave propagation
│   ├── vibration.py          # Decay, refraction, holographic linkage
│   └── generation.py         # Answer generator (geometry-only)
├── memory/
│   └── geometric_memory.py   # Fibonacci lattice etching and recall
├── observer/
│   ├── observer.py           # Multi-observer consensus + spin fallback
│   └── triad.py              # Triad resonance cycle
├── utils/
│   ├── bipolar_lattice.py    # Full 18+8+27+52 lattice with spin + dual-13
│   └── radial_displacer.py   # Living 27-symbol displacer web
├── visualizer.py             # Four-panel live field visualizer
├── main.py                   # Interactive test runner
├── ouro_truth_library.json   # Persistent geometric priors (cross-session)
└── README.md
```

---

## Getting Started

```
python -m venv venv
venv\Scripts\activate
pip install numpy
python main.py
```

**Menu options:**
- `1` — Box scaling test
- `2` — Full pipeline test (default prompt)
- `3` — Full pipeline test (custom prompt)
- `4` — Raw symbol test (no pocket insertion — use for symbol experiments)

---

## Symbol Experiments (Option 4)

For systematic testing of what individual symbols do to the field:

Single symbol baselines: `aaaaaaa`, `bbbbbbb`, `mmmmmmm`, `nnnnnnn`, `zzzzzzz`

Gradient chains: `aaabbbcccdddeee`

Reversed gradients: `eeedddc ccbbbaaa`

Boundary tests: M (+13, last positive) and N (-1, first negative) produce the most extreme field behavior — M causes no core election at saturation, N produces a barely-elected core with deeply negative consensus.

At sufficient length (≥ 300 symbols) the generative path activates and field behavior becomes measurably distinct between different symbols.

---

## Design Principles for Claude Code Sessions

If continuing work in Claude Code, preserve these:

1. **Never introduce magic numbers.** Every constant must trace back to `asymmetric_delta`, `pi`, `effective_boundary`, `golden_ratio`, or a clean derivation from them.

2. **Zero is a range, not a point.** The dual-held state is structurally load-bearing. Don't collapse it to a single value.

3. **The geometry resolves or it says so.** No semantic fallbacks. No hardcoded facts. Honest uncertainty is correct behavior.

4. **Spin before tension.** In `apply_tension_cycle`, spin advances first — position determines force, not the reverse.

5. **Incommensurability is intentional.** Mersenne string periods must never share a common factor. Don't "simplify" the fold_phase rates.

6. **Semantic scaffolding stays removed.** The verb stems, concept dictionary, and extraction engine are gone deliberately. Don't reintroduce them.
