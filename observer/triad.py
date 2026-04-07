import numpy as np
from typing import Dict, Any
from core.invariants import invariants
from core.clarity_ratio import clarity_ratio
from core.safeguards import safeguards

class Triad:
    """
    Triad Resonance Cycle - Pure Geometric Version (no weights, no torch)
    Matter → Wave → Data resonance using radial invariants + clarity + safeguards.
    Everything is a displacer. Convergence-based reasoning at intersection points.
    """
    def __init__(self):
        pass  # everything is displacer + invariant-driven

    def triad_resonance_cycle(self, hidden: np.ndarray, prompt_features: Dict = None) -> Dict[str, Any]:
        """Pure geometric triad cycle."""
        # Convert to radial signature using invariants
        signature = hidden.mean(axis=0)
        persistence = float(np.mean(np.abs(signature)))

        # Apply π-gradient + odd/even bias (toroidal breathing)
        pi_grad = invariants.get_pi_gradient(scale=persistence)
        layer = len(clarity_ratio.history) % 2
        signature = invariants.odd_even_bias(signature * pi_grad, layer)

        # Persistence-guided sparsity (geometric threshold, no quantile)
        threshold = np.mean(np.abs(signature)) * 1.7  # stronger but deterministic
        mask = np.abs(signature) > threshold
        modulated = signature * mask * 1.2

        # ClarityRatio + Safeguards feedback (the real crankshaft)
        clarity = clarity_ratio.current_ratio
        if safeguards.should_force_reversal(persistence, len(clarity_ratio.history)):
            modulated = safeguards.apply_reversal_trigger(modulated.mean()) * np.ones_like(modulated)

        # Residual injection into radial web (everything is a displacer)
        modulated = np.tile(modulated, (hidden.shape[0], 1))[:, :hidden.shape[1]]

        return {
            "hidden": hidden + modulated * 0.85,   # same residual strength you had
            "persistence": persistence,
            "sparsity": float(1 - np.mean(mask)),
            "clarity_ratio": round(clarity, 4)
        }

    def forward(self, x: np.ndarray, prompt: str = "") -> tuple:
        """Deterministic forward — no transformer."""
        hidden = x.astype(float)  # treat input as raw radial displacements
        layer_histories = []
        for layer in range(4):  # keep your original num_layers
            triad_out = self.triad_resonance_cycle(hidden, {"prompt": prompt})
            hidden = triad_out["hidden"]
            layer_histories.append({
                "layer": layer,
                "persistence": triad_out["persistence"],
                "sparsity": triad_out["sparsity"],
                "clarity_ratio": triad_out["clarity_ratio"]
            })
        # Final logits become geometric projection (no nn.Linear)
        logits = hidden.mean(axis=1)  # collapse to consensus scalar per “token”
        return logits, layer_histories

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": "pure_geometric",
            "uses": "invariants + clarity_ratio + safeguards",
            "history_length": len(clarity_ratio.history)
        }