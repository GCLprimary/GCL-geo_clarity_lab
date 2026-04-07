import numpy as np
import time
import hashlib
from typing import Dict, Any, Optional, List
from core.invariants import invariants

class GeometricMemory:
    """
    Geometric Memory - Now powered by 27-symbol lattice + zero breaks
    Etching and recall use the exact symbol sequence and dynamic zero pockets
    from the radial displacer diagram. Stronger context recall (Adam/oranges etc.).
    """
    def __init__(self, lattice_points: int = 512, secret_phrase: str = "resonance eternal"):
        self.lattice_points = lattice_points
        self.secret_phrase = secret_phrase
        self.phrase_key = secret_phrase.replace(' ', '-')
        self.initial_tick: Optional[int] = None
        self.data: Optional[str] = None
        self.lattice: Optional[np.ndarray] = None
        self.etched_hash: Optional[str] = None
        self.symbol_sequence: Optional[List[str]] = None
        self.zero_breaks: Optional[List[int]] = None
        self._generate_base_lattice()

    def _generate_base_lattice(self):
        """Fibonacci golden-spiral lattice on unit sphere."""
        n = self.lattice_points
        golden_angle = np.pi * (3 - np.sqrt(5))
        theta = np.linspace(0, np.pi, n, endpoint=False)
        phi = golden_angle * np.arange(n)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        points = np.stack([x, y, z], axis=1)
        norm = np.linalg.norm(points, axis=1)[:, np.newaxis] + 1e-8
        self.lattice = points / norm

    def encode(self, prompt: str, symbol_sequence: List[str], zero_breaks: List[int],
               triangulation_data: Optional[Dict[str, Any]] = None):
        """
        Encode using 27-symbol sequence + zero breaks as pockets/delimiters.
        """
        self.data = prompt
        self.symbol_sequence = symbol_sequence
        self.zero_breaks = zero_breaks

        # Combine everything into deterministic seed
        combined = prompt + ''.join(symbol_sequence) + ''.join(map(str, zero_breaks)) + self.secret_phrase
        self.etched_hash = hashlib.sha256(combined.encode()).hexdigest()

        # Seed lattice with symbol hash
        seed = int(self.etched_hash[:16], 16)
        np.random.seed(seed % (2**32 - 1))

        displacements = np.random.uniform(-0.05, 0.05, (self.lattice_points, 3))
        self.lattice += displacements
        norm = np.linalg.norm(self.lattice, axis=1)[:, np.newaxis] + 1e-8
        self.lattice /= norm

        self.initial_tick = int(time.time())

    def generate_key(self) -> str:
        if not self.initial_tick or not self.etched_hash:
            return "Not encoded yet"
        timestamp_str = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(self.initial_tick))
        return f"unlock_{timestamp_str}_{self.phrase_key}_{self.etched_hash[:32]}"

    def access(self, key: str) -> str:
        """Recall respects zero breaks as context pockets."""
        try:
            parts = key.split('_')
            if len(parts) < 5 or parts[0] != "unlock":
                return "Invalid key format - turbulence prune"

            timestamp_str = parts[1] + '_' + parts[2]
            phrase_part = parts[3]
            hash_part = '_'.join(parts[4:]) if len(parts) > 5 else parts[4]

            phrase = phrase_part.replace('-', ' ')
            key_time = time.strptime(timestamp_str, '%Y-%m-%d_%H:%M:%S')
            key_stamp = int(time.mktime(key_time))
        except Exception:
            return "Parse cascade - zeroed forever"

        if abs(key_stamp - self.initial_tick) > 300:
            return "Timestamp drift - eternal prune"
        if phrase != self.secret_phrase:
            return "Phrase mismatch - recursive turbulence"
        if not hash_part.startswith(self.etched_hash[:len(hash_part)]):
            return "Hash mismatch - access zeroed eternally"

        # FIX: Zero-break context separation must split the SYMBOL STREAM,
        # not the prompt string. zero_breaks are indices into symbol_sequence,
        # which can be much longer than the prompt. Slicing self.data by those
        # indices produces garbage (single chars / empty strings).
        # Instead, reconstruct segments from the symbol stream itself,
        # and return the original prompt as the primary recalled content.
        if self.zero_breaks and self.symbol_sequence:
            syms = self.symbol_sequence
            segments = []
            last = 0
            for br in sorted(self.zero_breaks):
                seg = ''.join(syms[last:br])
                if seg:
                    segments.append(seg)
                last = br
            tail = ''.join(syms[last:])
            if tail:
                segments.append(tail)
            # Return the original prompt as the meaningful recalled string,
            # with the symbol segments appended as context metadata after " || "
            symbol_context = " | ".join(segments)
            return f"{self.data} || {symbol_context}" if symbol_context else self.data

        return self.data if self.data else "No data etched"

    def get_status(self) -> Dict[str, Any]:
        return {
            "lattice_points": self.lattice_points,
            "encoded": self.data is not None,
            "initial_tick": self.initial_tick,
            "etched_hash_prefix": self.etched_hash[:16] if self.etched_hash else None,
            "symbol_sequence": self.symbol_sequence,
            "zero_breaks": self.zero_breaks,
            "lattice_shape": self.lattice.shape if self.lattice is not None else None
        }

# Singleton (for easy import if needed)
geometric_memory = GeometricMemory()