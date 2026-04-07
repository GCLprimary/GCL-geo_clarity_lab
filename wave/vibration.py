import numpy as np
from typing import Optional, Dict, Any

class VibrationPropagator:
    """
    Vibration Propagation Module - Clarity Ratio Laboratory Edition
    
    Handles resilient wave sustain, refraction, and linkage.
    Works with box dimensions and propagation results from SymbolicWave + WavePropagator.
    """

    def __init__(self, asymmetric_delta: float = 0.01639510239):
        self.asymmetric_delta = asymmetric_delta
        self.base_range = (-1.0, 1.0)

    def propagate_vibration(self, amplitude: float, distance: float = 10.0, 
                           position_ratio: float = 0.5) -> float:
        """
        Propagate vibration with natural decay and tension.
        """
        # Natural exponential decay modulated by distance
        decayed = amplitude * np.exp(-0.01 * distance)
        
        # Gentle tension based on position in the geometric box
        tension = 1.0 + 0.15 * (position_ratio - 0.5)
        
        scaled = decayed * tension
        return np.clip(scaled, *self.base_range)

    def refract(self, amplitude: float, position_ratio: float = 0.5) -> float:
        """
        Apply refraction effect (bending) based on position in the structure.
        Creates dynamic tension between observers or passes.
        """
        # Simple refraction simulation using asymmetric delta
        bend_factor = 1.0 - self.asymmetric_delta * position_ratio
        bent = amplitude * bend_factor * 0.85
        return np.clip(bent, *self.base_range)

    def holographic_linkage(self, data: np.ndarray, 
                           position_ratio: float = 0.5,
                           real_freq: Optional[float] = None) -> np.ndarray:
        """
        Simple holographic-style linkage using FFT for wave binding.
        """
        if len(data) == 0:
            return np.array([0.0])
        
        # FFT-based linkage with optional frequency modulation
        if real_freq is not None:
            freq = real_freq / np.pi
        else:
            freq = 1.0
            
        chain = np.fft.fft(data)
        # Second-pass squaring for emphasis on stronger components
        realized = np.real(chain * (freq ** 2))
        
        # Return equilibrium (mean-centered normalized)
        eq = realized - np.mean(realized)
        eq = eq / (np.std(eq) + 1e-8)
        
        return np.clip(eq, *self.base_range)

    def get_status(self) -> Dict[str, Any]:
        """Return current vibration module status."""
        return {
            "asymmetric_delta": self.asymmetric_delta,
            "base_range": self.base_range
        }


# Simple self-test
if __name__ == "__main__":
    vib = VibrationPropagator()
    
    # Test with sample amplitude from a box
    test_amp = 0.75
    propagated = vib.propagate_vibration(test_amp, distance=8.0, position_ratio=0.6)
    refracted = vib.refract(test_amp, position_ratio=0.4)
    
    test_data = np.random.randn(32) * 0.3
    linked = vib.holographic_linkage(test_data)
    
    print("VibrationPropagator test complete")
    print(f"Propagated vibration: {propagated:.4f}")
    print(f"Refracted value: {refracted:.4f}")
    print(f"Linked sample mean: {np.mean(linked):.4f}")
    print("Status:", vib.get_status())