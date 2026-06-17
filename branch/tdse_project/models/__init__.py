# TDSE Project - Models Package
"""
Quantum mechanical models with exact analytical solutions for 1D TDSE.

Models:
- HarmonicOscillatorGroundState: V(x) = x^2/2, ground state
- HarmonicOscillatorCoherentState: V(x) = x^2/2, coherent state (x0=2, k0=0)
- FreeParticle: V(x) = 0, Gaussian wavepacket (spreading)
"""

from .harmonic_oscillator import HarmonicOscillatorGroundState, HarmonicOscillatorCoherentState
from .free_particle import FreeParticle

__all__ = [
    "HarmonicOscillatorGroundState",
    "HarmonicOscillatorCoherentState",
    "FreeParticle",
]
