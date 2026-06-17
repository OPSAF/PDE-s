# TDSE Project - Solvers Package
"""
Numerical solvers for 1D Time-Dependent Schrodinger Equation:
i * dpsi/dt = -(1/2) * d^2psi/dx^2 + V(x)*psi

Solvers:
- CrankNicolson: Unconditionally stable implicit method (O(dt^2))
- SplitStepFourier: Spectral method with O(dx^N) spatial accuracy
- FTCS: Explicit forward-time central-space (unstable for demonstration)
"""

from .crank_nicolson import CrankNicolson
from .split_step_fourier import SplitStepFourier
from .ftcs import FTCS

__all__ = [
    "CrankNicolson",
    "SplitStepFourier",
    "FTCS",
]
