"""Split-Step Fourier method solver for the 1D Time-Dependent Schrodinger Equation.

This module implements the split-operator (split-step) Fourier method, which
exploits the fact that the kinetic energy operator is diagonal in momentum
(Fourier) space while the potential energy operator is diagonal in position
space. The time evolution operator is approximated via symmetric Strang
splitting:

    exp(-i*H*dt/hbar) ~ exp(-i*T*dt/(2hbar)) * exp(-i*V*dt/hbar) * exp(-i*T*dt/(2hbar))

This method is unitary to machine precision, second-order accurate in time,
and spectrally accurate in space (limited by FFT resolution).
"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq

from .base import TDSESolverBase


class SplitStepFourier(TDSESolverBase):
    """Split-Step Fourier (pseudo-spectral) solver for the 1D TDSE.

    The kinetic half-steps are applied in momentum (Fourier) space where the
    Laplacian is a simple multiplication by -k^2. The potential full step is
    applied in position space as a phase factor. This yields a fast, accurate,
    and norm-preserving propagation scheme.

    Parameters
    ----------
    x : np.ndarray
        1D spatial grid array.
    hbar : float, optional
        Reduced Planck constant. Default is 1.0.
    m : float, optional
        Particle mass. Default is 1.0.

    Attributes
    ----------
    name : str
        Solver name, set to "Split-Step Fourier".
    k : np.ndarray
        Momentum grid (wavenumbers) matching the FFT frequencies.
    """

    def __init__(self, x: np.ndarray, hbar: float = 1.0, m: float = 1.0) -> None:
        super().__init__(x, hbar=hbar, m=m)
        self.name = "Split-Step Fourier"
        # Precompute momentum grid from FFT frequencies
        self.k: np.ndarray = 2.0 * np.pi * fftfreq(self.nx, self.dx)

    def propagate(self, psi: np.ndarray, V: np.ndarray, dt: float) -> np.ndarray:
        """Propagate the wavefunction by one time step using split-step Fourier.

        Applies the symmetric Strang splitting:

            1. Half kinetic step in Fourier space
            2. Full potential step in position space
            3. Half kinetic step in Fourier space

        Parameters
        ----------
        psi : np.ndarray
            Current wavefunction on the spatial grid (complex-valued).
        V : np.ndarray
            Potential energy array on the spatial grid.
        dt : float
            Time step size.

        Returns
        -------
        np.ndarray
            Wavefunction after one split-step time step.
        """
        hbar: float = self.hbar
        m: float = self.m
        k: np.ndarray = self.k

        # Kinetic phase factor for a half step:
        #   T = hbar^2 * k^2 / (2*m)
        #   exp(-i * T * dt / (2*hbar)) = exp(-i * hbar * k^2 * dt / (4*m))
        kinetic_half: np.ndarray = np.exp(-1j * hbar * k ** 2 * dt / (4.0 * m))

        # Potential phase factor for a full step:
        #   exp(-i * V * dt / hbar)
        potential_full: np.ndarray = np.exp(-1j * V * dt / hbar)

        # --- Step 1: Half kinetic step in Fourier space ---
        psi_k: np.ndarray = fft(psi)
        psi_k *= kinetic_half
        psi: np.ndarray = ifft(psi_k)

        # --- Step 2: Full potential step in position space ---
        psi *= potential_full

        # --- Step 3: Half kinetic step in Fourier space ---
        psi_k = fft(psi)
        psi_k *= kinetic_half
        psi = ifft(psi_k)

        return psi
