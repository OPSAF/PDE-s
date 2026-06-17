"""
Free Particle Gaussian Wavepacket model with exact analytical solution for 1D TDSE.

TDSE: i * dpsi/dt = -(1/2) * d²psi/dx² + V(x) * psi
Units: dimensionless (hbar = m = 1)
Potential: V(x) = 0

This is one of the few exactly-solvable time-dependent problems.
A Gaussian wavepacket spreads over time while its center moves at
constant velocity (classical free motion).

Analytical solution derived via Fourier transform method:
  psi(x,t) = N(t) * exp(-(x-x_cl(t))^2 / (4*sigma0^2*gamma)) * exp(i*phi)

Reference: Griffiths "Introduction to QM" Sec. 2.4; Sakurai "Modern QM" Sec. 2.1.4
"""

import numpy as np
from typing import Union


class FreeParticle:
    """
    Free Gaussian wavepacket (V = 0).

    Demonstrates quantum wavepacket spreading: the spatial width grows
    as sqrt(sigma0^2 + t^2/(4*sigma0^2)), while the center follows
    the classical trajectory x_cl(t) = x0 + (hbar*k0/m)*t.

    Attributes:
        name (str): Model identifier.
        description (str): Human-readable description.
        x0 (float): Initial center position.
        k0 (float): Initial wave number (momentum/hbar).
        sigma0 (float): Initial Gaussian width parameter.
    """

    def __init__(self, x0: float = 0.0, k0: float = 1.0,
                 sigma0: float = None) -> None:
        self.x0 = x0
        self.k0 = k0
        # Default width matches harmonic oscillator ground state (minimum uncertainty)
        self.sigma0 = sigma0 if sigma0 is not None else 1.0 / np.sqrt(2.0)
        self.name = "Free Particle Gaussian"
        self.description = (
            f"Free Gaussian wavepacket (V=0), x0={x0}, k0={k0}"
        )

    def potential(self, x: np.ndarray) -> np.ndarray:
        """V(x) = 0 (free particle)."""
        return np.zeros_like(x)

    def initial_wavefunction(self, x: np.ndarray) -> np.ndarray:
        """
        Minimum-uncertainty Gaussian wavepacket at t=0.

        psi_0(x) = (2*pi*sigma0^2)^(-1/4) * exp(-(x-x0)^2/(4*sigma0^2) + i*k0*(x-x0))
        """
        s2 = self.sigma0 ** 2
        prefactor = (2.0 * np.pi * s2) ** (-0.25)
        gaussian = np.exp(-(x - self.x0) ** 2 / (4.0 * s2))
        plane_wave = np.exp(1j * self.k0 * (x - self.x0))
        return prefactor * gaussian * plane_wave

    def analytical_solution(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Exact time-dependent solution for free-particle Gaussian spreading.

        The solution is obtained by Fourier transform:
          psi_tilde(p,t) = psi_tilde_0(p) * exp(-i*p^2*t/(2*m*hbar))

        For a Gaussian initial packet, this gives the closed-form result:

          psi(x,t) = (2*pi*s_t^2)^(-1/4) *
                     exp(-(x - x_cl(t))^2 / (4*s_t^2)) *
                     exp(i*[k0*(x - x_cl(t)/2) - E_k*t/hbar])

        where:
          gamma   = 1 + i*hbar*t/(2*m*sigma0^2)     [complex spreading]
          s_t^2   = sigma0^2 * gamma                   [complex variance]
          x_cl(t) = x0 + hbar*k0*t/m                  [classical trajectory]
          E_k     = hbar^2*k0^2/(2m)                   [kinetic energy]

        With hbar=m=1:
          gamma = 1 + i*t/(2*sigma0^2)
          x_cl  = x0 + k0*t
        """
        s2 = self.sigma0 ** 2
        hbar, m = 1.0, 1.0

        # Complex spreading parameter
        gamma = 1.0 + 1j * hbar * t / (2.0 * m * s2)

        # Classical trajectory (center moves at constant velocity v = hbar*k0/m)
        x_cl = self.x0 + hbar * self.k0 * t / m

        # Coordinate relative to moving center
        x_rel = x - x_cl

        # Normalization (includes spreading)
        prefactor = (2.0 * np.pi * s2) ** (-0.25) / np.sqrt(gamma)

        # Spreading Gaussian envelope
        gaussian = np.exp(-x_rel ** 2 / (4.0 * s2 * gamma))

        # Phase factor: plane-wave momentum + kinetic energy
        E_kinetic = (hbar * self.k0) ** 2 / (2.0 * m)
        phase = np.exp(
            1j * self.k0 * (x - self.x0)
            - 1j * E_kinetic * t / hbar
        )

        return prefactor * gaussian * phase
