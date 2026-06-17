"""
Constant Force (linear potential) model for 1D TDSE.

TDSE: i * dpsi/dt = -(1/2) * d²psi/dx² + V(x) * psi
Units: dimensionless (hbar = m = 1)
Potential: V(x) = F * x

The exact solution is computed via the analytically-known quantum propagator
(Green's function) for a uniform force field:

  K(x, x', t) = sqrt(1/(2*pi*i*t)) *
               exp{ i * [(x-x')^2/(2t) + F*(x+x')*t/2 - F^2*t^3/24] }

The time-dependent wavefunction is obtained by convolving the initial
Gaussian wavepacket with this propagator. For a Gaussian initial condition,
the integral can be evaluated in closed form (performed symbolically).

Reference: Landau & Lifshitz QM Sec. 24; Schiff QM Sec. 28;
           Gol'dman et al., "Problems in Quantum Mechanics" Problem 4.10
"""

import numpy as np
from typing import Union


class ConstantForce:
    """
    Gaussian wavepacket under constant uniform force V(x) = F*x.

    Classical dynamics (force = -dV/dx = -F):
        x_cl(t) = x0 + k0*t - F*t^2/2
        p_cl(t) = k0 - F*t

    The quantum solution combines classical trajectory motion with
    free-particle spreading, plus a gauge phase from the linear potential.
    """

    def __init__(self, F: float = 0.5) -> None:
        self.F = F
        self.x0 = 0.0
        self.sigma0 = 1.0 / np.sqrt(2.0)
        self.k0 = 0.0
        self.name = "Constant Force (Linear Potential)"
        self.description = f"Gaussian wavepacket under V(x)={F}*x"

    def potential(self, x: np.ndarray) -> np.ndarray:
        """V(x) = F * x."""
        return self.F * x

    def initial_wavefunction(self, x: np.ndarray) -> np.ndarray:
        """
        Initial Gaussian wavepacket: psi_0(x) = pi^(-1/4) * exp(-x^2/4).
        Minimum-uncertainty Gaussian centered at origin, width sigma0 = 1/sqrt(2).
        """
        s2 = self.sigma0 ** 2
        prefactor = (2.0 * np.pi * s2) ** (-0.25)
        return prefactor * np.exp(-(x - self.x0) ** 2 / (4.0 * s2))

    def analytical_solution(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Exact solution via propagator convolution (closed-form result).

        The propagator for V(x) = F*x is:
          K(x,x',t) = (2*pi*i*t)^(-1/2) *
                       exp{i*[(x-x')^2/(2t) + F*(x+x')*t/2 - F^2*t^3/24]}

        Convolution with Gaussian psi_0(x') = N*exp(-x'^2/(4*s0^2)) yields:

          psi(x,t) = N(t) * exp(-(x - x_c)^2 / (4*s0^2*gamma)) * exp(i*Phi)

        where:
          gamma   = 1 + i*t/(2*s0^2)              [free-particle spreading]
          x_c     = -F*t^2/2                       [classical position]
          Phi     = F*x*t - F^2*t^3/6              [gauge + action phase]
          N(t)    = (2*pi*s0^2)^(-1/4) / sqrt(gamma)
        """
        F = self.F
        s2 = self.sigma0 ** 2  # = 1/2

        # Free-particle spreading parameter
        gamma = 1.0 + 1j * t / (2.0 * s2)  # = 1 + i*t

        # Classical trajectory
        x_cl = -0.5 * F * t ** 2
        x_rel = x - x_cl

        # Amplitude: normalization + Gaussian envelope (free-particle shape)
        prefactor = (2.0 * np.pi * s2) ** (-0.25) / np.sqrt(gamma)
        gaussian = np.exp(-x_rel ** 2 / (4.0 * s2 * gamma))

        # Phase factor from the propagator evaluation.
        #
        # The full closed-form result of the propagator-Gaussian convolution gives:
        #   Phi_total = F*x*t - S_cl(t)
        # where S_cl(t) = F^2*t^3/6 is the classical action along x_cl(t).
        #
        # This form satisfies:
        #   (1) psi(x,0) = psi_0(x)  [initial condition]
        #   (2) |psi|^2 integrates to 1  [norm conservation]
        #   (3) <x> follows classical trajectory asymptotically
        S_cl = F ** 2 * t ** 3 / 6.0
        phase = np.exp(1j * F * x * t - 1j * S_cl)

        return prefactor * gaussian * phase
