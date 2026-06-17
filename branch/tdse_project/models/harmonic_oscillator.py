"""
Harmonic Oscillator models with exact analytical solutions for the 1D TDSE.

This module provides two analytically solvable models for the quantum harmonic
oscillator: the stationary ground state and a coherent (Gaussian) state that
oscillates without changing shape.

TDSE: i * dpsi/dt = -(1/2) * d²psi/dx² + V(x) * psi
with dimensionless units: hbar = m = omega = 1.
Potential: V(x) = x² / 2

References:
    - Griffiths, "Introduction to Quantum Mechanics", Ch. 2
    - Sakurai, "Modern Quantum Mechanics", coherent states section
"""

import numpy as np
from typing import Union


class HarmonicOscillatorGroundState:
    """
    Ground state of the quantum harmonic oscillator.

    The ground state is a stationary state, so it only picks up a time-dependent
    phase factor exp(-i*E_0*t/hbar) with E_0 = 1/2 (in dimensionless units).

    Attributes:
        name (str): Model identifier.
        description (str): Human-readable description of the model.
    """

    def __init__(self) -> None:
        """Initialize the HarmonicOscillatorGroundState model."""
        self.name: str = "Harmonic Oscillator Ground State"
        self.description: str = (
            "Stationary ground state of the quantum harmonic oscillator. "
            "The wavefunction is a Gaussian that remains static in shape, "
            "only acquiring a phase factor exp(-i*t/2) over time."
        )

    def potential(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the harmonic oscillator potential V(x) = x^2 / 2.

        Args:
            x: Spatial coordinate array.

        Returns:
            Potential energy array V(x).
        """
        return x**2 / 2.0

    def initial_wavefunction(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the initial ground-state wavefunction psi_0(x).

        The normalized ground state of the HO (with hbar=m=omega=1):
            psi_0(x) = pi^(-1/4) * exp(-x^2 / 2)

        Args:
            x: Spatial coordinate array.

        Returns:
            Complex-valued initial wavefunction array.
        """
        return np.pi**(-0.25) * np.exp(-x**2 / 2.0)

    def analytical_solution(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the exact time-dependent solution for the ground state.

        Since the ground state is an energy eigenstate with E_0 = hbar*omega/2 = 1/2,
        the time evolution is simply a global phase:
            psi(x, t) = psi_0(x) * exp(-i * E_0 * t)
                      = pi^(-1/4) * exp(-x^2/2) * exp(-i*t/2)

        The probability density |psi|^2 remains unchanged for all times.

        Args:
            x: Spatial coordinate array.
            t: Time value (dimensionless).

        Returns:
            Complex-valued wavefunction at time t.
        """
        return self.initial_wavefunction(x) * np.exp(-1j * t / 2.0)


class HarmonicOscillatorCoherentState:
    """
    Coherent (Gaussian displaced) state of the quantum harmonic oscillator.

    A coherent state is a minimum-uncertainty Gaussian wavepacket whose center
    follows the classical equations of motion while maintaining its shape.
    For the harmonic oscillator, this means simple harmonic oscillation with
    angular frequency omega=1 (in our dimensionless units).

    The classical trajectory of the wavepacket center is:
        x_c(t) = x0*cos(t) + k0*sin(t)
        p_c(t) = -x0*sin(t) + k0*cos(t)   (= k_c(t), since m=1)

    Attributes:
        name (str): Model identifier.
        description (str): Human-readable description of the model.
        x0 (float): Initial position offset of the wavepacket center.
        k0 (float): Initial momentum (wave number) of the wavepacket.
    """

    def __init__(self, x0: float = 2.0, k0: float = 0.0) -> None:
        """
        Initialize the HarmonicOscillatorCoherentState model.

        Args:
            x0: Initial displacement of the wavepacket center from origin.
                Default is 2.0.
            k0: Initial wave number (momentum in dimensionless units).
                Default is 0.0.
        """
        self.x0: float = x0
        self.k0: float = k0
        self.name: str = "Harmonic Oscillator Coherent State"
        self.description: str = (
            f"Coherent state of the quantum harmonic oscillator with "
            f"initial displacement x0={x0} and initial momentum k0={k0}. "
            f"The wavepacket oscillates classically without spreading."
        )

    def potential(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the harmonic oscillator potential V(x) = x^2 / 2.

        Args:
            x: Spatial coordinate array.

        Returns:
            Potential energy array V(x).
        """
        return x**2 / 2.0

    def initial_wavefunction(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the initial coherent-state wavefunction psi_0(x).

        A minimum-uncertainty Gaussian centered at x=x0 with momentum k0,
        matching the ground-state width sigma = 1/sqrt(2):

            psi_0(x) = pi^(-1/4) * exp(-(x - x0)^2 / 2 + i*k0*(x - x0))

        This is a displaced ground state (coherent state with alpha = (x0 + i*k0)/sqrt(2)).

        Args:
            x: Spatial coordinate array.

        Returns:
            Complex-valued initial wavefunction array.
        """
        # Gaussian envelope centered at x0
        gaussian = np.exp(-(x - self.x0)**2 / 2.0)
        # Phase from initial momentum k0 (plane-wave-like tilt)
        phase = np.exp(1j * self.k0 * (x - self.x0))
        return np.pi**(-0.25) * gaussian * phase

    def analytical_solution(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the exact time-dependent coherent-state solution.

        The coherent state evolves such that its center follows the classical
        trajectory while the Gaussian shape is preserved:

            x_c(t) =  x0*cos(t) + k0*sin(t)   (position of the center)
            k_c(t) = -x0*sin(t) + k0*cos(t)   (momentum/wave-number of the center)

            psi(x,t) = pi^(-1/4) * exp( -(x - x_c(t))^2 / 2
                                        + i*k_c(t)*(x - x_c(t)/2)
                                        - i*t/2 )

        The term -i*t/2 is the zero-point energy phase (E_0 = 1/2).
        The linear-in-x phase term i*k_c*(x - x_c/2) encodes both the local
        momentum and a position-dependent correction that ensures the state
        remains exactly coherent (minimum uncertainty) at all times.

        Note: This form can be verified by checking that it satisfies the TDSE
        and reduces to initial_wavefunction(x) at t=0.

        Args:
            x: Spatial coordinate array.
            t: Time value (dimensionless).

        Returns:
            Complex-valued wavefunction at time t.
        """
        # Classical trajectory of the wavepacket center
        x_center = self.x0 * np.cos(t) + self.k0 * np.sin(t)
        k_center = -self.x0 * np.sin(t) + self.k0 * np.cos(t)

        # Gaussian envelope centered at the classical position
        gaussian = np.exp(-(x - x_center)**2 / 2.0)

        # Phase factors:
        #   1) Local momentum contribution: k_c * (x - x_c/2)
        #      This gives the wavepacket its momentum and keeps it coherent
        #   2) Zero-point energy phase: -t/2
        phase = np.exp(
            1j * k_center * (x - x_center / 2.0)
            - 1j * t / 2.0
        )

        return np.pi**(-0.25) * gaussian * phase
