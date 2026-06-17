"""Base class and utility functions for 1D Time-Dependent Schrodinger Equation solvers.

This module provides the abstract base class that all TDSE solvers inherit from,
along with a trapezoidal integration utility function.
"""

from abc import ABC, abstractmethod
import numpy as np


class TDSESolverBase(ABC):
    """Abstract base class for TDSE (Time-Dependent Schrodinger Equation) solvers.

    All concrete solver implementations must inherit from this class and implement
    the ``propagate`` method. The base class handles grid setup, normalization,
    and provides a common interface for 1D TDSE propagation:

        i * hbar * dpsi/dt = -(hbar^2 / 2m) * d^2psi/dx^2 + V(x) * psi

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
    x : np.ndarray
        Spatial grid.
    nx : int
        Number of grid points.
    dx : float
        Grid spacing.
    hbar : float
        Reduced Planck constant.
    m : float
        Particle mass.
    name : str
        Solver name identifier.
    """

    def __init__(self, x: np.ndarray, hbar: float = 1.0, m: float = 1.0) -> None:
        self.x: np.ndarray = x
        self.nx: int = len(x)
        self.dx: float = x[1] - x[0]
        self.hbar: float = hbar
        self.m: float = m
        self.name: str = "Base"

    @abstractmethod
    def propagate(self, psi: np.ndarray, V: np.ndarray, dt: float) -> np.ndarray:
        """Propagate the wavefunction by one time step.

        Parameters
        ----------
        psi : np.ndarray
            Current wavefunction array (complex-valued).
        V : np.ndarray
            Potential energy array on the spatial grid.
        dt : float
            Time step size.

        Returns
        -------
        np.ndarray
            Wavefunction after propagation by one time step.
        """
        pass

    def normalize(self, psi: np.ndarray) -> np.ndarray:
        """Normalize the wavefunction so that integral |psi|^2 dx = 1.

        Parameters
        ----------
        psi : np.ndarray
            Wavefunction to normalize.

        Returns
        -------
        np.ndarray
            Normalized wavefunction. If the norm is zero, returns the input unchanged.
        """
        norm: float = np.sqrt(trapz(np.abs(psi) ** 2, self.x))
        if norm > 0:
            return psi / norm
        return psi


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Compute the definite integral using the trapezoidal rule.

    Assumes uniform spacing in the x array.

    Parameters
    ----------
    y : np.ndarray
        Integrand values sampled on the grid.
    x : np.ndarray
        Coordinate grid with uniform spacing.

    Returns
    -------
    float
        Approximate value of the integral of y over x.
    """
    dx: float = x[1] - x[0]
    return float(dx * (np.sum(y) - 0.5 * (y[0] + y[-1])))
