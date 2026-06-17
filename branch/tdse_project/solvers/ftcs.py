"""Forward-Time Central-Space (FTCS) explicit solver for the 1D TDSE.

WARNING: This method is UNSTABLE for the Schrodinger equation!

The FTCS scheme is included here purely for educational / demonstration
purposes, to illustrate how an naive explicit discretization of the TDSE
fails due to numerical instability. The norm of the wavefunction will grow
exponentially for any finite time step.

For stable propagation, use CrankNicolson or SplitStepFourier instead.
"""

import numpy as np

from .base import TDSESolverBase


class FTCS(TDSESolverBase):
    """Forward-Time Central-Space explicit solver for the 1D TDSE (UNSTABLE).

    .. warning::
        This method is **unconditionally unstable** for the time-dependent
        Schrodinger equation. The wavefunction norm will grow without bound
        regardless of how small the time step is chosen. This solver exists
        solely as a pedagogical example of what **not** to do.

    The update formula (applied to interior points only) is:

        psi_new[i] = psi[i]
            + (i*dt/hbar) * (hbar^2/(2m)) * (psi[i+1] - 2*psi[i] + psi[i-1]) / dx^2
            - (i*dt/hbar) * V[i] * psi[i]

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
        Solver name, set to "FTCS (Unstable)".
    """

    def __init__(self, x: np.ndarray, hbar: float = 1.0, m: float = 1.0) -> None:
        super().__init__(x, hbar=hbar, m=m)
        self.name = "FTCS (Unstable)"

    def propagate(self, psi: np.ndarray, V: np.ndarray, dt: float) -> np.ndarray:
        """Propagate the wavefunction by one time step using FTCS (unstable).

        Applies the forward-time central-space stencil to interior grid points.
        Boundary values are held at zero (Dirichlet conditions).

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
            Wavefunction after one FTCS time step. The result will exhibit
            growing instability after a few steps.
        """
        psi_new: np.ndarray = np.array(psi, dtype=complex, copy=True)
        dx: float = self.dx
        hbar: float = self.hbar
        m: float = self.m

        # Prefactor for kinetic term: i * dt * hbar^2 / (2 * m * hbar * dx^2)
        #   = i * dt * hbar / (2 * m * dx^2)
        kin_factor: complex = 1j * dt * hbar / (2.0 * m * dx ** 2)

        # Prefactor for potential term: -i * dt / hbar
        pot_factor: complex = -1j * dt / hbar

        # Apply stencil to interior points only (boundaries stay zero)
        for i in range(1, self.nx - 1):
            laplacian: complex = psi[i + 1] - 2.0 * psi[i] + psi[i - 1]
            psi_new[i] = psi[i] + kin_factor * laplacian + pot_factor * V[i] * psi[i]

        return psi_new
