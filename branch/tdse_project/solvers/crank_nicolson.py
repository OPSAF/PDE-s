"""Crank-Nicolson implicit solver for the 1D Time-Dependent Schrodinger Equation.

This module implements the Crank-Nicolson (CN) scheme, which is an unconditionally
stable, second-order accurate (in both space and time) implicit method for
solving the TDSE. It solves the linear system:

    A @ psi_new = B @ psi_old

where A and B are sparse tridiagonal matrices constructed from the discretized
Hamiltonian.
"""

from typing import Tuple
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from .base import TDSESolverBase


class CrankNicolson(TDSESolverBase):
    """Crank-Nicolson implicit solver for the 1D TDSE.

    The CN method is unconditionally stable and unitary, making it well-suited
    for long-time quantum propagation. It uses sparse LU decomposition
    (``scipy.sparse.linalg.splu``) for efficient solution of the implicit step.

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
        Solver name, set to "Crank-Nicolson".
    """

    def __init__(self, x: np.ndarray, hbar: float = 1.0, m: float = 1.0) -> None:
        super().__init__(x, hbar=hbar, m=m)
        self.name = "Crank-Nicolson"

    def build_cn_matrices(self, V: np.ndarray, dt: float) -> Tuple:
        """Build the Crank-Nicolson A and B tridiagonal matrices.

        Constructs the sparse matrices for the implicit (A) and explicit (B)
        halves of the CN scheme applied to interior grid points with Dirichlet
        boundary conditions (psi=0 at boundaries).

        The discretized Hamiltonian uses:

            d^2 psi / dx^2 ~ (psi[i+1] - 2*psi[i] + psi[i-1]) / dx^2

        With r = i * hbar * dt / (4 * m * dx^2):

            A_diag[i]  = 1 + 2*r + i*dt*V[i]/(2*hbar)   (implicit diagonal)
            A_off       = -r                               (implicit off-diagonal)
            B_diag[i]  = 1 - 2*r - i*dt*V[i]/(2*hbar)   (explicit diagonal)
            B_off       = r                                (explicit off-diagonal)

        Parameters
        ----------
        V : np.ndarray
            Potential energy array on the full spatial grid.
        dt : float
            Time step size.

        Returns
        -------
        tuple of (scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, scipy.sparse.linalg.SuperLU)
            The A matrix (implicit), B matrix (explicit), and the LU factorization of A.
        """
        nx_full: int = self.nx
        n_interior: int = nx_full - 2  # exclude boundaries
        dx: float = self.dx
        hbar: float = self.hbar
        m: float = self.m

        # Prefactor for kinetic term
        r: complex = 1j * hbar * dt / (4.0 * m * dx ** 2)

        # Interior potential values
        V_int: np.ndarray = V[1:-1]

        # Diagonal and off-diagonal entries for A (implicit) and B (explicit)
        # A = I + i*dt*H/(2*hbar),  B = I - i*dt*H/(2*hbar)
        diag_A: np.ndarray = 1.0 + 2.0 * r + 1j * dt * V_int / (2.0 * hbar)
        off_A: complex = -r
        diag_B: np.ndarray = 1.0 - 2.0 * r - 1j * dt * V_int / (2.0 * hbar)
        off_B: complex = r

        # Build sparse tridiagonal matrices (interior only)
        A = diags(
            [off_A * np.ones(n_interior - 1), diag_A, off_A * np.ones(n_interior - 1)],
            offsets=[-1, 0, 1],
            shape=(n_interior, n_interior),
            format="csr",
        )
        B = diags(
            [off_B * np.ones(n_interior - 1), diag_B, off_B * np.ones(n_interior - 1)],
            offsets=[-1, 0, 1],
            shape=(n_interior, n_interior),
            format="csr",
        )

        # LU factorization of A for efficient repeated solves
        lu_A = splu(A)

        return A, B, lu_A

    def propagate(self, psi: np.ndarray, V: np.ndarray, dt: float) -> np.ndarray:
        """Propagate the wavefunction by one time step using Crank-Nicolson.

        Extracts interior points, builds CN matrices, solves the linear system,
        and reconstructs the full wavefunction with zero boundary conditions.

        Parameters
        ----------
        psi : np.ndarray
            Current wavefunction on the full spatial grid (complex-valued).
        V : np.ndarray
            Potential energy array on the full spatial grid.
        dt : float
            Time step size.

        Returns
        -------
        np.ndarray
            Wavefunction after one CN time step, with zero boundary conditions.
        """
        _, _, lu_A = self.build_cn_matrices(V, dt)

        # Extract interior (boundaries remain zero)
        psi_interior: np.ndarray = psi[1:-1].copy()

        # Build B matrix again to apply explicit step
        # (we could cache this; rebuilding is simple and avoids state)
        n_interior: int = self.nx - 2
        dx: float = self.dx
        hbar: float = self.hbar
        m: float = self.m
        r: complex = 1j * hbar * dt / (4.0 * m * dx ** 2)
        V_int: np.ndarray = V[1:-1]
        diag_B: np.ndarray = 1.0 - 2.0 * r - 1j * dt * V_int / (2.0 * hbar)
        off_B: complex = r

        from scipy.sparse import diags as _diags

        B = _diags(
            [off_B * np.ones(n_interior - 1), diag_B, off_B * np.ones(n_interior - 1)],
            offsets=[-1, 0, 1],
            shape=(n_interior, n_interior),
            format="csr",
        )

        # RHS: B @ psi_old (interior)
        rhs: np.ndarray = B.dot(psi_interior)

        # Solve A @ psi_new = rhs
        psi_new_interior: np.ndarray = lu_A.solve(rhs)

        # Reconstruct full wavefunction with zero boundaries
        psi_new: np.ndarray = np.zeros(self.nx, dtype=np.complex128)
        psi_new[1:-1] = psi_new_interior

        return psi_new
