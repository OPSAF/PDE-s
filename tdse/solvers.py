"""
TDSE Solvers Module
===================

Time-dependent Schrodinger equation solvers for 1D and 2D simulations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
from scipy.linalg import solve_banded
from tqdm import tqdm

from tdse.potentials import normalize, probability_mass, print_section, grid, l1_l2_linf_error, exact_free_gaussian, gaussian_wavepacket, potential_free, mass_2d

Array = np.ndarray


# =============================================================================
# Solver Base Classes
# =============================================================================

class Solver1D(ABC):
    """
    Abstract base class for 1D TDSE solvers.

    This class defines the interface for all 1D time-stepping methods.
    Subclasses must implement the step method.
    """

    def __init__(self, name: str, stable: bool = True, unitary: bool = False):
        self.name = name
        self.stable = stable
        self.unitary = unitary

    @abstractmethod
    def step(self, psi: Array, v: Array, x: Array, dx: float, dt: float) -> Array:
        """
        Perform one time step.

        Args:
            psi: Current wave function
            v: Potential array
            x: Spatial grid
            dx: Grid spacing
            dt: Time step

        Returns:
            psi: Wave function at next time step
        """
        pass

    def info(self) -> str:
        """Return solver information."""
        stability = "unconditionally stable" if self.stable else "conditionally stable"
        unitary = " (unitary)" if self.unitary else ""
        return f"{self.name}: {stability}{unitary}"


class FTCSSolver(Solver1D):
    """Forward-Time Centered-Space (explicit FTCS) solver."""

    def __init__(self):
        super().__init__("FTCS", stable=False, unitary=False)

    def step(self, psi: Array, v: Array, x: Array, dx: float, dt: float) -> Array:
        return step_ftcs(psi, v, dx, dt)


class BackwardEulerSolver(Solver1D):
    """Backward Euler (implicit) solver."""

    def __init__(self):
        super().__init__("Backward-Euler", stable=True, unitary=False)

    def step(self, psi: Array, v: Array, x: Array, dx: float, dt: float) -> Array:
        return step_backward_euler(psi, v, dx, dt)


class CrankNicolsonSolver(Solver1D):
    """Crank-Nicolson (implicit, unitary) solver."""

    def __init__(self):
        super().__init__("Crank-Nicolson", stable=True, unitary=True)

    def step(self, psi: Array, v: Array, x: Array, dx: float, dt: float) -> Array:
        return step_crank_nicolson(psi, v, dx, dt)


class RKK4Solver(Solver1D):
    """4th-order Runge-Kutta solver."""

    def __init__(self):
        super().__init__("RK4", stable=True, unitary=False)

    def step(self, psi: Array, v: Array, x: Array, dx: float, dt: float) -> Array:
        return step_rk4(psi, v, dx, dt)


class SplitStepFFTSolver(Solver1D):
    """Split-Step Fourier method (spectral) solver."""

    def __init__(self):
        super().__init__("Split-Step-FFT", stable=True, unitary=True)

    def step(self, psi: Array, v: Array, x: Array, dx: float, dt: float) -> Array:
        return step_split_step_fft(psi, v, x, dx, dt)


class TDSESolverFactory:
    """
    Factory class for creating TDSE solvers.

    This factory provides a unified interface for creating various
    solver types for both 1D and 2D simulations.

    Example:
        >>> factory = TDSESolverFactory()
        >>> solver = factory.create("CN")  # Crank-Nicolson
        >>> solver = factory.create("SSF")  # Split-Step FFT
    """

    _solvers = {
        "FTCS": FTCSSolver,
        "BE": BackwardEulerSolver,
        "CN": CrankNicolsonSolver,
        "RK4": RKK4Solver,
        "SSF": SplitStepFFTSolver,
        "Split-Step-FFT": SplitStepFFTSolver,
        "Crank-Nicolson": CrankNicolsonSolver,
        "Backward-Euler": BackwardEulerSolver,
    }

    @classmethod
    def create(cls, method: str) -> Solver1D:
        """
        Create a solver object.

        Args:
            method: Solver method name
                - "FTCS" or "ftcs": Forward-Time Centered-Space
                - "BE" or "backward-euler": Backward Euler
                - "CN" or "crank-nicolson": Crank-Nicolson
                - "RK4" or "rk4": 4th-order Runge-Kutta
                - "SSF" or "split-step-fft": Split-Step Fourier

        Returns:
            Solver1D: Configured solver object

        Raises:
            ValueError: If method is not recognized
        """
        method_upper = method.upper()
        method_lower = method.lower()

        for key in cls._solvers:
            if key.upper() == method_upper or key.lower() == method_lower:
                return cls._solvers[key]()

        raise ValueError(f"Unknown solver method: {method}. "
                         f"Available: {list(set(cls._solvers.keys()))}")

    @classmethod
    def list_solvers(cls) -> List[str]:
        """List unique available solver names."""
        return list(set(cls._solvers.keys()))

    @classmethod
    def get_all_solvers(cls) -> List[Solver1D]:
        """Get instances of all available solvers."""
        return [s() for s in set(cls._solvers.values())]


# =============================================================================
# Finite-difference operators and solvers
# =============================================================================

def laplacian_dirichlet(psi: Array, dx: float) -> Array:
    """Second-order centered Laplacian with homogeneous Dirichlet endpoints."""
    out = np.zeros_like(psi, dtype=complex)
    out[1:-1] = (psi[:-2] - 2.0 * psi[1:-1] + psi[2:]) / dx**2
    return out


def hamiltonian_apply(psi: Array, v: Array, dx: float) -> Array:
    return -0.5 * laplacian_dirichlet(psi, dx) + v * psi


def banded_hamiltonian(v: Array, dx: float) -> Array:
    """Banded matrix for H = -1/2 Dxx + V with Dirichlet endpoints."""
    n = len(v)
    main = np.ones(n) / dx**2 + v
    off = -0.5 * np.ones(n - 1) / dx**2
    ab = np.zeros((3, n), dtype=complex)
    ab[0, 1:] = off
    ab[1, :] = main
    ab[2, :-1] = off
    return ab


def apply_tridiagonal(ab: Array, psi: Array) -> Array:
    """Apply a tridiagonal banded matrix stored in scipy solve_banded layout."""
    out = ab[1] * psi
    out[1:] += ab[2, :-1] * psi[:-1]
    out[:-1] += ab[0, 1:] * psi[1:]
    return out


def step_ftcs(psi: Array, v: Array, dx: float, dt: float) -> Array:
    """Explicit forward Euler in time, centered second difference in space."""
    return psi - 1j * dt * hamiltonian_apply(psi, v, dx)


def step_backward_euler(psi: Array, v: Array, dx: float, dt: float) -> Array:
    """Implicit backward Euler: (I + i dt H) psi^{n+1} = psi^n."""
    h = banded_hamiltonian(v, dx)
    a = h.copy()
    a[1] = 1.0 + 1j * dt * h[1]
    a[0] = 1j * dt * h[0]
    a[2] = 1j * dt * h[2]
    return solve_banded((1, 1), a, psi)


def step_crank_nicolson(psi: Array, v: Array, dx: float, dt: float) -> Array:
    """Crank-Nicolson: (I+i dt H/2) psi^{n+1}=(I-i dt H/2) psi^n."""
    h = banded_hamiltonian(v, dx)
    left = h.copy()
    left[1] = 1.0 + 0.5j * dt * h[1]
    left[0] = 0.5j * dt * h[0]
    left[2] = 0.5j * dt * h[2]

    right = h.copy()
    right[1] = 1.0 - 0.5j * dt * h[1]
    right[0] = -0.5j * dt * h[0]
    right[2] = -0.5j * dt * h[2]
    rhs = apply_tridiagonal(right, psi)
    return solve_banded((1, 1), left, rhs)


def step_rk4(psi: Array, v: Array, dx: float, dt: float) -> Array:
    """Classical RK4 for psi_t = -i H psi with second-order spatial stencil."""
    def f(y: Array) -> Array:
        return -1j * hamiltonian_apply(y, v, dx)

    k1 = f(psi)
    k2 = f(psi + 0.5 * dt * k1)
    k3 = f(psi + 0.5 * dt * k2)
    k4 = f(psi + dt * k3)
    return psi + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def step_split_step_fft(psi: Array, v: Array, x: Array, dx: float, dt: float) -> Array:
    """
    Strang split-step Fourier:
        exp(-i V dt/2) -> FFT -> exp(-i k^2 dt/2) -> IFFT -> exp(-i V dt/2)
    because H_kin = k^2/2 in Fourier space.
    """
    k = 2.0 * np.pi * np.fft.fftfreq(len(x), d=dx)
    psi_half = np.exp(-0.5j * v * dt) * psi
    psi_k = np.fft.fft(psi_half)
    psi_k *= np.exp(-0.5j * k**2 * dt)
    psi_new = np.fft.ifft(psi_k)
    psi_new *= np.exp(-0.5j * v * dt)
    return psi_new


def step_split_step_fft_2d(
    psi: Array, V: Array, KX: Array, KY: Array, dt: float
) -> Array:
    """
    2D Strang split-step Fourier for i psi_t = -1/2 (psi_xx + psi_yy) + V(x,y) psi.

    Potential half-step: exp(-i V dt/2).
    Kinetic step in Fourier space: exp(-i (kx^2+ky^2)/2 * dt).
    KX, KY must be 2D meshgrid arrays built with numpy.fft.fftfreq and ij-indexing.
    Uses numpy.fft.fft2 / ifft2 (periodic boundary conditions).
    """
    psi_half = np.exp(-0.5j * V * dt) * psi
    psi_k = np.fft.fft2(psi_half)
    psi_k *= np.exp(-0.5j * (KX ** 2 + KY ** 2) * dt)
    psi_new = np.fft.ifft2(psi_k)
    psi_new *= np.exp(-0.5j * V * dt)
    return psi_new


def step_adi_2d(psi: Array, V: Array, dx: float, dy: float, dt: float) -> Array:
    """
    2D Alternating Direction Implicit (ADI) method for time-dependent Schrödinger equation.
    
    Solves: i ∂ψ/∂t = -1/2 (∂²ψ/∂x² + ∂²ψ/∂y²) + V(x,y)ψ
    
    The ADI scheme splits the Hamiltonian into x and y components:
        H = H_x + H_y where H_x = -1/2 ∂²/∂x² + V/2 and H_y = -1/2 ∂²/∂y² + V/2
    
    Time step: exp(-i H dt) ≈ exp(-i H_x dt/2) exp(-i H_y dt) exp(-i H_x dt/2)
    
    Args:
        psi: Current wavefunction (2D array)
        V: Potential array (2D array)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction
        dt: Time step
    
    Returns:
        psi_new: Wavefunction at next time step
    """
    nx, ny = psi.shape
    
    # Precompute diagonal and off-diagonal for tridiagonal solver
    # For x-direction (rows)
    diag_x = np.ones(nx, dtype=complex) / dx**2 + V[0, :]
    off_x = -0.5 * np.ones(nx - 1, dtype=complex) / dx**2
    
    # For y-direction (columns)
    diag_y = np.ones(ny, dtype=complex) / dy**2
    off_y = -0.5 * np.ones(ny - 1, dtype=complex) / dy**2
    
    # First half-step in x-direction: (I + i dt/2 H_x) psi = psi_prev
    psi_temp = np.zeros_like(psi)
    for j in range(ny):
        # Build banded matrix for column j
        ab = np.zeros((3, nx), dtype=complex)
        ab[0, 1:] = 0.25j * dt * off_x
        ab[1, :] = 1.0 + 0.5j * dt * (1.0 / dx**2 + 0.5 * V[:, j])
        ab[2, :-1] = 0.25j * dt * off_x
        rhs = psi[:, j] * (1.0 - 0.5j * dt * 0.5 * V[:, j])
        psi_temp[:, j] = solve_banded((1, 1), ab, rhs)
    
    # Full step in y-direction: (I + i dt H_y) psi = psi_temp
    psi_new = np.zeros_like(psi)
    for i in range(nx):
        # Build banded matrix for row i
        ab = np.zeros((3, ny), dtype=complex)
        ab[0, 1:] = 0.5j * dt * off_y
        ab[1, :] = 1.0 + 0.5j * dt * (1.0 / dy**2 + 0.5 * V[i, :])
        ab[2, :-1] = 0.5j * dt * off_y
        rhs = psi_temp[i, :] * (1.0 - 0.5j * dt * 0.5 * V[i, :])
        psi_new[i, :] = solve_banded((1, 1), ab, rhs)
    
    # Second half-step in x-direction: (I + i dt/2 H_x) psi = psi_new
    for j in range(ny):
        ab = np.zeros((3, nx), dtype=complex)
        ab[0, 1:] = 0.25j * dt * off_x
        ab[1, :] = 1.0 + 0.5j * dt * (1.0 / dx**2 + 0.5 * V[:, j])
        ab[2, :-1] = 0.25j * dt * off_x
        rhs = psi_new[:, j] * (1.0 - 0.5j * dt * 0.5 * V[:, j])
        psi_new[:, j] = solve_banded((1, 1), ab, rhs)
    
    return psi_new


def solve(
    method: str,
    psi0: Array,
    v: Array,
    x: Array,
    t: Array,
    dx: float,
    dt: float,
    store_every: int = 1,
    renormalize: bool = False,
) -> Tuple[Array, Array]:
    """
    Unified solver interface required by the prompt.

    Returns:
        saved_t, saved_psi
    """
    method_key = method.lower()
    psi = psi0.astype(complex).copy()
    saved_t = [float(t[0])]
    saved_psi = [psi.copy()]

    for n in range(1, len(t)):
        if method_key == "ftcs":
            psi = step_ftcs(psi, v, dx, dt)
        elif method_key in ("backward-euler", "be"):
            psi = step_backward_euler(psi, v, dx, dt)
        elif method_key in ("crank-nicolson", "cn"):
            psi = step_crank_nicolson(psi, v, dx, dt)
        elif method_key in ("split-step", "split-step-fft", "fft", "ssf"):
            psi = step_split_step_fft(psi, v, x, dx, dt)
        elif method_key == "rk4":
            psi = step_rk4(psi, v, dx, dt)
        else:
            raise ValueError(f"Unknown method: {method}")

        if renormalize and method_key not in ("ftcs",):
            psi = normalize(psi, dx)
        if n % store_every == 0 or n == len(t) - 1:
            saved_t.append(float(t[n]))
            saved_psi.append(psi.copy())

    return np.array(saved_t), np.array(saved_psi)


def solve_2d(
    psi0: Array,
    V: Array,
    KX: Array,
    KY: Array,
    t: Array,
    dt: float,
    dx: float,
    dy: float,
    store_every: int = 1,
    progress: bool = False,
    method: str = "split-step-fft",
) -> Tuple[List[float], List[Array]]:
    """
    2D time evolution solver.
    
    Args:
        psi0: Initial wavefunction
        V: Potential array
        KX, KY: Fourier wavevectors (only needed for split-step-fft)
        t: Time array
        dt: Time step
        dx, dy: Grid spacings
        store_every: Store every n-th time step
        progress: Show progress bar
        method: Solver method - "split-step-fft" (default) or "adi"
    
    Returns:
        Lists of times and psi(t)
    """
    psi = psi0.astype(complex).copy()
    saved_t = [float(t[0])]
    saved_psi = [psi.copy()]
    
    method_key = method.lower()

    for n in tqdm(range(1, len(t)), desc="2D evolution", disable=not progress):
        if method_key == "adi":
            psi = step_adi_2d(psi, V, dx, dy, dt)
        else:  # split-step-fft (default)
            psi = step_split_step_fft_2d(psi, V, KX, KY, dt)
        
        if n % store_every == 0 or n == len(t) - 1:
            saved_t.append(float(t[n]))
            saved_psi.append(psi.copy())

    return saved_t, saved_psi

