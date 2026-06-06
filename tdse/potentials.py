"""
TDSE Potentials Module
======================

Potential classes and wave function utilities for TDSE simulations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict

import numpy as np

Array = np.ndarray


# =============================================================================
# Utility functions
# =============================================================================

def normalize(psi: Array, dx: float) -> Array:
    """Normalize the wave function: integral |psi|^2 dx = 1."""
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero wave function.")
    return psi / norm


def probability_mass(psi: Array, dx: float) -> float:
    """Compute the total probability mass of the wave function."""
    return float(np.sum(np.abs(psi) ** 2) * dx)


def mass_2d(psi: Array, dx: float, dy: float) -> float:
    """Compute the total mass for 2D wave function."""
    return float(np.sum(np.abs(psi) ** 2) * dx * dy)


def l1_l2_linf_error(psi_num: Array, psi_ref: Array, dx: float) -> Tuple[float, float, float]:
    """Compute L1, L2, and Linf errors between numerical and reference solutions,
    after aligning global phase to eliminate physically irrelevant phase differences."""
    psi_aligned = align_global_phase(psi_num, psi_ref, dx)
    diff = psi_aligned - psi_ref
    l1 = float(np.sum(np.abs(diff)) * dx)
    l2 = float(np.sqrt(np.sum(np.abs(diff) ** 2) * dx))
    linf = float(np.max(np.abs(diff)))
    return l1, l2, linf


def align_global_phase(psi: Array, reference: Array, dx: float) -> Array:
    """Remove irrelevant global phase before comparing wave functions."""
    inner = np.sum(np.conj(reference) * psi) * dx
    if np.abs(inner) < 1e-14:
        return psi
    return psi * np.exp(-1j * np.angle(inner))


def print_section(title: str) -> None:
    """Print a section header with formatting."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def dt_warning(method: str, dx: float, dt: float, vmax: float = 0.0) -> str:
    """Heuristic stability/accuracy note for the dimensionless TDSE.

    Provides practical guidance on time-step selection for each numerical method.
    """
    r = dt / dx**2
    method_lower = method.lower()
    if method_lower == "ftcs":
        return (
            f"FTCS has amplification > 1 for Schrodinger dynamics; "
            f"r = dt/dx^2 = {r:.3g}. Use only as an instability demonstration."
        )
    if method_lower == "rk4":
        return (
            f"RK4 with centered Laplacian is conditionally stable. "
            f"Heuristic r = dt/dx^2 = {r:.3g}; keep r <= 0.25 for safety."
        )
    if method_lower in ("cn", "crank-nicolson", "backward-euler", "be"):
        return (
            f"{method} is unconditionally L2-stable for this linear problem, "
            f"but dt still controls phase accuracy."
        )
    if method_lower in ("split-step", "split-step-fft", "fft", "ssf"):
        return (
            f"Split-Step FFT is unitary for real V with periodic BCs; "
            f"accuracy depends on dt and spectral resolution. max(V) = {vmax:.3g}."
        )
    return "Unknown method."


def grid(xmin: float, xmax: float, n: int) -> Tuple[Array, float]:
    """Create a 1D grid from xmin to xmax with n points (endpoint=False)."""
    x = np.linspace(xmin, xmax, n, endpoint=False)
    dx = float(x[1] - x[0])
    return x, dx


def make_2d_grid(
    nx: int, ny: int,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
) -> Tuple[Array, Array, Array, Array, float, float, Array, Array]:
    """Set up a 2D periodic grid and matching Fourier frequency arrays.

    Returns (X, Y, x, y, dx, dy, KX, KY) where all 2D arrays use ij-indexing
    so that psi[ix, iy] corresponds to the point (x[ix], y[iy]).

    Args:
        nx, ny: Number of grid points in x and y directions.
        xmin, xmax: Domain boundaries in x.
        ymin, ymax: Domain boundaries in y.

    Returns:
        X, Y: 2D meshgrid arrays (ij-indexing).
        x, y: 1D coordinate arrays.
        dx, dy: Grid spacings.
        KX, KY: 2D Fourier wavenumber arrays (ij-indexing).
    """
    x = np.linspace(xmin, xmax, nx, endpoint=False)
    y = np.linspace(ymin, ymax, ny, endpoint=False)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    X, Y = np.meshgrid(x, y, indexing="ij")
    KX, KY = np.meshgrid(kx_1d, ky_1d, indexing="ij")
    return X, Y, x, y, dx, dy, KX, KY


# =============================================================================
# Potential Base Classes
# =============================================================================

class Potential1D(ABC):
    """
    Abstract base class for 1D potentials.

    This class defines the interface for all 1D potential functions.
    Subclasses must implement the __call__ method.
    """

    @abstractmethod
    def __call__(self, x: Array) -> Array:
        """Evaluate potential at positions x."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for the potential."""
        pass

    @property
    def vmax(self) -> float:
        """Return maximum value of potential (for stability analysis)."""
        return 0.0

    def info(self) -> str:
        """Return information about this potential."""
        return f"{self.name}: vmax={self.vmax}"


class Potential2D(ABC):
    """
    Abstract base class for 2D potentials.

    This class defines the interface for all 2D potential functions.
    Subclasses must implement the __call__ method.
    """

    @abstractmethod
    def __call__(self, X: Array, Y: Array) -> Array:
        """Evaluate potential at grid positions X, Y."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for the potential."""
        pass

    @property
    def vmax(self) -> float:
        """Return maximum value of potential."""
        return 0.0


# =============================================================================
# 1D Potential Classes
# =============================================================================

class FreePotential(Potential1D):
    """Free particle potential (V = 0)."""

    @property
    def name(self) -> str:
        return "Free (V=0)"

    def __call__(self, x: Array) -> Array:
        return potential_free(x)


class HarmonicPotential(Potential1D):
    """Harmonic oscillator potential V(x) = (1/2) * omega^2 * x^2."""

    def __init__(self, omega: float = 1.0):
        self.omega = omega

    @property
    def name(self) -> str:
        return f"Harmonic (omega={self.omega})"

    @property
    def vmax(self) -> float:
        return float('inf')

    def __call__(self, x: Array) -> Array:
        return potential_harmonic(x, self.omega)


class RectangularBarrier(Potential1D):
    """Rectangular barrier potential V(x) = V0 for x in [a, b], 0 otherwise."""

    def __init__(self, v0: float = 1.0, a: float = -0.5, b: float = 0.5):
        self.v0 = v0
        self.a = a
        self.b = b

    @property
    def name(self) -> str:
        return f"Rectangular Barrier (V0={self.v0})"

    @property
    def vmax(self) -> float:
        return self.v0

    def __call__(self, x: Array) -> Array:
        return potential_rect_barrier(x, self.v0, self.a, self.b)


class DoubleBarrier1D(Potential1D):
    """Resonant tunneling double-barrier potential (RTD structure).

    V(x) = V0 for x in [a1, b1] ∪ [a2, b2], 0 otherwise.
    Two identical barriers with a quantum well between them,
    forming a resonant tunneling diode (RTD) structure.
    """

    def __init__(self, v0: float = 1.0,
                 a1: float = -3.0, b1: float = -1.5,
                 a2: float = 1.5, b2: float = 3.0):
        self.v0 = v0
        self.a1 = a1
        self.b1 = b1
        self.a2 = a2
        self.b2 = b2

    @property
    def name(self) -> str:
        return f"Double Barrier (V0={self.v0})"

    @property
    def vmax(self) -> float:
        return self.v0

    def __call__(self, x: Array) -> Array:
        return potential_double_barrier(x, self.v0,
                                        self.a1, self.b1, self.a2, self.b2)


# =============================================================================
# 2D Potential Classes
# =============================================================================

class CircularObstacle(Potential2D):
    """Circular obstacle potential in 2D."""

    def __init__(self, cx: float = 2.0, cy: float = 0.0, radius: float = 1.5, height: float = 50.0):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.height = height

    @property
    def name(self) -> str:
        return f"Circular Obstacle (R={self.radius})"

    @property
    def vmax(self) -> float:
        return self.height

    def __call__(self, X: Array, Y: Array) -> Array:
        return potential_circle_2d(X, Y, self.cx, self.cy, self.radius, self.height)


class Waveguide(Potential2D):
    """2D waveguide potential V(x,y) = alpha * y^2."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    @property
    def name(self) -> str:
        return f"Waveguide (alpha={self.alpha})"

    @property
    def vmax(self) -> float:
        return float('inf')

    def __call__(self, X: Array, Y: Array) -> Array:
        return potential_waveguide_2d(X, Y, self.alpha)


# =============================================================================
# Potential Factory
# =============================================================================

class TDSEPotentialFactory:
    """
    Factory class for creating potential objects.

    This factory provides a unified interface for creating various
    potential types for both 1D and 2D simulations.

    Example:
        >>> factory = TDSEPotentialFactory()
        >>> pot1d = factory.create("barrier", v0=1.0, a=-0.5, b=0.5)
        >>> pot2d = factory.create("circle", cx=2.0, cy=0.0, radius=1.5)
    """

    _1d_potentials = {
        "free": FreePotential,
        "harmonic": HarmonicPotential,
        "barrier": RectangularBarrier,
        "double_barrier": DoubleBarrier1D,
    }

    _2d_potentials = {
        "circle": CircularObstacle,
        "waveguide": Waveguide,
    }

    @classmethod
    def create(cls, potential_type: str, **kwargs) -> Potential1D | Potential2D:
        """
        Create a potential object.

        Args:
            potential_type: Type of potential ("free", "harmonic", "barrier", "circle", "waveguide")
            **kwargs: Parameters passed to the potential constructor

        Returns:
            Potential object (1D or 2D based on type)

        Raises:
            ValueError: If potential_type is not recognized
        """
        if potential_type in cls._1d_potentials:
            return cls._1d_potentials[potential_type](**kwargs)
        elif potential_type in cls._2d_potentials:
            return cls._2d_potentials[potential_type](**kwargs)
        else:
            raise ValueError(f"Unknown potential type: {potential_type}. "
                           f"Available: {list(cls._1d_potentials.keys()) + list(cls._2d_potentials.keys())}")

    @classmethod
    def list_potentials(cls, dim: Optional[int] = None) -> List[str]:
        """
        List available potential types.

        Args:
            dim: If 1 or 2, return only potentials of that dimension.
                 If None, return all potentials.
        """
        if dim == 1:
            return list(cls._1d_potentials.keys())
        elif dim == 2:
            return list(cls._2d_potentials.keys())
        else:
            return list(cls._1d_potentials.keys()) + list(cls._2d_potentials.keys())


# =============================================================================
# Potential functions
# =============================================================================

def potential_free(x: Array) -> Array:
    """Free particle potential (V = 0)."""
    return np.zeros_like(x, dtype=float)


def potential_harmonic(x: Array, omega: float = 1.0) -> Array:
    """Harmonic oscillator potential V(x) = (1/2) * omega^2 * x^2."""
    return 0.5 * omega**2 * x**2


def potential_rect_barrier(x: Array, v0: float = 1.0, a: float = -0.5, b: float = 0.5) -> Array:
    """Rectangular potential barrier."""
    return np.where((x > a) & (x < b), v0, 0.0).astype(float)


def potential_double_barrier(x: Array, v0: float = 1.0,
                             a1: float = -3.0, b1: float = -1.5,
                             a2: float = 1.5, b2: float = 3.0) -> Array:
    """RTD double-barrier potential: two identical barriers with a well between."""
    barrier1 = np.where((x > a1) & (x < b1), v0, 0.0)
    barrier2 = np.where((x > a2) & (x < b2), v0, 0.0)
    return (barrier1 + barrier2).astype(float)


def potential_infinite_well_mask(x: Array, a: float = -5.0, b: float = 5.0) -> Array:
    """Finite computational representation of an infinite well domain mask."""
    return ((x > a) & (x < b)).astype(float)


def potential_barrier_2d(
    X: Array, Y: Array,
    x_left: float = -0.5, x_right: float = 0.5, v0: float = 5.0,
) -> Array:
    """Rectangular potential barrier extending across the full y-range."""
    return np.where((X > x_left) & (X < x_right), v0, 0.0).astype(float)


def potential_circle_2d(
    X: Array, Y: Array,
    xc: float = 0.0, yc: float = 0.0, R: float = 1.0, v0: float = 5.0,
) -> Array:
    """Circular obstacle: V = v0 inside the circle of radius R centered at (xc, yc)."""
    r = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
    return np.where(r < R, v0, 0.0).astype(float)


def potential_waveguide_2d(X: Array, Y: Array, alpha: float = 0.5) -> Array:
    """Parabolic waveguide: V(x,y) = alpha * y^2, confines the wave near y=0."""
    return (alpha * Y ** 2).astype(float)


def absorbing_potential_2d(X: Array, Y: Array, width: float = 3.0, strength: float = 2.0) -> Array:
    """
    Complex absorbing potential for 2D boundaries.
    V = -i * W, where W increases quadratically towards the domain edges.
    """
    xmin, xmax = X[0, 0], X[-1, 0]
    ymin, ymax = Y[0, 0], Y[0, -1]

    # Distance to x boundaries
    dx_left = xmin + width - X
    dx_right = X - (xmax - width)
    dx = np.maximum(dx_left, dx_right)
    dx = np.maximum(dx, 0.0)

    # Distance to y boundaries
    dy_bottom = ymin + width - Y
    dy_top = Y - (ymax - width)
    dy = np.maximum(dy_bottom, dy_top)
    dy = np.maximum(dy, 0.0)

    # Quadratic profile
    W = strength * ((dx / width) ** 2 + (dy / width) ** 2)
    return -1j * W.astype(complex)


# =============================================================================
# Wave functions and initial states
# =============================================================================

def gaussian_wavepacket(x: Array, x0: float, sigma: float, k0: float, dx: float) -> Array:
    """Create a normalized 1D Gaussian wave packet."""
    psi = np.exp(-((x - x0) ** 2) / (2.0 * sigma**2)) * np.exp(1j * k0 * x)
    return normalize(psi.astype(complex), dx)


def gaussian_wavepacket_2d(
    X: Array, Y: Array,
    x0: float, y0: float, sigma: float,
    kx0: float, ky0: float,
    dx: float, dy: float,
) -> Array:
    """
    Normalized 2D Gaussian wave packet:
        psi(x,y) = exp(-((x-x0)^2+(y-y0)^2)/(2*sigma^2)) * exp(i*(kx0*x + ky0*y))
    """
    envelope = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma ** 2))
    phase = np.exp(1j * (kx0 * X + ky0 * Y))
    psi = (envelope * phase).astype(complex)
    norm = np.sqrt(mass_2d(psi, dx, dy))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero 2D wave function.")
    return psi / norm


def exact_free_gaussian_2d(
    X: Array, Y: Array, t: float,
    x0: float, y0: float, sigma: float,
    kx0: float, ky0: float,
    dx: float, dy: float,
) -> Array:
    """Exact 2D free-particle Gaussian wave packet."""
    denom = sigma ** 2 + 1j * t
    # For 2D, each dimension has prefactor sigma / sqrt(denom)
    prefactor = (sigma / np.sqrt(denom)) ** 2
    envelope = np.exp(-((X - x0 - kx0 * t) ** 2 + (Y - y0 - ky0 * t) ** 2) / (2.0 * denom))
    phase = np.exp(1j * (kx0 * X + ky0 * Y - 0.5 * (kx0 ** 2 + ky0 ** 2) * t))
    psi = (prefactor * envelope * phase).astype(complex)
    # Normalize numerically
    norm = np.sqrt(mass_2d(psi, dx, dy))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero 2D wave function.")
    return psi / norm


def exact_free_gaussian(x: Array, t: float, x0: float, sigma: float, k0: float, dx: float) -> Array:
    """
    Exact free-particle Gaussian propagation for initial
        exp(-(x-x0)^2/(2 sigma^2)) exp(i k0 x)
    """
    denom = sigma**2 + 1j * t
    prefactor = sigma / np.sqrt(denom)
    phase = np.exp(1j * k0 * x - 0.5j * k0**2 * t)
    envelope = np.exp(-((x - x0 - k0 * t) ** 2) / (2.0 * denom))
    return normalize((prefactor * envelope * phase).astype(complex), dx)


def infinite_well_eigenstate(x: Array, n: int, a: float = -5.0, b: float = 5.0) -> Tuple[Array, float]:
    """Analytic eigenstate and energy for an infinite well on [a,b]."""
    length = b - a
    psi = np.zeros_like(x, dtype=complex)
    inside = (x >= a) & (x <= b)
    psi[inside] = np.sqrt(2.0 / length) * np.sin(n * np.pi * (x[inside] - a) / length)
    energy = 0.5 * (n * np.pi / length) ** 2
    return psi, energy


def harmonic_eigenstate_low(x: Array, n: int = 0, omega: float = 1.0, dx: float = 1.0) -> Tuple[Array, float]:
    """
    Low-lying analytic harmonic oscillator states for n=0,1,2.
    H = -1/2 dxx + 1/2 omega^2 x^2, E_n = omega(n+1/2).
    """
    xi = np.sqrt(omega) * x
    gaussian = (omega / np.pi) ** 0.25 * np.exp(-0.5 * xi**2)
    if n == 0:
        psi = gaussian
    elif n == 1:
        psi = gaussian * np.sqrt(2) * xi
    elif n == 2:
        psi = gaussian * (1 / np.sqrt(2)) * (2 * xi**2 - 1)
    else:
        raise ValueError("Only n=0,1,2 are available in this function.")
    return normalize(psi.astype(complex), dx), omega * (n + 0.5)


def compute_transmission_reflection(
    psi: Array,
    x: Array,
    dx: float,
    barrier_left: float,
    barrier_right: float,
    buffer: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Compute transmission and reflection coefficients automatically.

    This function calculates the probability of finding the particle
    transmitted through the barrier (right side) and reflected (left side).

    Args:
        psi: Wave function array
        x: Spatial grid
        dx: Grid spacing
        barrier_left: Left boundary of the barrier
        barrier_right: Right boundary of the barrier
        buffer: Buffer zone near barrier to exclude (default: 1.0)

    Returns:
        Tuple of (transmission, reflection, total)
        - transmission: Probability of being transmitted (right of barrier)
        - reflection: Probability of being reflected (left of barrier)
        - total: Sum (should be close to 1.0 for conserved systems)

    Example:
        >>> T, R, total = compute_transmission_reflection(
        ...     psi, x, dx,
        ...     barrier_left=-0.5, barrier_right=0.5,
        ...     buffer=0.5
        ... )
    """
    x_left = x < (barrier_left - buffer)
    x_right = x > (barrier_right + buffer)

    reflection = float(np.sum(np.abs(psi[x_left]) ** 2) * dx)
    transmission = float(np.sum(np.abs(psi[x_right]) ** 2) * dx)
    total = reflection + transmission

    return transmission, reflection, total


def analyze_barrier_scattering(
    psi: Array,
    x: Array,
    dx: float,
    barrier_center: float = 0.0,
    barrier_width: float = 1.0,
    buffer: float = 1.0,
) -> Dict[str, float]:
    """
    Comprehensive barrier scattering analysis.

    Computes transmission, reflection, and various interference measures
    for a wave packet scattering off a potential barrier.

    Args:
        psi: Final wave function after scattering
        x: Spatial grid
        dx: Grid spacing
        barrier_center: Center position of the barrier
        barrier_width: Width of the barrier region
        buffer: Buffer zone near barrier

    Returns:
        Dictionary containing:
        - transmission: T coefficient
        - reflection: R coefficient
        - total: T + R (should be ~1.0)
        - unaccounted: 1 - (T + R) (lost in barrier region)
        - barrier_amplitude: Average |psi|^2 in barrier region
    """
    barrier_left = barrier_center - barrier_width / 2
    barrier_right = barrier_center + barrier_width / 2

    x_left = x < (barrier_left - buffer)
    x_right = x > (barrier_right + buffer)
    x_barrier = (x >= barrier_left) & (x <= barrier_right)

    reflection = float(np.sum(np.abs(psi[x_left]) ** 2) * dx)
    transmission = float(np.sum(np.abs(psi[x_right]) ** 2) * dx)
    total = reflection + transmission
    unaccounted = 1.0 - total
    barrier_amplitude = float(np.mean(np.abs(psi[x_barrier]) ** 2))

    return {
        "transmission": transmission,
        "reflection": reflection,
        "total": total,
        "unaccounted": unaccounted,
        "barrier_amplitude": barrier_amplitude,
    }


def transmission_reflection(psi: Array, x: Array, dx: float, barrier_center: float = 0.0, buffer: float = 0.0) -> Tuple[float, float]:
    """
    Compute transmission and reflection coefficients by integrating |psi|^2
    on the left and right sides of a barrier centered approximately at
    barrier_center (often 0). Optional buffer (e.g., 0.5) can exclude
    regions near the barrier where interference is present.
    """
    x_left = x < (barrier_center - buffer)
    x_right = x > (barrier_center + buffer)
    reflection = float(np.sum(np.abs(psi[x_left]) ** 2) * dx)
    transmission = float(np.sum(np.abs(psi[x_right]) ** 2) * dx)
    return transmission, reflection

