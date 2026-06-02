"""
Time-Dependent Schrodinger Equation (1D + 2D) numerical PDE demo.

Dimensionless model:
    1D:  i psi_t = -1/2 psi_xx + V(x) psi
    2D:  i psi_t = -1/2 (psi_xx + psi_yy) + V(x,y) psi

Implemented 1D methods:
    FTCS, Backward Euler, Crank-Nicolson, Split-Step Fourier (SSFM), RK4.

Implemented 2D method:
    Split-Step FFT (step_split_step_fft_2d, uses numpy.fft.fft2 / ifft2).

1D outputs (./tdse_outputs):
    Figure 1: analytic vs numerical
    Figure 2: error convergence
    Figure 3: stability map
    Figure 4: performance table
    Figure 5: wavepacket animation
    Figure 6: tunneling simulation

2D outputs (./tdse_outputs):
    Figure 7: 2D free Gaussian propagation + mass conservation
    Figure 8: 2D circular obstacle scattering
    Figure 9: 2D waveguide vs free propagation

Dependencies:
    numpy scipy matplotlib pandas tqdm

Module Structure:
    - TDSEConfig: Unified configuration for all TDSE simulations
    - Potential1D/Potential2D: Base classes for 1D/2D potentials
    - Solver1D/Solver2D: Base classes for 1D/2D solvers
    - TDSESolver: Factory class for creating solvers
    - ExperimentRunner: Unified interface for running experiments
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.linalg import solve_banded
from scipy.sparse import diags
from tqdm import tqdm


Array = np.ndarray


# =============================================================================
# Unified Configuration System
# =============================================================================


@dataclass
class TDSEConfig:
    """
    Unified configuration for all TDSE simulations.
    
    This is the main configuration class that controls all aspects
    of TDSE simulations including grid parameters, solver settings,
    output options, and visualization quality.
    
    Attributes:
        outdir: Output directory for generated files
        quick: If True, use reduced grid sizes for faster execution
        save_gif: Whether to save GIF animations
        dpi: Resolution for saved figures (300 for publication quality)
        dim: Spatial dimension (1 or 2)
        grid_size: Number of grid points
        xmin, xmax: Spatial domain boundaries (x-direction)
        ymin, ymax: Spatial domain boundaries (y-direction, 2D only)
        dt: Time step size
        t_end: Final simulation time
        store_every: Save state every N steps
    """
    
    outdir: str = "tdse_outputs2"
    quick: bool = False
    save_gif: bool = True
    dpi: int = 300
    dim: int = 1
    grid_size: int = 384
    xmin: float = -30.0
    xmax: float = 30.0
    ymin: float = -10.0
    ymax: float = 10.0
    dt: float = 0.004
    t_end: float = 3.0
    store_every: int = 10
    
    def __post_init__(self):
        """Adjust parameters based on quick mode."""
        if self.quick:
            self.grid_size = min(self.grid_size, 384)
            self.t_end = min(self.t_end, 10.0)
            self.store_every = max(1, self.store_every // 2)
    
    def to_run_config(self) -> 'RunConfig':
        """Convert to legacy RunConfig for compatibility."""
        return RunConfig(
            outdir=self.outdir,
            quick=self.quick,
            save_gif=self.save_gif,
            dpi=self.dpi
        )


@dataclass 
class RunConfig:
    """Legacy configuration class for backward compatibility."""
    outdir: str = "tdse_outputs2"
    quick: bool = False
    save_gif: bool = True
    dpi: int = 300


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def setup_plot_style(dpi: int = 300) -> None:
    """
    Configure global matplotlib style for high-quality publication-ready output.
    
    Sets up consistent fonts, colors, grid lines, and figure quality parameters
    across all plots in the project.
    """
    # Use a clean base style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Font configuration - use matplotlib's built-in DejaVu Sans only
    plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display
    
    # Font sizes - optimized for readability without overlap
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Line quality improvements
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['lines.markeredgewidth'] = 1.5
    
    # High DPI output settings
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.2  # Increased padding to prevent cutoff
    plt.rcParams['savefig.format'] = 'png'
    
    # Grid styling - subtle but visible
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.color'] = '#CCCCCC'
    
    # Professional color palette with good distinguishability
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', [
            '#2E86AB',  # Blue
            '#A23B72',  # Magenta
            '#F18F01',  # Orange
            '#C73E1D',  # Red
            '#3B1F2B',  # Dark purple
            '#95C623',  # Lime green
            '#6B4C9A',  # Purple
            '#1B998B',  # Teal
        ]
    )
    
    # Better default colormap for heatmaps
    plt.rcParams['image.cmap'] = 'inferno'
    plt.rcParams['image.interpolation'] = 'bilinear'
    
    # Spine and tick improvements
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['xtick.direction'] = 'out'
    
    # Layout adjustments - prevent label cutoff and text overlap
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['figure.constrained_layout.h_pad'] = 0.5
    plt.rcParams['figure.constrained_layout.w_pad'] = 0.5
    plt.rcParams['figure.constrained_layout.hspace'] = 0.2
    plt.rcParams['figure.constrained_layout.wspace'] = 0.2
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    
    # Legend improvements
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = '#CCCCCC'


def normalize(psi: Array, dx: float) -> Array:
    """Normalize a wave function in L2."""

    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm == 0:
        raise ValueError("Cannot normalize a zero wave function.")
    return psi / norm


def probability_mass(psi: Array, dx: float) -> float:
    return float(np.sum(np.abs(psi) ** 2) * dx)


def mass_2d(psi: Array, dx: float, dy: float) -> float:
    """Probability mass integral for a 2D wave function on a uniform (dx, dy) grid."""
    return float(np.sum(np.abs(psi) ** 2) * dx * dy)


def l1_l2_linf_error(psi_num: Array, psi_ref: Array, dx: float) -> Tuple[float, float, float]:
    """Errors for complex wave functions, after global phase alignment."""

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
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def grid(xmin: float, xmax: float, n: int) -> Tuple[Array, float]:
    x = np.linspace(xmin, xmax, n, endpoint=False)
    dx = float(x[1] - x[0])
    return x, dx


def dt_warning(method: str, dx: float, dt: float, vmax: float = 0.0) -> str:
    """Heuristic stability/accuracy note for dimensionless TDSE."""

    r = dt / dx**2
    method = method.lower()
    if method == "ftcs":
        return (
            f"FTCS has amplification > 1 for Schrodinger dynamics; r=dt/dx^2={r:.3g}. "
            "Use only as an instability demonstration."
        )
    if method == "rk4":
        return (
            f"RK4 with centered Laplacian is conditionally stable. "
            f"Heuristic r=dt/dx^2={r:.3g}; keep it small, e.g. r <= 0.25."
        )
    if method in ("cn", "crank-nicolson", "backward-euler"):
        return (
            f"{method} is unconditionally stable in norm/boundedness sense for this linear problem, "
            "but dt still controls phase accuracy."
        )
    if method in ("split-step", "split-step-fft", "fft"):
        return (
            "Split-Step FFT is unitary for real V with periodic boundary assumptions; "
            f"accuracy depends on dt and spectral resolution. max(V)={vmax:.3g}."
        )
    return "Unknown method."


# =============================================================================
# Potential Module: Base Classes and Factory
# =============================================================================


class Potential1D(ABC):
    """
    Abstract base class for 1D potentials.
    
    This class defines the interface for all 1D potential functions.
    Subclasses must implement the __call__ method.
    
    Example:
        >>> potential = RectangularBarrier(v0=1.0, a=-0.5, b=0.5)
        >>> V = potential(x)  # Returns potential array
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


class FreePotential(Potential1D):
    """Free particle potential (V = 0)."""
    
    @property
    def name(self) -> str:
        return "Free (V=0)"
    
    def __call__(self, x: Array) -> Array:
        return np.zeros_like(x, dtype=float)


class HarmonicPotential(Potential1D):
    """Harmonic oscillator potential V(x) = (1/2) * omega^2 * x^2."""
    
    def __init__(self, omega: float = 1.0):
        self.omega = omega
    
    @property
    def name(self) -> str:
        return f"Harmonic (ω={self.omega})"
    
    @property
    def vmax(self) -> float:
        return float('inf')
    
    def __call__(self, x: Array) -> Array:
        return 0.5 * self.omega**2 * x**2


class RectangularBarrier(Potential1D):
    """Rectangular barrier potential V(x) = V0 for x in [a, b], 0 otherwise."""
    
    def __init__(self, v0: float = 1.0, a: float = -0.5, b: float = 0.5):
        self.v0 = v0
        self.a = a
        self.b = b
    
    @property
    def name(self) -> str:
        return f"Rectangular Barrier (V₀={self.v0})"
    
    @property
    def vmax(self) -> float:
        return self.v0
    
    def __call__(self, x: Array) -> Array:
        return potential_rect_barrier(x, self.v0, self.a, self.b)


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
        return f"Waveguide (α={self.alpha})"
    
    @property
    def vmax(self) -> float:
        return float('inf')
    
    def __call__(self, X: Array, Y: Array) -> Array:
        return potential_waveguide_2d(X, Y, self.alpha)


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
# Physical models and analytic references
# =============================================================================


def potential_free(x: Array) -> Array:
    return np.zeros_like(x, dtype=float)


def potential_harmonic(x: Array, omega: float = 1.0) -> Array:
    return 0.5 * omega**2 * x**2


def potential_rect_barrier(x: Array, v0: float = 1.0, a: float = -0.5, b: float = 0.5) -> Array:
    return np.where((x > a) & (x < b), v0, 0.0).astype(float)


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


def gaussian_wavepacket(x: Array, x0: float, sigma: float, k0: float, dx: float) -> Array:
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
    """
    Exact 2D free-particle Gaussian wave packet.
    """
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
    in dimensionless units. The expression is normalized numerically to match
    the discrete grid used by the demo.
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
        psi = np.sqrt(2.0) * xi * gaussian
    elif n == 2:
        psi = (1.0 / np.sqrt(2.0)) * (2.0 * xi**2 - 1.0) * gaussian
    else:
        raise ValueError("This compact demo implements harmonic oscillator states n=0,1,2.")
    energy = omega * (n + 0.5)
    return normalize(psi.astype(complex), dx), energy


# =============================================================================
# Solver Module: Base Classes and Factory
# =============================================================================


class Solver1D(ABC):
    """
    Abstract base class for 1D TDSE solvers.
    
    This class defines the interface for all 1D time-stepping methods.
    Subclasses must implement the step method.
    
    Example:
        >>> solver = TDSESolver.create("CN")
        >>> psi_new = solver.step(psi_old, V, x, dx, dt)
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


# =============================================================================
# Experiments
# =============================================================================


def experiment_analytic_vs_numerical(cfg: RunConfig) -> pd.DataFrame:
    """Figure 1 and a compact analytic comparison table."""

    print_section("Figure 1: analytic vs numerical")
    n = 384 if cfg.quick else 512
    x, dx = grid(-30.0, 30.0, n)
    t_end = 2.0
    dt = 0.0025 if cfg.quick else 0.002
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    x0, sigma, k0 = -8.0, 1.2, 2.0
    psi0 = gaussian_wavepacket(x, x0, sigma, k0, dx)
    v = potential_free(x)
    methods = ["FTCS", "Backward-Euler", "Crank-Nicolson", "Split-Step-FFT", "RK4"]
    rows = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    psi_exact = exact_free_gaussian(x, t_end, x0, sigma, k0, dx)
    axes[0, 0].plot(x, np.abs(psi_exact) ** 2, "k--", lw=2.5, label="Exact", alpha=0.8)

    final_solutions = {}
    for method in tqdm(methods, desc="analytic comparison"):
        start = time.perf_counter()
        _, psi_hist = solve(method, psi0, v, x, t, dx, dt, store_every=len(t) - 1)
        runtime = time.perf_counter() - start
        psi_num = psi_hist[-1]
        final_solutions[method] = psi_num
        l1, l2, linf = l1_l2_linf_error(psi_num, psi_exact, dx)
        mass = probability_mass(psi_num, dx)
        rows.append(
            {
                "method": method,
                "N": n,
                "dx": dx,
                "dt": dt,
                "runtime_s": runtime,
                "L1": l1,
                "L2": l2,
                "Linf": linf,
                "mass": mass,
            }
        )
        if method != "FTCS":
            axes[0, 0].plot(x, np.abs(psi_num) ** 2, lw=1.8, label=method)

    axes[0, 0].set_title("Free Gaussian: |ψ|² at Final Time", fontweight='bold', pad=10)
    axes[0, 0].set_xlabel("Position x", labelpad=8)
    axes[0, 0].set_ylabel("Probability Density |ψ|²", labelpad=8)
    axes[0, 0].legend(fontsize=9, loc='upper right', framealpha=0.95)
    axes[0, 0].set_xlim(-15, 5)
    axes[0, 0].grid(True, alpha=0.3)

    l2_values = [r["L2"] for r in rows]
    bars = axes[0, 1].bar([r["method"] for r in rows], l2_values, color=['#C73E1D', '#A23B72', '#2E86AB', '#F18F01', '#3B1F2B'], alpha=0.8, edgecolor='white', linewidth=0.5)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("L₂ Error vs Exact Gaussian", fontweight='bold', pad=10)
    axes[0, 1].set_xlabel("Numerical Method", labelpad=8)
    axes[0, 1].set_ylabel("L₂ Error (log scale)", labelpad=8)
    axes[0, 1].tick_params(axis="x", rotation=35)

    # Infinite well analytic eigenstate: CN should preserve phase and density.
    xw, dxw = grid(-5.0, 5.0, n)
    psi_well0, e_well = infinite_well_eigenstate(xw, n=2, a=-5.0, b=5.0)
    tw_end, dtw = 1.5, 0.0025
    tw = np.arange(0.0, tw_end + 0.5 * dtw, dtw)
    vw = np.zeros_like(xw)
    _, psi_well_hist = solve("Crank-Nicolson", psi_well0, vw, xw, tw, dxw, dtw, store_every=len(tw) - 1)
    psi_well_exact = psi_well0 * np.exp(-1j * e_well * tw_end)
    axes[1, 0].plot(xw, np.abs(psi_well_exact) ** 2, "k--", lw=2.5, label="Exact n=2", alpha=0.8)
    axes[1, 0].plot(xw, np.abs(psi_well_hist[-1]) ** 2, lw=1.8, label="Crank-Nicolson")
    axes[1, 0].set_title("Infinite Well Eigenstate (n=2)", fontweight='bold', pad=10)
    axes[1, 0].set_xlabel("Position x", labelpad=8)
    axes[1, 0].set_ylabel("Probability Density |ψ|²", labelpad=8)
    axes[1, 0].legend(fontsize=9, loc='upper right', framealpha=0.95)

    # Harmonic oscillator low state: compare against analytic phase evolution.
    xh, dxh = grid(-10.0, 10.0, n)
    omega = 1.0
    psi_h0, e_h = harmonic_eigenstate_low(xh, n=1, omega=omega, dx=dxh)
    vh = potential_harmonic(xh, omega)
    th_end, dth = 1.0, 0.002
    th = np.arange(0.0, th_end + 0.5 * dth, dth)
    _, psi_h_hist = solve("Crank-Nicolson", psi_h0, vh, xh, th, dxh, dth, store_every=len(th) - 1)
    psi_h_exact = psi_h0 * np.exp(-1j * e_h * th_end)
    axes[1, 1].plot(xh, np.abs(psi_h_exact) ** 2, "k--", lw=2.5, label="Exact n=1", alpha=0.8)
    axes[1, 1].plot(xh, np.abs(psi_h_hist[-1]) ** 2, lw=1.8, label="Crank-Nicolson")
    axes[1, 1].set_title("Harmonic Oscillator (n=1)", fontweight='bold', pad=10)
    axes[1, 1].set_xlabel("Position x", labelpad=8)
    axes[1, 1].set_ylabel("Probability Density |ψ|²", labelpad=8)
    axes[1, 1].legend(fontsize=9, loc='upper right', framealpha=0.95)

    fig.suptitle("Figure 1: Analytical vs Numerical Solutions", fontsize=16, fontweight='bold', y=1.02)

    fig.savefig(os.path.join(cfg.outdir, "figure1_analytic_vs_numerical.png"), dpi=cfg.dpi)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "analytic_error_table.csv"), index=False)
    print(df.to_string(index=False))
    return df


def experiment_convergence(cfg: RunConfig) -> pd.DataFrame:
    """Figure 2: error vs dx and fitted log-log convergence order."""

    print_section("Figure 2: convergence study")
    ns = [50, 100, 200, 400]
    methods = ["Crank-Nicolson", "Split-Step-FFT", "RK4"]
    rows = []
    xmin, xmax = -30.0, 30.0
    t_end = 0.8
    x0, sigma, k0 = -8.0, 1.0, 1.5

    for n in tqdm(ns, desc="convergence grids"):
        x, dx = grid(xmin, xmax, n)
        dt = min(0.0015, 0.04 * dx**2)
        if cfg.quick:
            dt = min(0.003, 0.06 * dx**2)
        t = np.arange(0.0, t_end + 0.5 * dt, dt)
        psi0 = gaussian_wavepacket(x, x0, sigma, k0, dx)
        v = potential_free(x)
        psi_ref = exact_free_gaussian(x, t[-1], x0, sigma, k0, dx)
        for method in methods:
            _, hist = solve(method, psi0, v, x, t, dx, dt, store_every=len(t) - 1)
            l1, l2, linf = l1_l2_linf_error(hist[-1], psi_ref, dx)
            rows.append({"method": method, "N": n, "dx": dx, "dt": dt, "L1": l1, "L2": l2, "Linf": linf})

    df = pd.DataFrame(rows)
    orders = []
    for method in methods:
        sub = df[df["method"] == method].sort_values("dx")
        coeff = np.polyfit(np.log(sub["dx"]), np.log(sub["L2"]), 1)
        order = float(coeff[0])
        orders.append({"method": method, "fitted_L2_order": order})
    order_df = pd.DataFrame(orders)
    df.to_csv(os.path.join(cfg.outdir, "convergence_errors.csv"), index=False)
    order_df.to_csv(os.path.join(cfg.outdir, "convergence_orders.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for method in methods:
        sub = df[df["method"] == method].sort_values("dx")
        label = f"{method}, p={order_df[order_df.method == method].fitted_L2_order.iloc[0]:.2f}"
        ax.loglog(sub["dx"], sub["L2"], "o-", label=label, markersize=8, linewidth=2)
    ax.invert_xaxis()
    ax.set_xlabel("Grid Spacing Δx", labelpad=10)
    ax.set_ylabel("L₂ Error", labelpad=10)
    ax.set_title("Error Convergence vs Exact Free Gaussian", fontweight='bold', pad=15)
    ax.grid(True, which="both", alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
    fig.savefig(os.path.join(cfg.outdir, "figure2_error_convergence.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(df.to_string(index=False))
    print("\nFitted log-log convergence orders:")
    print(order_df.to_string(index=False))
    return df


def experiment_stability(cfg: RunConfig) -> pd.DataFrame:
    """Figure 3: stability map over dt and dx."""

    print_section("Figure 3: stability scan")
    methods = ["FTCS", "Backward-Euler", "Crank-Nicolson", "Split-Step-FFT", "RK4"]
    ns = [64, 96, 128, 192] if not cfg.quick else [64, 96, 128]
    dts = [0.001, 0.004, 0.010, 0.020, 0.040, 0.080] if not cfg.quick else [0.004, 0.020, 0.080]
    rows = []
    t_end = 1.5 if not cfg.quick else 1.0
    for n in tqdm(ns, desc="stability grids"):
        x, dx = grid(-20.0, 20.0, n)
        psi0 = gaussian_wavepacket(x, -5.0, 1.0, 2.0, dx)
        v = potential_free(x)
        for dt in dts:
            t = np.arange(0.0, t_end + 0.5 * dt, dt)
            for method in methods:
                try:
                    _, hist = solve(method, psi0, v, x, t, dx, dt, store_every=len(t) - 1)
                    mass = probability_mass(hist[-1], dx)
                    max_amp = float(np.max(np.abs(hist[-1])))
                    stable = np.isfinite(mass) and 0.2 < mass < 1.8 and max_amp < 8.0
                except Exception:
                    mass = np.nan
                    max_amp = np.nan
                    stable = False
                rows.append(
                    {
                        "method": method,
                        "N": n,
                        "dx": dx,
                        "dt": dt,
                        "r_dt_over_dx2": dt / dx**2,
                        "mass": mass,
                        "max_amp": max_amp,
                        "stable": stable,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "stability_scan.csv"), index=False)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    axes_flat = axes.flatten()
    summary = df.groupby("method")["stable"].agg(["sum", "count"])
    summary["verdict"] = np.where(summary["sum"] == summary["count"], "stable in scan", "conditional/divergent cases")
    
    for idx, (ax, method) in enumerate(zip(axes_flat, methods)):
        sub = df[df["method"] == method]
        pivot = sub.pivot(index="dt", columns="N", values="stable").astype(int)
        im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f"{method}", fontweight='bold', fontsize=12, pad=10)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:g}" for v in pivot.index], fontsize=10)
        ax.set_xlabel("Grid Points N", labelpad=8)
        ax.set_ylabel("Time Step Δt", labelpad=8)
    
    ax_summary = axes_flat[-1]
    stable_counts = summary["sum"].values
    x_pos = np.arange(len(methods))
    ax_summary.bar(x_pos, stable_counts, color=['#C73E1D' if s < 9 else '#2E86AB' for s in stable_counts], 
                   edgecolor='white', linewidth=1.5)
    ax_summary.set_xticks(x_pos)
    ax_summary.set_xticklabels(methods, rotation=35, ha='right', fontsize=10)
    ax_summary.set_ylabel("Stable Configurations", labelpad=8)
    ax_summary.set_title("Stability Summary", fontweight='bold', fontsize=12, pad=10)
    ax_summary.set_ylim(0, 10)
    ax_summary.grid(True, alpha=0.3, axis='y')
    
    cbar = fig.colorbar(im, ax=axes_flat[:-1], label="Stable (1) / Divergent (0)", shrink=0.75, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle("Figure 3: Stability Map Across Methods", fontsize=16, fontweight='bold', y=0.99)
    fig.savefig(os.path.join(cfg.outdir, "figure3_stability_map.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(summary.to_string())
    print("\nStability notes:")
    for method in methods:
        print(f"- {method}: {dt_warning(method, 0.2, 0.001)}")
    return df


def transmission_reflection(psi: Array, x: Array, dx: float, barrier_center: float = 0.0, buffer: float = 0.0) -> Tuple[float, float]:
    left = x < barrier_center - buffer
    right = x > barrier_center + buffer
    r = float(np.sum(np.abs(psi[left]) ** 2) * dx)
    t = float(np.sum(np.abs(psi[right]) ** 2) * dx)
    return t, r


def experiment_tunneling(cfg: RunConfig) -> pd.DataFrame:
    """Figure 6: rectangular barrier tunneling, with R+T approximately conserved."""

    print_section("Figure 6: tunneling simulation")
    n = 768 if cfg.quick else 1024
    x, dx = grid(-140.0, 140.0, n)
    v0 = 1.0
    a, b = -1.0, 1.0
    v = potential_rect_barrier(x, v0=v0, a=a, b=b)
    sigma, x0 = 4.0, -55.0
    dt = 0.02
    t_end = 70.0 if cfg.quick else 85.0
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    cases = [
        ("E<V0", np.sqrt(2.0 * 0.55 * v0)),
        ("E~V0", np.sqrt(2.0 * 1.00 * v0)),
        ("E>V0", np.sqrt(2.0 * 1.80 * v0)),
    ]
    rows = []

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes_flat = axes.flatten()
    animation_hist = None
    animation_t = None
    animation_label = None
    results = []
    
    for ax, (label, k0) in zip(axes_flat[:3], cases):
        psi0 = gaussian_wavepacket(x, x0, sigma, k0, dx)
        energy = 0.5 * k0**2
        store_every = max(1, len(t) // 220)
        saved_t, hist = solve("Split-Step-FFT", psi0, v, x, t, dx, dt, store_every=store_every)
        psi_final = hist[-1]
        tprob, rprob = transmission_reflection(psi_final, x, dx, barrier_center=0.0, buffer=0.0)
        rows.append({"case": label, "E": energy, "V0": v0, "T": tprob, "R": rprob, "R_plus_T": rprob + tprob})
        results.append({"label": label, "T": tprob, "R": rprob})
        
        ax.plot(x, np.abs(psi0) ** 2, color="gray", lw=1.2, label="Initial", alpha=0.7, linestyle='--')
        ax.plot(x, np.abs(psi_final) ** 2, lw=2.0, label="Final", color='#2E86AB')
        ax.fill_between(x, 0, v / max(v0, 1e-12) * 0.05, color="#C73E1D", alpha=0.3, label=f"Barrier V₀={v0}")
        ax.set_xlim(-95, 95)
        ax.set_ylabel("Probability Density |ψ|²", labelpad=8)
        ax.set_title(f"{label}: E={energy:.2f}, T={tprob:.3f}, R={rprob:.3f}", 
                    fontweight='bold', fontsize=11, pad=8)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        if label == "E<V0":
            animation_hist = hist
            animation_t = saved_t
            animation_label = label
    
    ax_summary = axes_flat[-1]
    labels = [r["label"] for r in results]
    T_vals = [r["T"] for r in results]
    R_vals = [r["R"] for r in results]
    x_pos = np.arange(len(labels))
    width = 0.35
    ax_summary.bar(x_pos - width/2, T_vals, width, label='Transmission', color='#2E86AB', edgecolor='white')
    ax_summary.bar(x_pos + width/2, R_vals, width, label='Reflection', color='#A23B72', edgecolor='white')
    ax_summary.set_xticks(x_pos)
    ax_summary.set_xticklabels(labels, fontsize=10)
    ax_summary.set_ylabel("Probability", labelpad=8)
    ax_summary.set_title("Transmission vs Reflection Comparison", fontweight='bold', fontsize=12, pad=10)
    ax_summary.legend(fontsize=9)
    ax_summary.set_ylim(0, 1.1)
    ax_summary.grid(True, alpha=0.3, axis='y')
    
    axes_flat[2].set_xlabel("Position x", labelpad=8)
    axes_flat[3].set_xlabel("Case", labelpad=8)
    fig.suptitle("Figure 6: Quantum Tunneling Through Rectangular Barrier", fontsize=15, fontweight='bold', y=0.98)
    fig.savefig(os.path.join(cfg.outdir, "figure6_tunneling_simulation.png"), dpi=cfg.dpi)
    plt.close(fig)

    if cfg.save_gif and animation_hist is not None:
        save_tunneling_animation(cfg, x, v, animation_t, animation_hist, animation_label)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "tunneling_RT.csv"), index=False)
    print(df.to_string(index=False))
    return df


def experiment_performance(cfg: RunConfig) -> pd.DataFrame:
    """Figure 4: runtime comparison table for FTCS, CN and Split-Step FFT."""

    print_section("Figure 4: performance comparison")
    ns = [256, 512, 1024] if not cfg.quick else [192, 384]
    methods = ["FTCS", "Crank-Nicolson", "Split-Step-FFT"]
    rows = []
    for n in tqdm(ns, desc="performance"):
        x, dx = grid(-30.0, 30.0, n)
        dt = 0.002
        t_end = 0.35
        t = np.arange(0.0, t_end + 0.5 * dt, dt)
        psi0 = gaussian_wavepacket(x, -8.0, 1.2, 2.0, dx)
        v = potential_harmonic(x, omega=0.08)
        for method in methods:
            start = time.perf_counter()
            _, hist = solve(method, psi0, v, x, t, dx, dt, store_every=len(t) - 1)
            runtime = time.perf_counter() - start
            rows.append(
                {
                    "method": method,
                    "grid_size": n,
                    "dx": dx,
                    "dt": dt,
                    "steps": len(t) - 1,
                    "runtime_s": runtime,
                    "mass_final": probability_mass(hist[-1], dx),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "performance_table.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    ax.axis("off")
    shown = df.copy()
    shown["runtime_s"] = shown["runtime_s"].map(lambda z: f"{z:.4f}")
    shown["dx"] = shown["dx"].map(lambda z: f"{z:.4g}")
    shown["mass_final"] = shown["mass_final"].map(lambda z: f"{z:.6f}")
    table = ax.table(cellText=shown.values, colLabels=shown.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    ax.set_title("Performance table")
    fig.savefig(os.path.join(cfg.outdir, "figure4_performance_table.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(df.to_string(index=False))
    return df


def experiment_method_comparison(cfg: RunConfig) -> None:
    """Additional method comparison plot required by visualization section."""

    print_section("Method comparison plot")
    n = 512 if not cfg.quick else 384
    x, dx = grid(-30.0, 30.0, n)
    dt, t_end = 0.002, 1.5
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    psi0 = gaussian_wavepacket(x, -8.0, 1.2, 2.0, dx)
    v = potential_harmonic(x, omega=0.1)
    methods = ["FTCS", "Backward-Euler", "Crank-Nicolson", "Split-Step-FFT", "RK4"]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for method in methods:
        _, hist = solve(method, psi0, v, x, t, dx, dt, store_every=len(t) - 1)
        psi = hist[-1]
        if np.max(np.abs(psi)) > 5:
            psi = psi / np.max(np.abs(psi)) * np.sqrt(probability_mass(psi0, dx))
        ax.plot(x, np.abs(psi) ** 2, lw=1.2, label=method)
    ax.set_xlim(-15, 5)
    ax.set_xlabel("x")
    ax.set_ylabel("|psi|^2")
    ax.set_title("Method comparison from same initial condition")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(cfg.outdir, "method_comparison.png"), dpi=cfg.dpi)
    plt.close(fig)


# =============================================================================
# 2D SSFM helper and experiments
# =============================================================================


def _make_2d_grid(
    nx: int, ny: int,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
) -> Tuple[Array, Array, Array, Array, float, float, Array, Array]:
    """Set up a 2D periodic grid and the matching Fourier frequency arrays.

    Returns (X, Y, x, y, dx, dy, KX, KY) where all 2D arrays use ij-indexing
    so that psi[ix, iy] corresponds to the point (x[ix], y[iy]).
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


def experiment_2d_free_propagation(cfg: RunConfig) -> None:
    """Figure 7: 2D free Gaussian wave packet propagation and mass conservation.

    Plots |psi|^2 at three snapshots (t=0, mid, final) and the total
    probability mass as a function of time to verify SSFM unitarity.
    """
    print_section("Figure 7: 2D free Gaussian propagation (Split-Step FFT)")
    nx = ny = 96 if cfg.quick else 128
    X, Y, x, y, dx, dy, KX, KY = _make_2d_grid(nx, ny, -15.0, 15.0, -15.0, 15.0)

    psi = gaussian_wavepacket_2d(
        X, Y, x0=-5.0, y0=-2.0, sigma=1.2, kx0=2.0, ky0=1.0, dx=dx, dy=dy
    )
    V = np.zeros_like(X)
    dt = 0.01
    n_steps = 150 if cfg.quick else 200
    snap_at = {0, n_steps // 2, n_steps}

    psi_cur = psi.copy()
    snaps: List[Tuple[float, Array]] = []
    mass_hist: List[float] = []
    t_hist: List[float] = []

    for step in range(n_steps + 1):
        if step in snap_at:
            snaps.append((step * dt, psi_cur.copy()))
        if step % 5 == 0:
            mass_hist.append(mass_2d(psi_cur, dx, dy))
            t_hist.append(step * dt)
        if step < n_steps:
            psi_cur = step_split_step_fft_2d(psi_cur, V, KX, KY, dt)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes_flat = axes.flatten()
    
    for idx, (t_snap, psi_snap) in enumerate(snaps):
        ax = axes_flat[idx]
        im = ax.imshow(
            np.abs(psi_snap).T ** 2, origin="lower", aspect="equal",
            extent=[x[0], x[-1], y[0], y[-1]], cmap="inferno",
            interpolation='bilinear'
        )
        cbar = plt.colorbar(im, ax=ax, shrink=0.82, label="|ψ|²", pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        ax.set_title(f"Time t = {t_snap:.2f}", fontweight='bold', fontsize=12, pad=10)
        ax.set_xlabel("x", labelpad=8)
        ax.set_ylabel("y", labelpad=8)
        ax.tick_params(labelsize=10)
    
    ax_mass = axes_flat[-1]
    ax_mass.plot(t_hist, mass_hist, lw=2.0, color='#2E86AB', marker='o', markersize=4, markevery=5)
    ax_mass.axhline(y=1.0, color='#C73E1D', linestyle='--', lw=1.5, alpha=0.7, label='Expected (M=1)')
    ax_mass.set_xlabel("Time t", labelpad=10)
    ax_mass.set_ylabel("Total Mass ∫|ψ|² dxdy", labelpad=10)
    ax_mass.set_title("Mass Conservation", fontweight='bold', fontsize=12, pad=10)
    ax_mass.set_ylim(0.95, 1.05)
    ax_mass.legend(fontsize=10, loc='best', framealpha=0.95)
    ax_mass.grid(True, alpha=0.3)
    
    fig.suptitle("Figure 7: 2D Free Gaussian Wave Packet (Split-Step FFT)", fontsize=15, fontweight='bold', y=0.98)
    fig.savefig(os.path.join(cfg.outdir, "figure7_2d_free_propagation.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(f"  Initial mass={mass_hist[0]:.6f}  Final mass={mass_hist[-1]:.6f}")


def experiment_2d_circle_scatter(cfg: RunConfig) -> None:
    """Figure 8: 2D wave scattering by a circular obstacle.

    A Gaussian packet moving in the +x direction encounters a high-potential
    circular obstacle. Three snapshots show the field before, approaching,
    and after the scattering event.
    """
    print_section("Figure 8: 2D circular obstacle scattering (Split-Step FFT)")
    nx = ny = 96 if cfg.quick else 128
    X, Y, x, y, dx, dy, KX, KY = _make_2d_grid(nx, ny, -14.0, 14.0, -10.0, 10.0)

    kx0 = 3.0
    psi0 = gaussian_wavepacket_2d(
        X, Y, x0=-7.0, y0=0.0, sigma=1.2, kx0=kx0, ky0=0.0, dx=dx, dy=dy
    )
    V = potential_circle_2d(X, Y, xc=2.0, yc=0.0, R=1.5, v0=20.0)
    dt = 0.01
    # Group center starts at x=-7; obstacle at x=2; travel time ≈ (7+2)/kx0 = 3
    t_approach = 1.5   # wave still approaching
    t_scatter = 3.5    # wave has passed / scattered
    step_approach = int(round(t_approach / dt))
    step_scatter = int(round(t_scatter / dt))

    psi_cur = psi0.copy()
    # Ordered dict so zip(axes, ...) always gives initial → approach → scatter
    psi_at: Dict[int, Array] = {0: psi0.copy()}
    for step in range(1, step_scatter + 1):
        psi_cur = step_split_step_fft_2d(psi_cur, V, KX, KY, dt)
        if step == step_approach:
            psi_at[step_approach] = psi_cur.copy()
    psi_at[step_scatter] = psi_cur.copy()

    titles = {
        0: "Initial  (t = 0)",
        step_approach: f"Approaching  (t = {t_approach:.1f})",
        step_scatter: f"Scattered  (t = {t_scatter:.1f})",
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes_flat = axes.flatten()
    vmax = float(np.max(np.abs(psi_at[step_scatter]) ** 2)) * 1.2
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    cx, cy, cr = 2.0, 0.0, 1.5
    
    for idx, (ax, (key, psi_plot)) in enumerate(zip(axes_flat[:3], psi_at.items())):
        im = ax.imshow(
            np.abs(psi_plot).T ** 2, origin="lower", aspect="equal",
            extent=[x[0], x[-1], y[0], y[-1]], cmap="inferno",
            vmin=0.0, vmax=max(vmax, 1e-12),
            interpolation='bilinear'
        )
        cbar = plt.colorbar(im, ax=ax, shrink=0.82, label="|ψ|²", pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        ax.plot(cx + cr * np.cos(theta), cy + cr * np.sin(theta),
                "w--", lw=2.5, label=f"Obstacle R={cr}")
        ax.set_title(titles[key], fontweight='bold', fontsize=12, pad=10)
        ax.set_xlabel("x", labelpad=8)
        ax.set_ylabel("y", labelpad=8)
        ax.tick_params(labelsize=10)
        if key == step_scatter:
            ax.legend(fontsize=10, loc='upper right', framealpha=0.95)
    
    ax_scatter = axes_flat[-1]
    psi_final = psi_at[step_scatter]
    radial_dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = radial_dist > cr + 0.5
    angles = np.arctan2(Y - cy, X - cx)
    hist, bins = np.histogram(angles[mask], bins=36, weights=np.abs(psi_final[mask])**2, density=True)
    ax_scatter.plot(bins[:-1], hist, lw=2.5, color='#F18F01')
    ax_scatter.fill_between(bins[:-1], hist, alpha=0.3, color='#F18F01')
    ax_scatter.set_xlabel("Scattering Angle (rad)", labelpad=8)
    ax_scatter.set_ylabel("Normalized Intensity", labelpad=8)
    ax_scatter.set_title("Scattering Angular Distribution", fontweight='bold', fontsize=12, pad=10)
    ax_scatter.set_xlim(-np.pi, np.pi)
    ax_scatter.grid(True, alpha=0.3)
    
    fig.suptitle("Figure 8: 2D Circular Obstacle Scattering (Split-Step FFT)", fontsize=15, fontweight='bold', y=0.98)
    fig.savefig(os.path.join(cfg.outdir, "figure8_2d_circle_scatter.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(f"  Circular scatter final mass={mass_2d(psi_cur, dx, dy):.6f}")


def experiment_2d_waveguide(cfg: RunConfig) -> None:
    """Figure 9: 2D waveguide vs free propagation.

    Compares a Gaussian packet propagating without confinement (spreads in y)
    against one inside V(x,y) = alpha*y^2 (remains narrow in y).
    Both 2D density maps and transverse (y) beam profiles are shown.
    """
    print_section("Figure 9: 2D waveguide vs free propagation (Split-Step FFT)")
    nx = ny = 96 if cfg.quick else 128
    X, Y, x, y, dx, dy, KX, KY = _make_2d_grid(nx, ny, -10.0, 16.0, -8.0, 8.0)

    psi0 = gaussian_wavepacket_2d(
        X, Y, x0=-6.0, y0=0.0, sigma=1.0, kx0=2.5, ky0=0.0, dx=dx, dy=dy
    )
    V_free = np.zeros_like(X)
    V_guide = potential_waveguide_2d(X, Y, alpha=0.4)
    dt = 0.01
    n_steps = 180 if cfg.quick else 250
    store_every = max(1, n_steps // 80)
    saved_t = []
    saved_psi_guide = []

    psi_free = psi0.copy()
    psi_guide = psi0.copy()
    for step in range(n_steps + 1):
        if step % store_every == 0 or step == n_steps:
            saved_t.append(step * dt)
            saved_psi_guide.append(psi_guide.copy())
        if step < n_steps:
            psi_free = step_split_step_fft_2d(psi_free, V_free, KX, KY, dt)
            psi_guide = step_split_step_fft_2d(psi_guide, V_guide, KX, KY, dt)

    t_final = n_steps * dt
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    cases = [
        ("Free Propagation", psi_free),
        (f"Waveguide V = {0.4}y²", psi_guide),
    ]
    for col, (label, psi_plot) in enumerate(cases):
        im = axes[0, col].imshow(
            np.abs(psi_plot).T ** 2, origin="lower", aspect="equal",
            extent=[x[0], x[-1], y[0], y[-1]], cmap="inferno",
            interpolation='bilinear'
        )
        cbar = plt.colorbar(im, ax=axes[0, col], shrink=0.82, label="|ψ|²", pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        axes[0, col].set_title(f"{label}\n|ψ|² at t = {t_final:.1f}", fontweight='bold', fontsize=11, pad=8)
        axes[0, col].set_xlabel("x", labelpad=6)
        axes[0, col].set_ylabel("y", labelpad=6)
        
        marginal_y = np.sum(np.abs(psi_plot) ** 2, axis=0) * dx
        axes[1, col].fill_between(y, marginal_y, alpha=0.3, color='#2E86AB')
        axes[1, col].plot(y, marginal_y, lw=2.0, color='#2E86AB')
        axes[1, col].set_xlabel("Transverse Position y", labelpad=8)
        axes[1, col].set_ylabel("Marginal ∫|ψ|² dx", labelpad=8)
        axes[1, col].set_title(f"{label}: Beam Profile", fontweight='bold', fontsize=11, pad=8)
        axes[1, col].grid(True, alpha=0.3)
    fig.suptitle("Figure 9: 2D Waveguide vs Free Propagation (Split-Step FFT)", fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(os.path.join(cfg.outdir, "figure9_2d_waveguide.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(
        f"  Free mass={mass_2d(psi_free, dx, dy):.6f}  "
        f"Waveguide mass={mass_2d(psi_guide, dx, dy):.6f}"
    )

    # Save animation
    if cfg.save_gif:
        save_2d_animation(
            cfg, X, Y, saved_t, saved_psi_guide,
            "figure9_waveguide_animation.gif",
            "2D Waveguide Propagation"
        )


def experiment_circular_obstacle_radius_sweep(cfg: RunConfig) -> pd.DataFrame:
    """Parameter sweep: circular obstacle radius vs transmitted/scattered mass."""
    print_section("Parameter Sweep: Circular Obstacle Radius")
    nx = ny = 80 if cfg.quick else 100
    X, Y, x, y, dx, dy, KX, KY = _make_2d_grid(nx, ny, -14.0, 14.0, -10.0, 10.0)
    kx0 = 3.0
    psi0 = gaussian_wavepacket_2d(
        X, Y, x0=-7.0, y0=0.0, sigma=1.2, kx0=kx0, ky0=0.0, dx=dx, dy=dy
    )
    dt = 0.01
    n_steps = 150 if cfg.quick else 200
    radii = [0.5, 1.0, 1.5, 2.0, 2.5]
    rows = []

    for R in radii:
        V = potential_circle_2d(X, Y, xc=2.0, yc=0.0, R=R, v0=20.0)
        psi = psi0.copy()
        for _ in range(n_steps):
            psi = step_split_step_fft_2d(psi, V, KX, KY, dt)
        
        # Calculate transmitted mass (right of obstacle) and scattered mass
        transmitted_mask = X > 6.0
        scattered_mask = (X > -2.0) & (X < 6.0) & (np.sqrt((X - 2.0)**2 + (Y - 0.0)**2) > R + 0.5)
        
        transmitted = mass_2d(psi * transmitted_mask.astype(float), dx, dy)
        scattered = mass_2d(psi * scattered_mask.astype(float), dx, dy)
        total = mass_2d(psi, dx, dy)
        
        rows.append({
            "radius": R,
            "transmitted_mass": transmitted,
            "scattered_mass": scattered,
            "total_mass": total
        })
        print(f"  R={R:.2f}: transmitted={transmitted:.4f}, scattered={scattered:.4f}, total={total:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "circular_obstacle_sweep.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(df["radius"], df["transmitted_mass"], "o-", label="Transmitted Mass", linewidth=2.5, markersize=10, color='#2E86AB')
    ax.plot(df["radius"], df["scattered_mass"], "s-", label="Scattered Mass", linewidth=2.5, markersize=10, color='#C73E1D')
    ax.set_xlabel("Obstacle Radius R", labelpad=10)
    ax.set_ylabel("Probability Mass", labelpad=10)
    ax.set_title("Circular Obstacle Radius vs Transmitted/Scattered Mass", fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)
    fig.savefig(os.path.join(cfg.outdir, "figure_circular_sweep.png"), dpi=cfg.dpi)
    plt.close(fig)

    return df


def experiment_waveguide_strength_sweep(cfg: RunConfig) -> pd.DataFrame:
    """Parameter sweep: waveguide strength (alpha) vs beam spreading."""
    print_section("Parameter Sweep: Waveguide Strength")
    nx = ny = 80 if cfg.quick else 100
    X, Y, x, y, dx, dy, KX, KY = _make_2d_grid(nx, ny, -10.0, 16.0, -8.0, 8.0)
    psi0 = gaussian_wavepacket_2d(
        X, Y, x0=-6.0, y0=0.0, sigma=1.0, kx0=2.5, ky0=0.0, dx=dx, dy=dy
    )
    dt = 0.01
    n_steps = 150 if cfg.quick else 200
    alphas = [0.1, 0.2, 0.4, 0.6, 1.0]
    rows = []

    for alpha in alphas:
        V = potential_waveguide_2d(X, Y, alpha=alpha)
        psi = psi0.copy()
        for _ in range(n_steps):
            psi = step_split_step_fft_2d(psi, V, KX, KY, dt)
        
        # Calculate beam width (standard deviation in y)
        marginal_y = np.sum(np.abs(psi) ** 2, axis=0) * dx
        mean_y = np.sum(y * marginal_y) * dy
        std_y = np.sqrt(np.sum((y - mean_y)**2 * marginal_y) * dy)
        
        rows.append({
            "alpha": alpha,
            "beam_width": std_y
        })
        print(f"  alpha={alpha:.2f}: beam_width={std_y:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "waveguide_sweep.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(df["alpha"], df["beam_width"], "o-", linewidth=2.5, markersize=10, color='#F18F01', markeredgecolor='white', markeredgewidth=1.5)
    ax.set_xlabel("Waveguide Strength α", labelpad=10)
    ax.set_ylabel("Beam Width σ_y (Standard Deviation)", labelpad=10)
    ax.set_title("Waveguide Strength vs Beam Spreading", fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)
    fig.savefig(os.path.join(cfg.outdir, "figure_waveguide_sweep.png"), dpi=cfg.dpi)
    plt.close(fig)

    return df


def experiment_1d_conservation_analysis(cfg: RunConfig) -> None:
    """Conservation analysis: mass error vs time for different methods."""
    print_section("Conservation Analysis: Mass Error vs Time")
    n = 256 if cfg.quick else 384
    x, dx = grid(-30.0, 30.0, n)
    t_end = 1.0
    dt = 0.002
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    psi0 = gaussian_wavepacket(x, -8.0, 1.2, 2.0, dx)
    v = potential_free(x)
    methods = ["FTCS", "Crank-Nicolson", "Split-Step-FFT"]
    
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    for method in methods:
        saved_t, saved_psi = solve(method, psi0, v, x, t, dx, dt)
        initial_mass = probability_mass(saved_psi[0], dx)
        mass_errors = [abs(probability_mass(psi, dx) - initial_mass) for psi in saved_psi]
        ax.semilogy(saved_t, mass_errors, lw=2.5, label=method, markersize=4, markevery=len(saved_t)//10)
    
    ax.set_xlabel("Time t", labelpad=12)
    ax.set_ylabel("Mass Error |M(t) - M(0)|", labelpad=12)
    ax.set_title("Mass Conservation Analysis: Error vs Time", fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.tick_params(labelsize=10)
    fig.savefig(os.path.join(cfg.outdir, "figure_conservation_analysis.png"), dpi=cfg.dpi)
    plt.close(fig)


def experiment_runtime_comparison(cfg: RunConfig) -> pd.DataFrame:
    """Runtime comparison for different methods."""
    print_section("Runtime Comparison")
    n = 256 if cfg.quick else 384
    x, dx = grid(-30.0, 30.0, n)
    t_end = 1.0
    dt = 0.002
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    psi0 = gaussian_wavepacket(x, -8.0, 1.2, 2.0, dx)
    v = potential_free(x)
    methods = ["FTCS", "Crank-Nicolson", "Split-Step-FFT"]
    rows = []

    for method in methods:
        start = time.perf_counter()
        saved_t, saved_psi = solve(method, psi0, v, x, t, dx, dt)
        runtime = time.perf_counter() - start
        initial_mass = probability_mass(saved_psi[0], dx)
        final_mass = probability_mass(saved_psi[-1], dx)
        mass_error = abs(final_mass - initial_mass)
        
        rows.append({
            "method": method,
            "runtime_s": runtime,
            "initial_mass": initial_mass,
            "final_mass": final_mass,
            "mass_error": mass_error
        })
        print(f"  {method}: runtime={runtime:.4f}s, mass_error={mass_error:.6e}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "runtime_comparison.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.axis("off")
    shown = df.copy()
    shown["runtime_s"] = shown["runtime_s"].map(lambda z: f"{z:.4f}")
    shown["initial_mass"] = shown["initial_mass"].map(lambda z: f"{z:.6f}")
    shown["final_mass"] = shown["final_mass"].map(lambda z: f"{z:.6f}")
    shown["mass_error"] = shown["mass_error"].map(lambda z: f"{z:.6e}")
    
    table = ax.table(
        cellText=shown.values, 
        colLabels=["Method", "Runtime (s)", "Initial Mass", "Final Mass", "Mass Error"],
        cellLoc="center", 
        loc="center",
        colColours=['#2E86AB'] * len(shown.columns),
        colWidths=[0.25, 0.18, 0.18, 0.18, 0.21]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    
    # Style header and data cells
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#2E86AB')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#F5F5F5')
    
    ax.set_title("Performance Comparison Table", fontsize=14, fontweight='bold', pad=20)
    fig.savefig(os.path.join(cfg.outdir, "figure_runtime_comparison.png"), dpi=cfg.dpi)
    plt.close(fig)

    return df


def experiment_2d_error_heatmap(cfg: RunConfig) -> None:
    """Error heatmap for 2D free Gaussian vs exact solution."""
    print_section("2D Error Heatmap")
    nx = ny = 80 if cfg.quick else 100
    X, Y, x, y, dx, dy, KX, KY = _make_2d_grid(nx, ny, -15.0, 15.0, -15.0, 15.0)
    x0, y0, sigma, kx0, ky0 = -5.0, -2.0, 1.2, 2.0, 1.0
    psi0 = gaussian_wavepacket_2d(X, Y, x0, y0, sigma, kx0, ky0, dx, dy)
    V = np.zeros_like(X)
    dt = 0.01
    n_steps = 100 if cfg.quick else 150
    
    psi = psi0.copy()
    for _ in range(n_steps):
        psi = step_split_step_fft_2d(psi, V, KX, KY, dt)
    
    psi_exact = exact_free_gaussian_2d(X, Y, n_steps * dt, x0, y0, sigma, kx0, ky0, dx, dy)
    error = np.abs(psi - psi_exact)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    extent = [x[0], x[-1], y[0], y[-1]]
    
    im1 = axes[0].imshow(np.abs(psi).T ** 2, origin="lower", aspect="equal", extent=extent, cmap="inferno", interpolation='bilinear')
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.82, label="|ψ|²")
    cbar1.ax.tick_params(labelsize=8)
    axes[0].set_title("Numerical Solution", fontweight='bold', fontsize=12, pad=10)
    
    im2 = axes[1].imshow(np.abs(psi_exact).T ** 2, origin="lower", aspect="equal", extent=extent, cmap="inferno", interpolation='bilinear')
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.82, label="|ψ|²")
    cbar2.ax.tick_params(labelsize=8)
    axes[1].set_title("Exact Analytical Solution", fontweight='bold', fontsize=12, pad=10)
    
    im3 = axes[2].imshow(error.T, origin="lower", aspect="equal", extent=extent, cmap="viridis", interpolation='bilinear')
    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.82, label="|ψ_num - ψ_exact|")
    cbar3.ax.tick_params(labelsize=8)
    axes[2].set_title("Absolute Error", fontweight='bold', fontsize=12, pad=10)
    
    for ax in axes:
        ax.set_xlabel("x", labelpad=8)
        ax.set_ylabel("y", labelpad=8)
        ax.tick_params(labelsize=9)
    
    fig.suptitle("Figure: 2D Numerical Error Analysis", fontsize=14, fontweight='bold', y=1.02)
    
    fig.savefig(os.path.join(cfg.outdir, "figure_2d_error_heatmap.png"), dpi=cfg.dpi)
    plt.close(fig)
    
    print(f"  Max error: {np.max(error):.6e}")


def experiment_2d_convergence(cfg: RunConfig) -> pd.DataFrame:
    """
    2D grid convergence study.

    Tests the Split-Step FFT method on a 2D free Gaussian wave packet
    against the analytical solution at various grid resolutions.
    Measures L1, L2, Linf errors and convergence order.
    """
    print_section("2D Convergence Study")

    grid_sizes = [40, 60, 80, 100] if not cfg.quick else [32, 48, 64]
    results = []

    for nx in tqdm(grid_sizes, desc="2D convergence"):
        ny = nx
        X, Y, x, y, dx, dy, KX, KY = _make_2d_grid(
            nx, ny, -15.0, 15.0, -15.0, 15.0
        )

        x0, y0, sigma, kx0, ky0 = -5.0, -2.0, 1.2, 2.0, 1.0
        psi0 = gaussian_wavepacket_2d(X, Y, x0, y0, sigma, kx0, ky0, dx, dy)
        V = np.zeros_like(X)

        dt = 0.005
        n_steps = 100 if cfg.quick else 150
        t_final = n_steps * dt

        psi = psi0.copy()
        for _ in range(n_steps):
            psi = step_split_step_fft_2d(psi, V, KX, KY, dt)

        psi_exact = exact_free_gaussian_2d(X, Y, t_final, x0, y0, sigma, kx0, ky0, dx, dy)

        diff = psi - psi_exact
        l1 = float(np.sum(np.abs(diff)) * dx * dy)
        l2 = float(np.sqrt(np.sum(np.abs(diff) ** 2) * dx * dy))
        linf = float(np.max(np.abs(diff)))

        mass_num = mass_2d(psi, dx, dy)
        mass_exact = mass_2d(psi_exact, dx, dy)

        results.append({
            "grid_size": nx,
            "N": nx * ny,
            "dx": dx,
            "dy": dy,
            "dt": dt,
            "n_steps": n_steps,
            "L1_error": l1,
            "L2_error": l2,
            "Linf_error": linf,
            "mass_numerical": mass_num,
            "mass_exact": mass_exact,
            "mass_error": abs(mass_num - mass_exact),
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(cfg.outdir, "2d_convergence_study.csv"), index=False)

    if len(df) >= 2:
        df["log_N"] = np.log(df["N"])
        df["log_L2"] = np.log(df["L2_error"].replace(0, np.nan))
        valid = df.dropna(subset=["log_L2"])
        if len(valid) >= 2:
            coeffs = np.polyfit(valid["log_N"].values, valid["log_L2"].values, 1)
            df["convergence_order"] = -coeffs[0]
            print(f"\nFitted convergence order (L2 vs N): {-coeffs[0]:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axes[0].loglog(df["N"], df["L1_error"], "o-", lw=2, color="#2E86AB", label="L1")
    axes[0].loglog(df["N"], df["L2_error"], "s-", lw=2, color="#A23B72", label="L2")
    axes[0].loglog(df["N"], df["Linf_error"], "^-", lw=2, color="#F18F01", label="Linf")
    axes[0].set_xlabel("Number of Grid Points (N)", labelpad=8)
    axes[0].set_ylabel("Error", labelpad=8)
    axes[0].set_title("L1, L2, Linf Errors vs Grid Size", fontweight='bold', fontsize=12, pad=10)
    axes[0].legend(fontsize=10, loc='best')
    axes[0].grid(True, alpha=0.3, which='both')

    axes[1].semilogy(df["grid_size"], df["mass_error"], "o-", lw=2, color="#2E86AB")
    axes[1].set_xlabel("Grid Size (nx = ny)", labelpad=8)
    axes[1].set_ylabel("Mass Conservation Error", labelpad=8)
    axes[1].set_title("Mass Conservation Error", fontweight='bold', fontsize=12, pad=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(range(len(df)), df["mass_numerical"], width=0.6,
                color="#2E86AB", alpha=0.7, label="Numerical")
    axes[2].axhline(y=1.0, color="#C73E1D", linestyle="--", lw=2, label="Expected (M=1)")
    axes[2].set_xticks(range(len(df)))
    axes[2].set_xticklabels([str(n) for n in df["grid_size"]])
    axes[2].set_xlabel("Grid Size", labelpad=8)
    axes[2].set_ylabel("Mass", labelpad=8)
    axes[2].set_title("Mass Conservation Across Grid Sizes", fontweight='bold', fontsize=12, pad=10)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_ylim(0.99, 1.01)

    fig.suptitle("Figure: 2D Convergence Analysis", fontsize=14, fontweight='bold', y=1.02)

    fig.savefig(os.path.join(cfg.outdir, "figure_2d_convergence.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(f"\n2D Convergence Results:")
    print(df.to_string(index=False))

    return df


def experiment_2d_circular_obstacle_with_animation(cfg: RunConfig) -> None:
    """2D circular obstacle scattering with animation."""
    print_section("2D Circular Obstacle (with Animation)")
    nx = ny = 96 if cfg.quick else 128
    X, Y, x, y, dx, dy, KX, KY = _make_2d_grid(nx, ny, -14.0, 14.0, -10.0, 10.0)

    kx0 = 3.0
    psi0 = gaussian_wavepacket_2d(
        X, Y, x0=-7.0, y0=0.0, sigma=1.2, kx0=kx0, ky0=0.0, dx=dx, dy=dy
    )
    V = potential_circle_2d(X, Y, xc=2.0, yc=0.0, R=1.5, v0=20.0).astype(complex)
    # Add absorbing boundaries
    V += absorbing_potential_2d(X, Y, width=3.0, strength=2.0)
    
    dt = 0.01
    n_steps = 250 if cfg.quick else 350
    store_every = max(1, n_steps // 80)
    saved_t = []
    saved_psi = []

    psi_cur = psi0.copy()
    for step in range(n_steps + 1):
        if step % store_every == 0 or step == n_steps:
            saved_t.append(step * dt)
            saved_psi.append(psi_cur.copy())
        if step < n_steps:
            psi_cur = step_split_step_fft_2d(psi_cur, V, KX, KY, dt)

    t_approach = 1.5
    t_scatter = 3.5
    step_approach = int(round(t_approach / dt))
    step_scatter = int(round(t_scatter / dt))
    step_approach = min(step_approach, n_steps)
    step_scatter = min(step_scatter, n_steps)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes_flat = axes.flatten()
    titles = {0: "Initial (t=0)", step_approach: f"Approaching (t={t_approach:.1f})", step_scatter: f"Scattered (t={t_scatter:.1f})"}
    
    def find_closest_idx(target_step):
        for i, t_val in enumerate(saved_t):
            if t_val >= target_step * dt:
                return i
        return len(saved_t) - 1
    
    idx0 = 0
    idx_approach = find_closest_idx(step_approach)
    idx_scatter = find_closest_idx(step_scatter)
    indices = [idx0, idx_approach, idx_scatter]
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    cx, cy, cr = 2.0, 0.0, 1.5
    
    for idx, (ax, i) in enumerate(zip(axes_flat[:3], indices)):
        im = ax.imshow(
            np.abs(saved_psi[i]).T ** 2, origin="lower", aspect="equal",
            extent=[x[0], x[-1], y[0], y[-1]], cmap="inferno",
            interpolation='bilinear'
        )
        cbar = plt.colorbar(im, ax=ax, shrink=0.82, label="|ψ|²", pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        ax.plot(cx + cr * np.cos(theta), cy + cr * np.sin(theta), "w--", lw=2.5)
        ax.set_title(titles.get(i, f"t={saved_t[i]:.1f}"), fontweight='bold', fontsize=13, pad=10)
        ax.set_xlabel("x", labelpad=8)
        ax.set_ylabel("y", labelpad=8)
        ax.tick_params(labelsize=10)
    
    ax_scatter = axes_flat[-1]
    psi_final = saved_psi[idx_scatter]
    radial_dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = radial_dist > cr + 0.5
    angles = np.arctan2(Y - cy, X - cx)
    hist, bins = np.histogram(angles[mask], bins=36, weights=np.abs(psi_final[mask])**2, density=True)
    ax_scatter.plot(bins[:-1], hist, lw=2.5, color='#F18F01')
    ax_scatter.fill_between(bins[:-1], hist, alpha=0.3, color='#F18F01')
    ax_scatter.set_xlabel("Scattering Angle (rad)", labelpad=8)
    ax_scatter.set_ylabel("Normalized Intensity", labelpad=8)
    ax_scatter.set_title("Scattering Angular Distribution", fontweight='bold', fontsize=12, pad=10)
    ax_scatter.set_xlim(-np.pi, np.pi)
    ax_scatter.grid(True, alpha=0.3)
    
    fig.suptitle("Figure 8: 2D Circular Obstacle Scattering with Absorbing Boundaries", fontsize=14, fontweight='bold', y=0.98)
    fig.savefig(os.path.join(cfg.outdir, "figure8_2d_circular_obstacle_absorbing.png"), dpi=cfg.dpi)
    plt.close(fig)

    # Save animation
    if cfg.save_gif:
        save_2d_animation(
            cfg, X, Y, saved_t, saved_psi,
            "figure8_circular_obstacle_animation.gif",
            "2D Circular Obstacle Scattering",
            obstacle_plot=(cx, cy, cr)
        )

    print(f"  Final mass={mass_2d(psi_cur, dx, dy):.6f}")


# =============================================================================
# Visualization Module
# =============================================================================


class TDSEVisualizer:
    """
    Unified visualization interface for TDSE simulations.
    
    This class provides high-level methods for creating standard
    visualizations of quantum wave function evolution.
    
    Example:
        >>> viz = TDSEVisualizer(config)
        >>> viz.plot_wavepacket(x, psi, title="Free Particle")
        >>> viz.plot_probability_density(x, psi, potential=V)
    """
    
    def __init__(self, config: TDSEConfig | RunConfig):
        self.config = config
        if isinstance(config, TDSEConfig):
            self.config = config.to_run_config()
    
    def plot_wavepacket(
        self,
        x: Array,
        psi: Array,
        title: str = "Wave Function",
        filename: Optional[str] = None,
        show_exact: Optional[Array] = None,
    ) -> None:
        """
        Plot wave function |psi|^2 with optional exact solution.
        
        Args:
            x: Spatial grid
            psi: Wave function array
            title: Plot title
            filename: If provided, save to file
            show_exact: Exact solution for comparison
        """
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        
        ax.plot(x, np.abs(psi) ** 2, lw=2.0, label="Numerical", color='#2E86AB')
        
        if show_exact is not None:
            ax.plot(x, np.abs(show_exact) ** 2, "k--", lw=1.5, 
                   label="Exact", alpha=0.8)
        
        ax.set_xlabel("Position x", labelpad=10)
        ax.set_ylabel("Probability Density |ψ|²", labelpad=10)
        ax.set_title(title, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(os.path.join(self.config.outdir, filename), 
                       dpi=self.config.dpi)
        
        plt.close()
    
    def plot_probability_density(
        self,
        x: Array,
        psi: Array,
        potential: Optional[Array] = None,
        title: str = "Probability Density",
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot probability density with potential barrier overlay.
        
        Args:
            x: Spatial grid
            psi: Wave function array
            potential: Potential array to overlay
            title: Plot title
            filename: If provided, save to file
        """
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        
        ax.plot(x, np.abs(psi) ** 2, lw=2.0, label="|ψ|²", color='#2E86AB')
        
        if potential is not None:
            vmax = np.max(potential)
            if vmax > 0:
                ax.fill_between(x, 0, potential / vmax * 0.1,
                              color="#C73E1D", alpha=0.3, label="Potential")
        
        ax.set_xlabel("Position x", labelpad=10)
        ax.set_ylabel("Probability Density |ψ|²", labelpad=10)
        ax.set_title(title, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(os.path.join(self.config.outdir, filename),
                       dpi=self.config.dpi)
        
        plt.close()
    
    def plot_2d_density(
        self,
        X: Array,
        Y: Array,
        psi: Array,
        title: str = "2D Probability Density",
        filename: Optional[str] = None,
        obstacle: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """
        Plot 2D probability density heatmap.
        
        Args:
            X, Y: 2D meshgrid arrays
            psi: 2D wave function array
            title: Plot title
            filename: If provided, save to file
            obstacle: Tuple of (cx, cy, radius) to draw circular obstacle
        """
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        
        extent = [X.min(), X.max(), Y.min(), Y.max()]
        im = ax.imshow(
            np.abs(psi).T ** 2,
            origin="lower",
            aspect="equal",
            extent=extent,
            cmap="inferno",
            interpolation='bilinear'
        )
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, label="|ψ|²", pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        
        if obstacle is not None:
            cx, cy, cr = obstacle
            theta = np.linspace(0.0, 2.0 * np.pi, 100)
            ax.plot(cx + cr * np.cos(theta), cy + cr * np.sin(theta),
                   "w--", lw=2.0, label=f"Obstacle R={cr}")
            ax.legend(fontsize=10, loc='upper right', framealpha=0.95)
        
        ax.set_xlabel("x", labelpad=10)
        ax.set_ylabel("y", labelpad=10)
        ax.set_title(title, fontweight='bold', pad=15)
        
        if filename:
            plt.savefig(os.path.join(self.config.outdir, filename),
                       dpi=self.config.dpi)
        
        plt.close()
    
    def plot_comparison(
        self,
        data: Dict[str, Array],
        x: Array,
        title: str = "Method Comparison",
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot comparison of multiple wave functions.
        
        Args:
            data: Dictionary mapping names to wave function arrays
            x: Spatial grid
            title: Plot title
            filename: If provided, save to file
        """
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#95C623']
        
        for idx, (name, psi) in enumerate(data.items()):
            ax.plot(x, np.abs(psi) ** 2, lw=2.0, label=name,
                   color=colors[idx % len(colors)])
        
        ax.set_xlabel("Position x", labelpad=10)
        ax.set_ylabel("Probability Density |ψ|²", labelpad=10)
        ax.set_title(title, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(os.path.join(self.config.outdir, filename),
                       dpi=self.config.dpi)
        
        plt.close()


# =============================================================================
# Animations
# =============================================================================


def save_wavepacket_animation(cfg: RunConfig) -> None:
    """Figure 5: free Gaussian wavepacket animation."""

    print_section("Figure 5: wavepacket animation")
    n = 512 if not cfg.quick else 384
    x, dx = grid(-30.0, 30.0, n)
    dt, t_end = 0.004, 3.0
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    psi0 = gaussian_wavepacket(x, -10.0, 1.1, 2.0, dx)
    v = potential_free(x)
    saved_t, hist = solve("Split-Step-FFT", psi0, v, x, t, dx, dt, store_every=max(1, len(t) // 140))

    # Reduce frames for faster saving
    max_frames = 30
    if len(saved_t) > max_frames:
        step = len(saved_t) // max_frames
        hist = np.array(hist)
        saved_t = np.array(saved_t)
        indices = np.arange(0, len(saved_t), step)[:max_frames]
        saved_t = saved_t[indices].tolist()
        hist = hist[indices]

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    line, = ax.plot([], [], lw=2, label="Split-Step FFT")
    exact_line, = ax.plot([], [], "k--", lw=1, label="exact")
    ax.set_xlim(-20, 8)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("x")
    ax.set_ylabel("|psi|^2")
    ax.set_title("Figure 5: free wavepacket propagation")
    ax.legend(fontsize=8)

    def init():
        line.set_data([], [])
        exact_line.set_data([], [])
        return line, exact_line

    def update(frame: int):
        tt = saved_t[frame]
        line.set_data(x, np.abs(hist[frame]) ** 2)
        exact = exact_free_gaussian(x, tt, -10.0, 1.1, 2.0, dx)
        exact_line.set_data(x, np.abs(exact) ** 2)
        ax.set_title(f"Figure 5: free wavepacket propagation, t={tt:.2f}")
        return line, exact_line

    if cfg.save_gif:
        ani = FuncAnimation(fig, update, frames=len(saved_t), init_func=init, blit=True)
        ani.save(os.path.join(cfg.outdir, "figure5_wavepacket_animation.gif"), writer=PillowWriter(fps=20), dpi=150)
    fig.savefig(os.path.join(cfg.outdir, "figure5_wavepacket_last_frame.png"), dpi=cfg.dpi)
    plt.close(fig)


def save_tunneling_animation(cfg: RunConfig, x: Array, v: Array, saved_t: Array, hist: Array, label: str) -> None:
    """Animated rectangular barrier tunneling for one representative case."""

    # Reduce frames for faster saving
    max_frames = 30
    if len(saved_t) > max_frames:
        step = len(saved_t) // max_frames
        hist = np.array(hist)
        saved_t = np.array(saved_t)
        indices = np.arange(0, len(saved_t), step)[:max_frames]
        saved_t = saved_t[indices].tolist()
        hist = hist[indices]

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    line, = ax.plot([], [], lw=2)
    barrier = v / max(np.max(v), 1e-12) * 0.08
    ax.fill_between(x, 0, barrier, color="tab:red", alpha=0.3)
    ax.set_xlim(-95, 95)
    ax.set_ylim(0, max(0.18, np.max(np.abs(hist) ** 2) * 1.15))
    ax.set_xlabel("x")
    ax.set_ylabel("|psi|^2")
    ax.set_title(f"Figure 6: tunneling animation {label}")

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame: int):
        line.set_data(x, np.abs(hist[frame]) ** 2)
        ax.set_title(f"Figure 6: tunneling animation {label}, t={saved_t[frame]:.2f}")
        return (line,)

    ani = FuncAnimation(fig, update, frames=len(saved_t), init_func=init, blit=True)
    ani.save(os.path.join(cfg.outdir, "figure6_tunneling_animation.gif"), writer=PillowWriter(fps=20), dpi=150)
    plt.close(fig)


def save_2d_animation(
    cfg: RunConfig,
    X: Array,
    Y: Array,
    saved_t: Array,
    hist: Array,
    filename: str,
    title: str,
    obstacle_plot: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Save a fast 2D animation as a GIF. Optimized for speed."""
    # Reduce frames for faster saving - sample 50 frames for longer animation
    max_frames = 100
    if len(saved_t) > max_frames:
        step = len(saved_t) // max_frames
        hist = np.array(hist)
        saved_t = np.array(saved_t)
        indices = np.arange(0, len(saved_t), step)[:max_frames]
        saved_t = saved_t[indices].tolist()
        hist = hist[indices]
    
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    x = X[:, 0]
    y = Y[0, :]
    extent = [x[0], x[-1], y[0], y[-1]]
    
    vmax = np.max(np.abs(hist) ** 2) * 1.1
    im = ax.imshow(
        np.abs(hist[0]).T ** 2,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
        interpolation='none'
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}, t = {saved_t[0]:.2f}", fontsize=10)

    if obstacle_plot is not None:
        cx, cy, cr = obstacle_plot
        theta = np.linspace(0.0, 2.0 * np.pi, 100)
        ax.plot(cx + cr * np.cos(theta), cy + cr * np.sin(theta), "w--", lw=1.5)

    def init():
        im.set_data(np.abs(hist[0]).T ** 2)
        return (im,)

    def update(frame: int):
        im.set_data(np.abs(hist[frame]).T ** 2)
        ax.set_title(f"{title}, t = {saved_t[frame]:.2f}", fontsize=10)
        return (im,)

    ani = FuncAnimation(fig, update, frames=len(saved_t), init_func=init, blit=True)
    ani.save(os.path.join(cfg.outdir, filename), writer=PillowWriter(fps=20), dpi=150)
    plt.close(fig)


# =============================================================================
# Experiment Runner Module
# =============================================================================


class ExperimentRunner:
    """
    Unified experiment runner for TDSE simulations.
    
    This class provides a high-level interface for running standard
    experiments and parameter studies in TDSE simulations.
    
    Example:
        >>> runner = ExperimentRunner(config)
        >>> runner.run_all()  # Run all experiments
        >>> runner.run_convergence_study()
        >>> runner.run_tunneling_experiment()
    """
    
    def __init__(self, config: TDSEConfig | RunConfig):
        self.config = config
        if isinstance(config, TDSEConfig):
            self._legacy_config = config.to_run_config()
        else:
            self._legacy_config = config
        ensure_outdir(self._legacy_config.outdir)
        setup_plot_style(self._legacy_config.dpi)
    
    def run_all(self) -> None:
        """Run all available experiments."""
        experiments = [
            self.run_analytic_comparison,
            self.run_convergence_study,
            self.run_stability_analysis,
            self.run_performance_comparison,
            self.run_tunneling_experiment,
            self.run_2d_free_propagation,
            self.run_2d_circle_scattering,
            self.run_2d_waveguide,
            self.run_parameter_sweeps,
            self.run_conservation_analysis,
            self.run_runtime_comparison,
            self.run_error_analysis,
        ]
        
        for experiment in experiments:
            try:
                experiment()
            except Exception as e:
                print(f"Error in {experiment.__name__}: {e}")
    
    def run_analytic_comparison(self) -> None:
        """Run analytic vs numerical comparison."""
        experiment_analytic_vs_numerical(self._legacy_config)
    
    def run_convergence_study(self) -> None:
        """Run convergence order study."""
        experiment_convergence(self._legacy_config)
    
    def run_stability_analysis(self) -> None:
        """Run stability analysis."""
        experiment_stability(self._legacy_config)
    
    def run_performance_comparison(self) -> None:
        """Run performance comparison."""
        experiment_performance(self._legacy_config)
    
    def run_tunneling_experiment(self) -> None:
        """Run quantum tunneling experiment."""
        experiment_tunneling(self._legacy_config)
    
    def run_2d_free_propagation(self) -> None:
        """Run 2D free Gaussian propagation."""
        experiment_2d_free_propagation(self._legacy_config)
    
    def run_2d_circle_scattering(self) -> None:
        """Run 2D circular obstacle scattering."""
        experiment_2d_circular_obstacle_with_animation(self._legacy_config)
    
    def run_2d_waveguide(self) -> None:
        """Run 2D waveguide propagation."""
        experiment_2d_waveguide(self._legacy_config)
    
    def run_parameter_sweeps(self) -> None:
        """Run parameter sweep experiments."""
        experiment_circular_obstacle_radius_sweep(self._legacy_config)
        experiment_waveguide_strength_sweep(self._legacy_config)
    
    def run_conservation_analysis(self) -> None:
        """Run mass conservation analysis."""
        experiment_1d_conservation_analysis(self._legacy_config)
    
    def run_runtime_comparison(self) -> None:
        """Run runtime comparison."""
        experiment_runtime_comparison(self._legacy_config)
    
    def run_error_analysis(self) -> None:
        """Run 2D error heatmap analysis."""
        experiment_2d_error_heatmap(self._legacy_config)


# =============================================================================
# Main driver
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="1D TDSE numerical PDE demo in dimensionless units.")
    parser.add_argument("--outdir", default="tdse_outputs2", help="Output directory.")
    parser.add_argument("--quick", action="store_true", help="Run a smaller/faster version.")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation.")
    args = parser.parse_args()

    cfg = RunConfig(outdir=args.outdir, quick=args.quick, save_gif=not args.no_gif)
    ensure_outdir(cfg.outdir)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Setup high-quality plot style
    setup_plot_style(cfg.dpi)

    print_section("Dimensionless 1D Time-Dependent Schrodinger Equation Demo")
    print("Equation: i psi_t = -1/2 psi_xx + V(x) psi")
    print("All quantities are dimensionless.")
    print(f"Output directory: {os.path.abspath(cfg.outdir)}")
    print("\nNumerical stability reminders:")
    print("- FTCS is included because it is requested, but it is intrinsically unstable for TDSE.")
    print("- Crank-Nicolson uses scipy.linalg.solve_banded on a tridiagonal system.")
    print("- Split-Step Fourier uses numpy.fft and assumes periodic spectral propagation.")
    print("- RK4 + centered differences is conditionally stable; keep dt/dx^2 small.")

    total_start = time.perf_counter()
    experiment_analytic_vs_numerical(cfg)
    experiment_convergence(cfg)
    experiment_stability(cfg)
    experiment_performance(cfg)
    experiment_method_comparison(cfg)
    if cfg.save_gif:
        save_wavepacket_animation(cfg)
    else:
        print_section("Figure 5: GIF skipped by --no-gif")
    experiment_tunneling(cfg)
    experiment_2d_free_propagation(cfg)
    experiment_2d_circular_obstacle_with_animation(cfg)
    experiment_2d_waveguide(cfg)
    
    # New experiments
    experiment_circular_obstacle_radius_sweep(cfg)
    experiment_waveguide_strength_sweep(cfg)
    experiment_1d_conservation_analysis(cfg)
    experiment_runtime_comparison(cfg)
    experiment_2d_convergence(cfg)
    experiment_2d_error_heatmap(cfg)
    
    total_runtime = time.perf_counter() - total_start

    print_section("Done")
    print(f"Total runtime: {total_runtime:.2f} s")
    print("Generated files:")
    for name in sorted(os.listdir(cfg.outdir)):
        print(f"- {name}")


if __name__ == "__main__":
    main()