"""
TDSE - Time-Dependent Schrodinger Equation Package
================================================

A modular Python package for numerical solutions of the time-dependent
Schrodinger equation (TDSE) in 1D and 2D.

Features:
--------
- Multiple numerical methods (FTCS, Crank-Nicolson, Split-Step FFT, RK4)
- 1D and 2D simulations
- Various potential types (free, barriers, waveguides, obstacles)
- Visualization and animation tools
- Convergence and stability analysis

Quick Start:
-----------
    import tdse

    # Create configuration
    config = tdse.TDSEConfig(quick=True)

    # Create potential and solver
    potential = tdse.TDSEPotentialFactory.create("free")
    solver = tdse.TDSESolverFactory.create("Crank-Nicolson")

    # Setup grid and initial state
    x, dx = tdse.grid(-30.0, 30.0, 384)
    psi0 = tdse.gaussian_wavepacket(x, -8.0, 1.2, 2.0, dx)

    # Run simulation
    saved_t, psi_hist = tdse.solve("Crank-Nicolson", psi0, potential(x), x, t, dx, dt)

    # Visualize
    viz = tdse.TDSEVisualizer(config)
    viz.plot_wavepacket(x, psi_hist[-1], title="Final State")

Modules:
--------
- config: Configuration classes
- potentials: Potential functions and factories
- solvers: Numerical solvers
- visualization: Plotting and animation tools
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "TDSE Development Team"

# Import from submodules using relative imports
from .config import TDSEConfig, RunConfig, ensure_outdir, setup_plot_style
from .potentials import (
    grid,
    normalize,
    probability_mass,
    mass_2d,
    l1_l2_linf_error,
    gaussian_wavepacket,
    gaussian_wavepacket_2d,
    exact_free_gaussian,
    exact_free_gaussian_2d,
    potential_free,
    potential_rect_barrier,
    compute_transmission_reflection,
    analyze_barrier_scattering,
    TDSEPotentialFactory,
)
from .solvers import (
    solve,
    solve_2d,
    TDSESolverFactory,
)
from .visualization import (
    TDSEVisualizer,
    save_wavepacket_animation,
    save_tunneling_animation,
    save_2d_animation,
)

# Convenience imports
from .potentials import (
    Potential1D,
    Potential2D,
    FreePotential,
    HarmonicPotential,
    RectangularBarrier,
    CircularObstacle,
    Waveguide,
)
from .solvers import (
    Solver1D,
    FTCSSolver,
    BackwardEulerSolver,
    CrankNicolsonSolver,
    RKK4Solver,
    SplitStepFFTSolver,
)

# Define public API
__all__ = [
    # Configuration
    "TDSEConfig",
    "RunConfig",
    "ensure_outdir",
    "setup_plot_style",
    # Potentials
    "grid",
    "normalize",
    "probability_mass",
    "mass_2d",
    "l1_l2_linf_error",
    "gaussian_wavepacket",
    "gaussian_wavepacket_2d",
    "exact_free_gaussian",
    "exact_free_gaussian_2d",
    "potential_free",
    "potential_rect_barrier",
    "compute_transmission_reflection",
    "analyze_barrier_scattering",
    "TDSEPotentialFactory",
    # Solvers
    "solve",
    "solve_2d",
    "TDSESolverFactory",
    # Visualization
    "TDSEVisualizer",
    "save_wavepacket_animation",
    "save_tunneling_animation",
    "save_2d_animation",
    # Base classes
    "Potential1D",
    "Potential2D",
    "Solver1D",
    "FreePotential",
    "HarmonicPotential",
    "RectangularBarrier",
    "CircularObstacle",
    "Waveguide",
    "FTCSSolver",
    "BackwardEulerSolver",
    "CrankNicolsonSolver",
    "RKK4Solver",
    "SplitStepFFTSolver",
]
