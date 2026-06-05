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
- Pre-built experiment suite

Quick Start:
-----------
    import tdse

    # Create configuration
    config = tdse.TDSEConfig(quick=True)

    # Setup grid and initial state
    x, dx = tdse.grid(-30.0, 30.0, 384)
    psi0 = tdse.gaussian_wavepacket(x, -8.0, 1.2, 2.0, dx)
    v = tdse.potential_free(x)

    # Run simulation
    saved_t, psi_hist = tdse.solve("Crank-Nicolson", psi0, v, x, t, dx, dt)

    # Visualize
    viz = tdse.TDSEVisualizer(config)
    viz.plot_wavepacket(x, psi_hist[-1], title="Final State")

Modules:
--------
- config: Configuration classes and plot style setup
- potentials: Potential classes, wave functions, and utility functions
- solvers: Numerical solvers (1D and 2D)
- visualization: Plotting and animation tools
- experiments: Pre-built experiment suite for 1D and 2D TDSE simulations
"""

from __future__ import annotations

__version__ = "1.1.0"
__author__ = "TDSE Development Team"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
from .config import TDSEConfig, RunConfig, ensure_outdir, setup_plot_style

# ---------------------------------------------------------------------------
# Potentials — utility functions
# ---------------------------------------------------------------------------
from .potentials import (
    grid,
    normalize,
    probability_mass,
    mass_2d,
    l1_l2_linf_error,
    align_global_phase,
    print_section,
    dt_warning,
    make_2d_grid,
)

# ---------------------------------------------------------------------------
# Potentials — wave functions and analytic solutions
# ---------------------------------------------------------------------------
from .potentials import (
    gaussian_wavepacket,
    gaussian_wavepacket_2d,
    exact_free_gaussian,
    exact_free_gaussian_2d,
    infinite_well_eigenstate,
    harmonic_eigenstate_low,
)

# ---------------------------------------------------------------------------
# Potentials — standalone potential functions
# ---------------------------------------------------------------------------
from .potentials import (
    potential_free,
    potential_harmonic,
    potential_rect_barrier,
    potential_infinite_well_mask,
    potential_barrier_2d,
    potential_circle_2d,
    potential_waveguide_2d,
    absorbing_potential_2d,
)

# ---------------------------------------------------------------------------
# Potentials — scattering analysis
# ---------------------------------------------------------------------------
from .potentials import (
    compute_transmission_reflection,
    analyze_barrier_scattering,
    transmission_reflection,
)

# ---------------------------------------------------------------------------
# Potentials — base classes and concrete potential classes
# ---------------------------------------------------------------------------
from .potentials import (
    Potential1D,
    Potential2D,
    FreePotential,
    HarmonicPotential,
    RectangularBarrier,
    CircularObstacle,
    Waveguide,
    TDSEPotentialFactory,
)

# ---------------------------------------------------------------------------
# Solvers — low-level step functions
# ---------------------------------------------------------------------------
from .solvers import (
    laplacian_dirichlet,
    hamiltonian_apply,
    banded_hamiltonian,
    apply_tridiagonal,
    step_ftcs,
    step_backward_euler,
    step_crank_nicolson,
    step_rk4,
    step_split_step_fft,
    step_split_step_fft_2d,
    step_adi_2d,
)

# ---------------------------------------------------------------------------
# Solvers — high-level solve dispatchers
# ---------------------------------------------------------------------------
from .solvers import (
    solve,
    solve_2d,
)

# ---------------------------------------------------------------------------
# Solvers — base classes, concrete solvers, and factory
# ---------------------------------------------------------------------------
from .solvers import (
    Solver1D,
    FTCSSolver,
    BackwardEulerSolver,
    CrankNicolsonSolver,
    RKK4Solver,
    SplitStepFFTSolver,
    TDSESolverFactory,
)

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
from .visualization import (
    TDSEVisualizer,
    save_wavepacket_animation,
    save_tunneling_animation,
    save_2d_animation,
)

# ---------------------------------------------------------------------------
# Experiments (lazy import to avoid circular deps)
# ---------------------------------------------------------------------------
from . import experiments  # noqa: F401

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # Configuration
    "TDSEConfig",
    "RunConfig",
    "ensure_outdir",
    "setup_plot_style",
    # Utility
    "grid",
    "normalize",
    "probability_mass",
    "mass_2d",
    "l1_l2_linf_error",
    "align_global_phase",
    "print_section",
    "dt_warning",
    "make_2d_grid",
    # Wave functions
    "gaussian_wavepacket",
    "gaussian_wavepacket_2d",
    "exact_free_gaussian",
    "exact_free_gaussian_2d",
    "infinite_well_eigenstate",
    "harmonic_eigenstate_low",
    # Potential functions
    "potential_free",
    "potential_harmonic",
    "potential_rect_barrier",
    "potential_infinite_well_mask",
    "potential_barrier_2d",
    "potential_circle_2d",
    "potential_waveguide_2d",
    "absorbing_potential_2d",
    # Scattering
    "compute_transmission_reflection",
    "analyze_barrier_scattering",
    "transmission_reflection",
    # Potential classes
    "Potential1D",
    "Potential2D",
    "FreePotential",
    "HarmonicPotential",
    "RectangularBarrier",
    "CircularObstacle",
    "Waveguide",
    "TDSEPotentialFactory",
    # Step functions
    "laplacian_dirichlet",
    "hamiltonian_apply",
    "banded_hamiltonian",
    "apply_tridiagonal",
    "step_ftcs",
    "step_backward_euler",
    "step_crank_nicolson",
    "step_rk4",
    "step_split_step_fft",
    "step_split_step_fft_2d",
    "step_adi_2d",
    # Solve dispatchers
    "solve",
    "solve_2d",
    # Solver classes
    "Solver1D",
    "FTCSSolver",
    "BackwardEulerSolver",
    "CrankNicolsonSolver",
    "RKK4Solver",
    "SplitStepFFTSolver",
    "TDSESolverFactory",
    # Visualization
    "TDSEVisualizer",
    "save_wavepacket_animation",
    "save_tunneling_animation",
    "save_2d_animation",
    # Experiments
    "experiments",
]
