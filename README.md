# TDSE - Time-Dependent Schrodinger Equation Package

A modular Python package for numerical solutions of the time-dependent Schrodinger equation (TDSE) in 1D and 2D.

## Project Structure

```
xde/
├── tdse/                      # Main package
│   ├── __init__.py           # Package initialization
│   ├── config.py              # Configuration classes
│   ├── potentials.py          # Potentials and wave functions
│   ├── solvers.py             # Numerical solvers
│   └── visualization.py       # Visualization tools
├── demo2.py                   # Full experiment suite
├── main.py                   # Main entry point
├── example.py                # Usage examples
└── README.md                 # This file
```

## Installation

No installation required! Just ensure all files are in the same directory.

## Quick Start

### Basic Usage

```python
import tdse

config = tdse.TDSEConfig(quick=True)
x, dx = tdse.grid(-30.0, 30.0, 384)
psi0 = tdse.gaussian_wavepacket(x, -8.0, 1.2, 2.0, dx)
v = tdse.potential_free(x)

saved_t, psi_hist = tdse.solve("Crank-Nicolson", psi0, v, x, t, dx, dt)
```

### Using Factories

```python
# Create potentials
factory = tdse.TDSEPotentialFactory()
potential = factory.create("barrier", v0=1.0, a=-0.5, b=0.5)

# Create solvers
solver_factory = tdse.TDSESolverFactory()
solver = solver_factory.create("CN")
```

### Visualization

```python
viz = tdse.TDSEVisualizer(config)
viz.plot_wavepacket(x, psi_hist[-1], title="Final State")
```

## Running Experiments

### Full Experiment Suite

```bash
python demo2.py --quick --no-gif
```

### Using main.py

```bash
python main.py --quick                # Quick mode
python main.py --no-gif             # Skip animations
python main.py --outdir OUTPUT      # Custom output
python main.py --1d-only            # Only 1D experiments
python main.py --2d-only            # Only 2D experiments
python main.py --analysis-only       # Only analysis
```

## Package Modules

### 1. config - Configuration
- `TDSEConfig`: Main configuration class
- `RunConfig`: Legacy compatibility
- `ensure_outdir()`: Create output directory
- `setup_plot_style()`: Configure matplotlib

### 2. potentials - Potentials and Wave Functions
**Base Classes:**
- `Potential1D`, `Potential2D`

**Concrete Classes:**
- `FreePotential`
- `HarmonicPotential`
- `RectangularBarrier`
- `CircularObstacle`
- `Waveguide`

**Factory:**
- `TDSEPotentialFactory`

**Functions:**
- `grid()`: Create spatial grid
- `gaussian_wavepacket()`: Initial wave function
- `exact_free_gaussian()`: Analytical solution
- `compute_transmission_reflection()`: T/R coefficients
- `analyze_barrier_scattering()`: Detailed analysis

### 3. solvers - Numerical Methods
**Base Classes:**
- `Solver1D`

**Concrete Classes:**
- `FTCSSolver`
- `BackwardEulerSolver`
- `CrankNicolsonSolver`
- `RKK4Solver`
- `SplitStepFFTSolver`

**Factory:**
- `TDSESolverFactory`

**Functions:**
- `solve()`: 1D time evolution
- `solve_2d()`: 2D time evolution

### 4. visualization - Plotting and Animation
- `TDSEVisualizer`: Unified visualization
- `save_wavepacket_animation()`: 1D animation
- `save_tunneling_animation()`: Tunneling animation
- `save_2d_animation()`: 2D animation

## Features

### 1D Capabilities
- Multiple numerical methods (FTCS, CN, RK4, Split-Step FFT)
- Convergence and stability analysis
- Tunneling simulations
- Mass conservation verification
- Transmission/reflection calculations

### 2D Capabilities
- Split-Step FFT for 2D
- Free Gaussian propagation
- Circular obstacle scattering
- Waveguide confinement
- Absorbing boundary layers
- Grid convergence studies

### Analysis Tools
- L1, L2, Linf error norms
- Convergence order estimation
- Stability analysis
- Runtime comparison
- Parameter sweeps

## Examples

See `example.py` for detailed usage examples:
- Basic 1D propagation
- Barrier scattering analysis
- Factory pattern usage
- 2D simulations

## Output Files

- `figure*.png`: Visualization plots
- `figure*.gif`: Animations
- `*.csv`: Numerical data
- Convergence orders, errors, performance metrics

## Requirements

- numpy
- scipy
- matplotlib
- pandas
- tqdm

## Version

1.0.0

## Author

TDSE Development Team
