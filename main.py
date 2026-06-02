"""
TDSE Main Entry Point
====================

This is the main entry point for running TDSE experiments.

Usage:
------
    python main.py                 # Run all experiments
    python main.py --quick        # Quick mode with smaller grids
    python main.py --no-gif       # Skip GIF generation
    python main.py --outdir OUTPUT # Custom output directory

Examples:
--------
    # Run full experiment suite
    python main.py

    # Quick test run
    python main.py --quick

    # Generate plots only (no animations)
    python main.py --no-gif

    # Custom output directory
    python main.py --outdir my_results
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
import sys


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TDSE numerical PDE solver - 1D and 2D simulations"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with reduced grid sizes for faster execution"
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip GIF animation generation"
    )
    parser.add_argument(
        "--outdir",
        default="tdse_outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--1d-only",
        dest="run_1d",
        action="store_true",
        help="Run only 1D experiments"
    )
    parser.add_argument(
        "--2d-only",
        dest="run_2d",
        action="store_true",
        help="Run only 2D experiments"
    )
    parser.add_argument(
        "--analysis-only",
        dest="run_analysis",
        action="store_true",
        help="Run only analysis experiments"
    )

    args = parser.parse_args()

    # Import configuration
    from tdse.config import RunConfig, ensure_outdir, setup_plot_style
    from tdse.potentials import print_section

    cfg = RunConfig(
        outdir=args.outdir,
        quick=args.quick,
        save_gif=not args.no_gif
    )

    ensure_outdir(cfg.outdir)
    warnings.filterwarnings("ignore", category=UserWarning)
    setup_plot_style(cfg.dpi)

    print_section("TDSE - Time-Dependent Schrodinger Equation Solver")
    print(f"Version: 1.0.0")
    print(f"Output directory: {os.path.abspath(cfg.outdir)}")
    print(f"Quick mode: {cfg.quick}")
    print(f"Save GIFs: {cfg.save_gif}")

    total_start = time.perf_counter()

    # Import demo2 module
    try:
        import demo2
    except ImportError:
        print("Error: demo2.py not found in the current directory.")
        print("Please ensure demo2.py is in the same directory as main.py")
        sys.exit(1)

    # Run experiments based on flags
    if args.run_analysis:
        from tdse.potentials import (
            grid, gaussian_wavepacket, potential_circle_2d,
            potential_waveguide_2d, probability_mass, mass_2d,
            exact_free_gaussian_2d, gaussian_wavepacket_2d,
        )
        from tdse.solvers import solve, step_split_step_fft_2d

        demo2.experiment_circular_obstacle_radius_sweep(cfg)
        demo2.experiment_waveguide_strength_sweep(cfg)
        demo2.experiment_1d_conservation_analysis(cfg)
        demo2.experiment_runtime_comparison(cfg)
        demo2.experiment_2d_convergence(cfg)
        demo2.experiment_2d_error_heatmap(cfg)

    elif args.run_1d:
        demo2.experiment_analytic_vs_numerical(cfg)
        demo2.experiment_convergence(cfg)
        demo2.experiment_stability(cfg)
        demo2.experiment_performance(cfg)
        demo2.experiment_method_comparison(cfg)
        if cfg.save_gif:
            demo2.save_wavepacket_animation(cfg)
        demo2.experiment_tunneling(cfg)

    elif args.run_2d:
        demo2.experiment_2d_free_propagation(cfg)
        demo2.experiment_2d_circular_obstacle_with_animation(cfg)
        demo2.experiment_2d_waveguide(cfg)

    else:
        demo2.experiment_analytic_vs_numerical(cfg)
        demo2.experiment_convergence(cfg)
        demo2.experiment_stability(cfg)
        demo2.experiment_performance(cfg)
        demo2.experiment_method_comparison(cfg)

        if cfg.save_gif:
            demo2.save_wavepacket_animation(cfg)

        demo2.experiment_tunneling(cfg)
        demo2.experiment_2d_free_propagation(cfg)
        demo2.experiment_2d_circular_obstacle_with_animation(cfg)
        demo2.experiment_2d_waveguide(cfg)
        demo2.experiment_circular_obstacle_radius_sweep(cfg)
        demo2.experiment_waveguide_strength_sweep(cfg)
        demo2.experiment_1d_conservation_analysis(cfg)
        demo2.experiment_runtime_comparison(cfg)
        demo2.experiment_2d_convergence(cfg)
        demo2.experiment_2d_error_heatmap(cfg)

    total_runtime = time.perf_counter() - total_start

    print_section("Execution Summary")
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"\nGenerated files in {cfg.outdir}/:")
    for name in sorted(os.listdir(cfg.outdir)):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
