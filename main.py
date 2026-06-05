"""
TDSE Main Entry Point
====================

This is the main entry point for running TDSE experiments.
All functionality is provided by the modular tdse package.

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

    # Selective execution
    python main.py --1d-only       # Only 1D experiments
    python main.py --2d-only       # Only 2D experiments
    python main.py --analysis-only # Only analysis experiments
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

from tdse.config import RunConfig, ensure_outdir, setup_plot_style
from tdse.potentials import print_section
from tdse.experiments import (
    experiment_analytic_vs_numerical,
    experiment_convergence,
    experiment_stability,
    experiment_performance,
    experiment_method_comparison,
    experiment_tunneling,
    experiment_2d_free_propagation,
    experiment_2d_circular_obstacle_with_animation,
    experiment_2d_waveguide,
    experiment_circular_obstacle_radius_sweep,
    experiment_waveguide_strength_sweep,
    experiment_1d_conservation_analysis,
    experiment_runtime_comparison,
    experiment_2d_convergence,
    experiment_2d_error_heatmap,
    save_wavepacket_animation_experiment,
)


def main() -> None:
    """Main entry point for TDSE experiment suite."""
    parser = argparse.ArgumentParser(
        description="TDSE numerical PDE solver — 1D and 2D time-dependent "
                    "Schrödinger equation simulations"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with reduced grid sizes for faster execution"
    )
    parser.add_argument(
        "--no-gif", action="store_true",
        help="Skip GIF animation generation"
    )
    parser.add_argument(
        "--outdir", default="tdse_outputs",
        help="Output directory for results (default: tdse_outputs)"
    )
    parser.add_argument(
        "--1d-only", dest="run_1d", action="store_true",
        help="Run only 1D experiments"
    )
    parser.add_argument(
        "--2d-only", dest="run_2d", action="store_true",
        help="Run only 2D experiments"
    )
    parser.add_argument(
        "--analysis-only", dest="run_analysis", action="store_true",
        help="Run only analysis experiments (parameter sweeps, convergence, etc.)"
    )

    args = parser.parse_args()

    # ---- Configuration ----
    cfg = RunConfig(
        outdir=args.outdir,
        quick=args.quick,
        save_gif=not args.no_gif
    )

    ensure_outdir(cfg.outdir)
    warnings.filterwarnings("ignore", category=UserWarning)
    setup_plot_style(cfg.dpi)

    print_section("TDSE -- Time-Dependent Schrodinger Equation Solver")
    print(f"Version: 1.1.0")
    print(f"Output directory: {os.path.abspath(cfg.outdir)}")
    print(f"Quick mode: {cfg.quick}")
    print(f"Save GIFs:  {cfg.save_gif}")
    print()

    total_start = time.perf_counter()

    # ---- Dispatch experiments ----
    if args.run_analysis:
        # Analysis-only experiments
        experiment_circular_obstacle_radius_sweep(cfg)
        experiment_waveguide_strength_sweep(cfg)
        experiment_1d_conservation_analysis(cfg)
        experiment_runtime_comparison(cfg)
        experiment_2d_convergence(cfg)
        experiment_2d_error_heatmap(cfg)

    elif args.run_1d:
        # 1D-only experiments
        experiment_analytic_vs_numerical(cfg)
        experiment_convergence(cfg)
        experiment_stability(cfg)
        experiment_performance(cfg)
        experiment_method_comparison(cfg)
        if cfg.save_gif:
            save_wavepacket_animation_experiment(cfg)
        experiment_tunneling(cfg)

    elif args.run_2d:
        # 2D-only experiments
        experiment_2d_free_propagation(cfg)
        experiment_2d_circular_obstacle_with_animation(cfg)
        experiment_2d_waveguide(cfg)

    else:
        # ---- Full experiment suite ----
        # 1D experiments
        experiment_analytic_vs_numerical(cfg)
        experiment_convergence(cfg)
        experiment_stability(cfg)
        experiment_performance(cfg)
        experiment_method_comparison(cfg)

        if cfg.save_gif:
            save_wavepacket_animation_experiment(cfg)

        experiment_tunneling(cfg)

        # 2D experiments
        experiment_2d_free_propagation(cfg)
        experiment_2d_circular_obstacle_with_animation(cfg)
        experiment_2d_waveguide(cfg)

        # Analysis experiments
        experiment_circular_obstacle_radius_sweep(cfg)
        experiment_waveguide_strength_sweep(cfg)
        experiment_1d_conservation_analysis(cfg)
        experiment_runtime_comparison(cfg)
        experiment_2d_convergence(cfg)
        experiment_2d_error_heatmap(cfg)

    # ---- Execution summary ----
    total_runtime = time.perf_counter() - total_start
    print_section("Execution Summary")
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"\nGenerated files in {cfg.outdir}/:")
    for name in sorted(os.listdir(cfg.outdir)):
        fpath = os.path.join(cfg.outdir, name)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  {name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
