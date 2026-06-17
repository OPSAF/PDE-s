"""
TDSE Numerical Simulation Project - Main Orchestrator
=====================================================

Complete 1D Time-Dependent Schrodinger Equation solver with:
- Three quantum models (HO ground state, HO coherent state, constant force)
- Three numerical methods (Crank-Nicolson, Split-Step Fourier, FTCS)
- Comprehensive error analysis and convergence studies
- Publication-quality figures and MP4 animations

Equation: i*hbar*dpsi/dt = -(hbar^2/2m)*d^2psi/dx^2 + V(x)*psi
Units: dimensionless (hbar = m = omega = 1)

Usage:
    python main.py                    # Run all experiments
    python main.py --model ho_ground  # Run specific model
    python main.py --quick            # Quick test run

Author: TDSE Project
"""

from __future__ import annotations

import sys
import os
import time
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import project modules
from models import (
    HarmonicOscillatorGroundState,
    HarmonicOscillatorCoherentState,
    FreeParticle,
)
from solvers import CrankNicolson, SplitStepFourier, FTCS
from analysis import ErrorMetrics, ConvergenceStudy, ExpectationValues
from visualization import FigureGenerator
from animations import AnimationGenerator


# =============================================================================
# Simulation Parameters
# =============================================================================

# Default parameters for each model
MODEL_PARAMS = {
    'ho_ground': {
        'class': HarmonicOscillatorGroundState,
        'x_range': (-12.0, 12.0),
        'nx': 512,
        'dt': 0.01,
        'total_time': 4.0 * np.pi,  # Two full oscillation periods
        'description': 'Harmonic Oscillator Ground State'
    },
    'ho_coherent': {
        'class': HarmonicOscillatorCoherentState,
        'x_range': (-12.0, 12.0),
        'nx': 512,
        'dt': 0.01,
        'total_time': 4.0 * np.pi,  # Two full periods
        'description': 'Harmonic Oscillator Coherent State (x0=2, k0=0)'
    },
    'free_particle': {
        'class': FreeParticle,
        'x_range': (-25.0, 25.0),
        'nx': 512,
        'dt': 0.01,
        'total_time': 5.0,
        'description': 'Free Particle Gaussian Wavepacket (V=0)'
    }
}

# Solver configurations
SOLVERS = {
    'cn': {'class': CrankNicolson, 'name': 'Crank-Nicolson', 'color': 'blue'},
    'ssf': {'class': SplitStepFourier, 'name': 'Split-Step Fourier', 'color': 'red'},
    'ftcs': {'class': FTCS, 'name': 'FTCS', 'color': 'green'},
}

# Convergence study parameters
CONVERGENCE_NX_LIST = [128, 256, 512, 1024]
CONVERGENCE_DT_LIST = [0.01, 0.005, 0.001]


# =============================================================================
# Core Simulation Functions
# =============================================================================

def run_single_simulation(model, solver_class, x: np.ndarray, V: np.ndarray,
                          dt: float, total_time: float,
                          save_interval: int = 10) -> Dict[str, Any]:
    """
    Run a single simulation: model + solver combination.
    
    Returns dict with:
        - times: list of time values where data was saved
        - psi_evolution: list of wavefunction arrays
        - errors: list of ErrorMetrics at each save point
        - expectations: list of ExpectationValues at each save point
        - final_error: final ErrorMetrics
    """
    psi0 = model.initial_wavefunction(x)
    
    # Create solver
    solver = solver_class(x)
    psi = psi0.copy()
    
    num_steps = int(total_time / dt)
    save_every = max(1, num_steps // (num_steps // save_interval + 1))
    
    times = [0.0]
    psi_evolution = [psi.copy()]
    errors = []
    expectations = []
    
    psi_exact = model.analytical_solution(x, 0.0)
    err = ErrorMetrics.compute(psi, psi_exact, x)
    errors.append(err)
    exp_val = ExpectationValues.compute(psi, x, V)
    expectations.append(exp_val)
    
    start_time = time.time()
    
    for step in range(1, num_steps + 1):
        t_current = step * dt
        
        # For time-dependent potentials, update V here
        # (all our models have time-independent V, so this is a no-op)
        
        psi = solver.propagate(psi, V, dt)
        
        if step % save_every == 0 or step == num_steps:
            times.append(t_current)
            psi_evolution.append(psi.copy())
            
            psi_exact = model.analytical_solution(x, t_current)
            err = ErrorMetrics.compute(psi, psi_exact, x)
            errors.append(err)
            
            exp_val = ExpectationValues.compute(psi, x, V)
            expectations.append(exp_val)
    
    elapsed = time.time() - start_time
    
    return {
        'times': times,
        'psi_evolution': psi_evolution,
        'errors': errors,
        'expectations': expectations,
        'final_error': errors[-1],
        'elapsed_time': elapsed,
        'num_steps': num_steps,
    }


def run_model_experiments(model_key: str, output_base: str,
                          quick_mode: bool = False) -> Dict[str, Any]:
    """
    Run all three solvers on a single model and generate outputs.
    
    Args:
        model_key: Key from MODEL_PARAMS ('ho_ground', 'ho_coherent', 'constant_force')
        output_base: Base directory for outputs
        quick_mode: If True, use fewer grid points and shorter time
    
    Returns:
        Dictionary with all results for this model
    """
    params = MODEL_PARAMS[model_key]
    model_class = params['class']
    description = params['description']
    
    # Adjust parameters for quick mode
    nx = 256 if quick_mode else params['nx']
    dt = params['dt'] * 2 if quick_mode else params['dt']
    total_time = min(params['total_time'], 2.0) if quick_mode else params['total_time']
    x_range = params['x_range']
    
    print(f"\n{'='*70}")
    print(f" Model: {description}")
    print(f" Grid: Nx={nx}, x=[{x_range[0]}, {x_range[1]}]")
    print(f" Time: dt={dt}, T={total_time:.2f}, steps={int(total_time/dt)}")
    print(f"{'='*70}\n")
    
    # Create spatial grid
    x = np.linspace(x_range[0], x_range[1], nx)
    model = model_class()
    V = model.potential(x)
    
    # Initialize output generators
    fig_gen = FigureGenerator(os.path.join(output_base, 'figures'))
    anim_gen = AnimationGenerator(os.path.join(output_base, 'animations'))
    
    # Storage for all results
    all_results = {}
    method_comparison_data = {}
    
    # Run each solver
    for solver_key, solver_info in SOLVERS.items():
        print(f"\n--- Running {solver_info['name']} ---")
        
        try:
            result = run_single_simulation(
                model, solver_info['class'], x, V, dt, total_time,
                save_interval=50 if not quick_mode else 20
            )
            
            result['solver_name'] = solver_info['name']
            result['solver_key'] = solver_key
            all_results[solver_key] = result
            
            print(f"  Completed in {result['elapsed_time']:.2f}s")
            print(f"  Final L2 error: {result['final_error'].l2_error:.6e}")
            print(f"  Final Max error: {result['final_error'].max_error:.6e}")
            print(f"  Final Prob error: {result['final_error'].probability_error:.6e}")
            
            # Prepare data for method comparison plot
            method_comparison_data[solver_info['name']] = {
                'times': result['times'],
                'l2_errors': [e.l2_error for e in result['errors']],
                'max_errors': [e.max_error for e in result['errors']],
                'prob_errors': [e.probability_error for e in result['errors']],
            }
            
        except Exception as e:
            print(f"  ERROR with {solver_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # =====================================================================
    # Generate Figures
    # =====================================================================
    print(f"\n--- Generating Figures ---")
    
    # 1. Wavefunction comparison snapshots for each method
    for solver_key, result in all_results.items():
        idx_final = len(result['psi_evolution']) - 1
        t_final = result['times'][idx_final]
        psi_exact = model.analytical_solution(x, t_final)
        
        fig_gen.plot_wavefunction_comparison(
            x, result['psi_evolution'][idx_final], psi_exact,
            t=t_final,
            title=f"{description} - {result['solver_name']}",
            filename=f"{model_key}_{solver_key}_comparison.png"
        )
    
    # 2. Error evolution plots
    for solver_key, result in all_results.items():
        fig_gen.plot_error_evolution(
            result['times'],
            [e.l2_error for e in result['errors']],
            [e.max_error for e in result['errors']],
            [e.probability_error for e in result['errors']],
            title=f"{description} - {result['solver_name']}",
            filename=f"{model_key}_{solver_key}_error_evolution.png"
        )
    
    # 3. Expectation value plots
    for solver_key, result in all_results.items():
        x_exp = [ev.x_exp for ev in result['expectations']]
        p_exp = [ev.p_exp for ev in result['expectations']]
        
        fig_gen.plot_expectation_values(
            result['times'], x_exp, [], p_exp, [],
            title=f"{description} - {result['solver_name']}",
            filename=f"{model_key}_{solver_key}_expectations.png"
        )
    
    # 4. Method comparison plot
    if len(method_comparison_data) > 1:
        fig_gen.plot_method_comparison(
            method_comparison_data,
            title=description,
            filename=f"{model_key}_method_comparison.png"
        )
    
    # 5. Phase space trajectory (use CN result as reference)
    if 'cn' in all_results:
        cn_result = all_results['cn']
        x_exp = [ev.x_exp for ev in cn_result['expectations']]
        p_exp = [ev.p_exp for ev in cn_result['expectations']]
        fig_gen.plot_phase_space(
            cn_result['times'], x_exp, p_exp,
            title=description,
            filename=f"{model_key}_phase_space.png"
        )
    
    # =====================================================================
    # Generate Animations
    # =====================================================================
    print(f"\n--- Generating Animations ---")
    
    # Main evolution animation using best solver (CN or SSF)
    primary_solver = 'cn' if 'cn' in all_results else next(iter(all_results))
    if primary_solver in all_results:
        result = all_results[primary_solver]
        
        # Compute analytical solutions at each time step
        psi_ana_evo = [model.analytical_solution(x, t) for t in result['times']]
        
        anim_gen.create_evolution_animation(
            x, result['psi_evolution'], psi_ana_evo,
            times=result['times'], V=V,
            title=f"{description} ({SOLVERS[primary_solver]['name']})",
            filename=f"{model_key}_evolution.gif",
            fps=15
        )
    
    # Method comparison animation (if multiple methods available)
    if len(all_results) >= 2:
        # Use the solver with most time points as reference
        ref_key = max(all_results.keys(), key=lambda k: len(all_results[k]['times']))
        ref_times = all_results[ref_key]['times']
        psi_ana_evo = [model.analytical_solution(x, t) for t in ref_times]
        
        # Subsample all to same length
        n_frames = min(len(r['psi_evolution']) for r in all_results.values())
        method_psi = {}
        for sk, r in all_results.items():
            step = len(r['psi_evolution']) // n_frames
            method_psi[SOLVERS[sk]['name']] = [
                r['psi_evolution'][i*step] for i in range(n_frames)
            ]
        
        anim_gen.create_method_comparison_animation(
            x, method_psi, 
            [model.analytical_solution(x, t) for t in ref_times[::len(ref_times)//n_frames]],
            ref_times[::len(ref_times)//n_frames],
            title=description,
            filename=f"{model_key}_method_comparison.gif",
            fps=12
        )
    
    return {
        'model_key': model_key,
        'description': description,
        'params': {'nx': nx, 'dt': dt, 'total_time': total_time, 'x_range': x_range},
        'results': all_results,
        'method_comparison_data': method_comparison_data,
    }


def run_convergence_studies(output_base: str, quick_mode: bool = False) -> Dict[str, Any]:
    """
    Run comprehensive spatial and temporal convergence studies.
    
    Tests all model + solver combinations across different grid sizes and time steps.
    """
    print(f"\n{'='*70}")
    print(" CONVERGENCE STUDIES")
    print(f"{'='*70}")
    
    conv_study = ConvergenceStudy(output_dir=os.path.join(output_base, 'figures'))
    fig_gen = FigureGenerator(os.path.join(output_base, 'figures'))
    
    all_convergence = {}
    
    nx_list = [128, 256, 512] if quick_mode else CONVERGENCE_NX_LIST
    dt_list = [0.01, 0.005] if quick_mode else CONVERGENCE_DT_LIST
    
    for model_key, params in MODEL_PARAMS.items():
        model_class = params['class']
        model = model_class()
        x_range = params['x_range']
        
        for solver_key, solver_info in SOLVERS.items():
            # Skip FTCS for convergence (it's unstable)
            if solver_key == 'ftcs':
                continue
            
            label = f"{model_key}_{solver_info['name']}"
            print(f"\n>>> {label}")
            
            try:
                result = conv_study.run_full_study(
                    model, solver_info['class'],
                    label=label,
                    nx_list=nx_list,
                    dt_list=dt_list
                )
                
                all_convergence[label] = result
                
                # Generate convergence plots
                fig_gen.plot_convergence(
                    result['spatial'], 'spatial',
                    title=label, filename=f"{label}_spatial_convergence.png"
                )
                fig_gen.plot_convergence(
                    result['temporal'], 'temporal',
                    title=label, filename=f"{label}_temporal_convergence.png"
                )
                
            except Exception as e:
                print(f"  Convergence study failed for {label}: {e}")
                import traceback
                traceback.print_exc()
    
    return all_convergence


def generate_error_tables(all_model_results: Dict, output_base: str) -> str:
    """
    Generate summary error tables in text format.
    """
    table_path = os.path.join(output_base, 'tables', 'error_summary.txt')
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    
    with open(table_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TDSE NUMERICAL SIMULATION - ERROR SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for model_key, data in all_model_results.items():
            f.write("-" * 80 + "\n")
            f.write(f"Model: {data['description']}\n")
            f.write(f"Parameters: Nx={data['params']['nx']}, "
                   f"dt={data['params']['dt']}, "
                   f"T={data['params']['total_time']}\n")
            f.write("-" * 80 + "\n\n")
            
            f.write(f"{'Solver':<25} {'L2 Error':<15} {'Max Error':<15} "
                  f"{'Prob Error':<15} {'Time(s)':<10}\n")
            f.write("-" * 80 + "\n")
            
            for solver_key, result in data['results'].items():
                err = result['final_error']
                f.write(f"{result['solver_name']:<25} "
                       f"{err.l2_error:<15.6e} "
                       f"{err.max_error:<15.6e} "
                       f"{err.probability_error:<15.6e} "
                       f"{result['elapsed_time']:<10.2f}\n")
            
            f.write("\n")
    
    print(f"\nError table saved to: {table_path}")
    return table_path


def generate_json_report(all_model_results: Dict, convergence_results: Dict,
                         output_base: str) -> str:
    """
    Generate machine-readable JSON report of all results.
    """
    report_path = os.path.join(output_base, 'tables', 'full_report.json')
    
    report = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'equation': 'i*hbar*dpsi/dt = -(hbar^2/2m)*d^2psi/dx^2 + V(x)*psi',
            'units': 'dimensionless (hbar=m=omega=1)',
        },
        'models': {},
        'convergence': {}
    }
    
    for model_key, data in all_model_results.items():
        report['models'][model_key] = {
            'description': data['description'],
            'parameters': data['params'],
            'solvers': {}
        }
        for solver_key, result in data['results'].items():
            report['models'][model_key]['solvers'][solver_key] = {
                'name': result['solver_name'],
                'final_l2_error': result['final_error'].l2_error,
                'final_max_error': result['final_error'].max_error,
                'final_prob_error': result['final_error'].probability_error,
                'elapsed_time': result['elapsed_time'],
                'num_steps': result['num_steps'],
            }
    
    for label, data in convergence_results.items():
        report['convergence'][label] = {
            'spatial_order': data.get('spatial_order'),
            'temporal_order': data.get('temporal_order'),
        }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"JSON report saved to: {report_path}")
    return report_path


# =============================================================================
# Stability Demonstration (FTCS instability)
# =============================================================================

def demonstrate_ftcs_instability(output_base: str):
    """
    Demonstrate FTCS instability compared to stable methods.
    Uses HO coherent state which clearly shows instability growth.
    """
    print(f"\n{'='*70}")
    print(" STABILITY DEMONSTRATION: FTCS Instability")
    print(f"{'='*70}")
    
    fig_gen = FigureGenerator(os.path.join(output_base, 'figures'))
    
    model = HarmonicOscillatorCoherentState()
    x = np.linspace(-12, 12, 256)
    V = model.potential(x)
    psi0 = model.initial_wavefunction(x)
    
    dt = 0.02  # Larger dt to show instability faster
    total_time = 3.0
    
    stability_data = {}
    
    for solver_key, solver_info in SOLVERS.items():
        print(f"\n  Testing {solver_info['name']}...")
        solver = solver_info['class'](x)
        psi = psi0.copy()
        
        times = [0.0]
        prob_norms = [1.0]
        l2_errs = [0.0]
        
        num_steps = int(total_time / dt)
        
        for step in range(num_steps):
            t = (step + 1) * dt
            
            try:
                psi = solver.propagate(psi, V, dt)
                
                # Check for blowup
                if np.any(np.isnan(psi)) or np.any(np.isinf(psi)):
                    print(f"    {solver_info['name']} BLEW UP at t={t:.3f}!")
                    break
                
                psi_exact = model.analytical_solution(x, t)
                dx = x[1] - x[0]
                l2_err = float(np.sqrt(dx * np.sum(np.abs(psi - psi_exact)**2)))
                prob = float(dx * np.sum(np.abs(psi)**2))
                
                times.append(t)
                l2_errs.append(l2_err)
                prob_norms.append(prob)
                
            except Exception as e:
                print(f"    {solver_info['name']} FAILED at t={t:.3f}: {e}")
                break
        
        stability_data[solver_info['name']] = {
            'times': times,
            'l2_errors': l2_errs,
            'probabilities': prob_norms,
        }
    
    # Plot stability comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['blue', 'red', 'green']
    for idx, (method, data) in enumerate(stability_data.items()):
        ax1.semilogy(data['times'], data['l2_errors'], '-',
                    color=colors[idx], lw=2, label=method)
        ax2.plot(data['times'], data['probabilities'], '-',
                color=colors[idx], lw=2, label=method)
    
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('L2 Error')
    ax1.set_title('Stability Comparison: Error Growth')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)
    
    ax2.set_xlabel('Time t')  
    ax2.set_ylabel('∫|ψ|²dx (Probability)')
    ax2.set_title('Stability Comparison: Probability Conservation')
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Unit norm')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = fig_gen._savefig(fig, 'stability_comparison.png')
    print(f"  Saved stability comparison figure.")
    
    return stability_data


# Import matplotlib for stability demo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main function to run all TDSE simulations."""
    parser = argparse.ArgumentParser(
        description='TDSE Numerical Simulation Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                 Run all experiments
  python main.py --model ho_coherent  Run only coherent state
  python main.py --quick         Quick test mode (reduced resolution)
  python main.py --no-anim       Skip animation generation
        """
    )
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'ho_ground', 'ho_coherent', 'free_particle'],
                       help='Which model to simulate')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced resolution')
    parser.add_argument('--no-anim', action='store_true',
                       help='Skip animation generation')
    parser.add_argument('--no-convergence', action='store_true',
                       help='Skip convergence studies')
    parser.add_argument('--output', type=str, default='',
                       help='Custom output directory')
    
    args = parser.parse_args()
    
    # Setup output directories
    output_base = args.output if args.output else os.path.join(PROJECT_ROOT, 'outputs')
    
    print("=" * 70)
    print("  1D TIME-DEPENDENT SCHRODINGER EQUATION NUMERICAL SIMULATOR")
    print("=" * 70)
    print(f"  Equation: i*hbar*dpsi/dt = -(hbar^2/2m)*d^2psi/dx^2 + V(x)*psi")
    print(f"  Units: dimensionless (hbar = m = omega = 1)")
    print(f"  Output: {output_base}")
    print(f"  Quick mode: {args.quick}")
    print("=" * 70)
    
    total_start = time.time()
    all_model_results = {}
    
    # Determine which models to run
    if args.model == 'all':
        models_to_run = list(MODEL_PARAMS.keys())
    else:
        models_to_run = [args.model]
    
    # =====================================================================
    # Run Model Experiments
    # =====================================================================
    for model_key in models_to_run:
        result = run_model_experiments(
            model_key, output_base, quick_mode=args.quick
        )
        all_model_results[model_key] = result
    
    # =====================================================================
    # Convergence Studies
    # =====================================================================
    if not args.no_convergence:
        convergence_results = run_convergence_studies(
            output_base, quick_mode=args.quick
        )
    else:
        convergence_results = {}
    
    # =====================================================================
    # Stability Demonstration
    # =====================================================================
    if not args.quick:
        demonstrate_ftcs_instability(output_base)
    
    # =====================================================================
    # Generate Reports
    # =====================================================================
    print(f"\n{'='*70}")
    print(" GENERATING REPORTS")
    print(f"{'='*70}")
    
    generate_error_tables(all_model_results, output_base)
    generate_json_report(all_model_results, convergence_results, output_base)
    
    # =====================================================================
    # Summary
    # =====================================================================
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(" SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Total runtime: {total_elapsed:.1f} seconds")
    print(f"  Models simulated: {len(all_model_results)}")
    print(f"  Output directory: {output_base}")
    print(f"\n  Outputs:")
    print(f"    - Figures: {os.path.join(output_base, 'figures')}/")
    print(f"    - Tables: {os.path.join(output_base, 'tables')}/")
    if not args.no_anim:
        print(f"    - Animations: {os.path.join(output_base, 'animations')}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
