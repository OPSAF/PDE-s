"""
Publication-quality figure generation for TDSE numerical simulations.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import os
import warnings


class FigureGenerator:
    """Generate publication-quality figures for TDSE simulation results."""

    # Publication-quality style settings
    STYLE = {
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white',
        # Disable constrained_layout globally to prevent inf-crash
        'figure.constrained_layout.use': False,
    }

    def __init__(self, output_dir: str = 'outputs/figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Apply style
        plt.rcParams.update(self.STYLE)

    def _savefig(self, fig, filename: str, **kwargs):
        """Save figure to output directory."""
        filepath = os.path.join(self.output_dir, filename)
        # Force all axes to have finite limits (prevents inf-crash in tick formatter)
        for ax in fig.get_axes():
            try:
                # Check and fix x limits
                xlim = ax.get_xlim()
                if not (np.isfinite(xlim[0]) and np.isfinite(xlim[1])):
                    ax.set_xlim(-1, 1)
                else:
                    # Clamp to finite range
                    ax.set_xlim(max(xlim[0], -1e15), min(xlim[1], 1e15))
                # Check and fix y limits
                ylim = ax.get_ylim()
                if not (np.isfinite(ylim[0]) and np.isfinite(ylim[1])):
                    ax.set_ylim(-1, 1)
                else:
                    ax.set_ylim(max(ylim[0], -1e15), min(ylim[1], 1e15))
                # Use ScalarFormatter to avoid log10(inf) crash in tick formatter
                from matplotlib.ticker import ScalarFormatter
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.set_major_formatter(ScalarFormatter())
            except Exception:
                pass
        # Try normal save first; fall back if inf-ticks crash
        try:
            fig.savefig(filepath, bbox_inches='tight', **kwargs)
        except (OverflowError, ValueError, ZeroDivisionError):
            try:
                fig.savefig(filepath, **kwargs)
            except Exception:
                # Last resort: create a blank figure with just title text
                fig2 = plt.figure(figsize=(10, 6))
                fig2.text(0.5, 0.5, f'[Plot skipped: numerical overflow]\n{filename}',
                         ha='center', va='center', fontsize=12)
                fig2.savefig(filepath, **kwargs)
                plt.close(fig2)
        plt.close(fig)
        print(f"  Saved: {filepath}")
        return filepath

    def _finalize_fig(self, fig):
        """Apply safe layout adjustment - never use constrained_layout."""
        fig.subplots_adjust(top=0.92, wspace=0.3, hspace=0.35)

    def plot_wavefunction_comparison(self, x: np.ndarray,
                                      psi_numerical: np.ndarray,
                                      psi_analytical: np.ndarray,
                                      t: float,
                                      title: str = '',
                                      filename: str = 'wavefunction_comparison.png') -> str:
        """
        Plot numerical vs analytical wavefunction comparison.

        Two-panel figure:
        - Left: |ψ|² for both solutions
        - Right: |ψ_num - ψ_anal| (absolute difference)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        prob_num = np.nan_to_num(np.abs(psi_numerical)**2, nan=0.0, posinf=0.0, neginf=0.0)
        prob_anal = np.nan_to_num(np.abs(psi_analytical)**2, nan=0.0, posinf=0.0, neginf=0.0)

        ax1.plot(x, prob_num, 'b-', lw=2, label='Numerical', alpha=0.8)
        ax1.plot(x, prob_anal, 'r--', lw=2, label='Analytical', alpha=0.8)
        ax1.set_xlabel('x')
        ax1.set_ylabel('|ψ|²')
        ax1.set_title(f'{title}\nProbability Density at t = {t:.3f}')
        ax1.legend()

        diff = np.abs(prob_num - prob_anal)
        ax2.fill_between(x, diff, color='green', alpha=0.5)
        ax2.plot(x, diff, 'g-', lw=1.5)
        ax2.set_xlabel('x')
        ax2.set_ylabel('| |ψ_num|² - |ψ_anal|² |')
        ax2.set_title(f'Absolute Difference (max={np.nan_to_num(diff.max(), nan=0.0, posinf=0.0):.2e})')

        fig.suptitle(title, fontsize=14)

        self._finalize_fig(fig)
        return self._savefig(fig, filename)

    def plot_error_evolution(self, times: List[float],
                              l2_errors: List[float],
                              max_errors: List[float],
                              prob_errors: List[float],
                              title: str = '',
                              filename: str = 'error_evolution.png') -> str:
        """
        Plot error evolution over time.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Clean inf/nan for safe plotting
        t_clean = np.array(times, dtype=float)
        l2_clean = np.nan_to_num(np.array(l2_errors, dtype=float), nan=0.0, posinf=1e15, neginf=0.0)
        max_clean = np.nan_to_num(np.array(max_errors, dtype=float), nan=0.0, posinf=1e15, neginf=0.0)
        prob_clean = np.nan_to_num(np.array(prob_errors, dtype=float), nan=0.0, posinf=1e15, neginf=0.0)

        ax1.semilogy(t_clean, l2_clean, 'b-o', lw=2, markersize=3, label='L2 Error')
        ax1.semilogy(t_clean, max_clean, 'r-s', lw=2, markersize=3, label='Max Error')
        ax1.set_xlabel('Time t')
        ax1.set_ylabel('Error')
        ax1.set_title(f'{title} - Error vs Time')
        ax1.legend()
        ax1.grid(True, which='both', alpha=0.3)

        ax2.plot(t_clean, prob_clean, 'g-^', lw=2, markersize=3)
        ax2.set_xlabel('Time t')
        ax2.set_ylabel('|∫|ψ|²dx - 1|')
        ax2.set_title('Probability Conservation Error')
        ax2.grid(True, alpha=0.3)

        self._finalize_fig(fig)
        return self._savefig(fig, filename)

    def plot_expectation_values(self, times: List[float],
                                 x_exp_num: List[float], x_exp_ana: List[float],
                                 p_exp_num: List[float], p_exp_ana: List[float],
                                 title: str = '',
                                 filename: str = 'expectation_values.png') -> str:
        """
        Plot <x> and <p> expectation values over time.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Clean data
        t_clean = np.array(times, dtype=float)
        xn = np.nan_to_num(np.array(x_exp_num, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        pn = np.nan_to_num(np.array(p_exp_num, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        ax1.plot(t_clean, xn, 'b-', lw=2, label='Numerical <x>')
        if x_exp_ana:
            xa = np.nan_to_num(np.array(x_exp_ana, dtype=float), nan=0.0)
            ax1.plot(t_clean, xa, 'r--', lw=2, label='Analytical <x>')
        ax1.set_xlabel('Time t')
        ax1.set_ylabel('<x>')
        ax1.set_title(f'{title} - Position Expectation')
        ax1.legend()

        ax2.plot(t_clean, pn, 'b-', lw=2, label='Numerical <p>')
        if p_exp_ana:
            pa = np.nan_to_num(np.array(p_exp_ana, dtype=float), nan=0.0)
            ax2.plot(t_clean, pa, 'r--', lw=2, label='Analytical <p>')
        ax2.set_xlabel('Time t')
        ax2.set_ylabel('<p>')
        ax2.set_title('Momentum Expectation')
        ax2.legend()

        self._finalize_fig(fig)
        return self._savefig(fig, filename)

    def plot_convergence(self, result, convergence_type: str = 'spatial',
                         title: str = '', filename: str = 'convergence.png') -> str:
        """
        Plot convergence study results (log-log plot).

        Args:
            result: ConvergenceResult object
            convergence_type: 'spatial' or 'temporal'
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        param_vals = result.parameter_values
        errors = result.errors

        if convergence_type == 'spatial':
            # Plot vs 1/Nx (mesh size)
            h_vals = [1.0/p for p in param_vals]
            ax.loglog(h_vals, errors, 'bo-', lw=2, markersize=8, label='Measured error')

            # Reference lines
            if len(errors) >= 2 and result.observed_order:
                order = result.observed_order
                h_ref = np.array(h_vals)
                e_ref = errors[-1] * (h_ref / h_vals[-1])**order
                ax.loglog(h_ref, e_ref, 'k--', lw=1.5,
                         label=f'Order {order:.1f} reference')

            ax.set_xlabel('Grid spacing h = Δx = L/(N-1)')
            ax.set_title(f'{title} - Spatial Convergence\n(Order ≈ {result.observed_order:.2f})')
        else:
            ax.loglog(param_vals, errors, 'ro-', lw=2, markersize=8, label='Measured error')

            if len(errors) >= 2 and result.observed_order:
                order = result.observed_order
                dt_ref = np.array(param_vals)
                e_ref = errors[-1] * (dt_ref / param_vals[-1])**order
                ax.loglog(dt_ref, e_ref, 'k--', lw=1.5,
                         label=f'Order {order:.1f} reference')

            ax.set_xlabel('Time step dt')
            ax.set_title(f'{title} - Temporal Convergence\n(Order ≈ {result.observed_order:.2f})')

        ax.set_ylabel('L2 Error')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

        self._finalize_fig(fig)
        return self._savefig(fig, filename)

    def plot_method_comparison(self, method_results: Dict[str, Dict],
                                metric: str = 'l2_error',
                                title: str = '',
                                filename: str = 'method_comparison.png') -> str:
        """
        Compare multiple methods side by side.

        method_results: dict of {method_name: {'times': [...], 'errors': [...]}}
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = ['blue', 'red', 'green', 'orange', 'purple']
        linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

        for idx, (method_name, data) in enumerate(method_results.items()):
            style = dict(color=colors[idx % len(colors)],
                        linestyle=linestyles[idx % len(linestyles)],
                        lw=2, marker='o', markersize=3)

            errs = np.nan_to_num(np.array(data.get(metric, data.get('l2_errors', [])), dtype=float),
                                  nan=0.0, posinf=1e10, neginf=0.0)
            axes[0].semilogy(data['times'], errs,
                           label=method_name, **style)

            if 'prob_errors' in data:
                prob_errs = np.nan_to_num(np.array(data['prob_errors'], dtype=float),
                                           nan=0.0, posinf=1e10, neginf=0.0)
                axes[1].plot(data['times'], prob_errs,
                           label=method_name, **style)

        axes[0].set_xlabel('Time t')
        axes[0].set_ylabel(f'{metric.replace("_", " ").title()}')
        axes[0].set_title(f'{title} - Method Comparison')
        axes[0].legend()
        axes[0].grid(True, which='both', alpha=0.3)

        axes[1].set_xlabel('Time t')
        axes[1].set_ylabel('Probability Conservation Error')
        axes[1].set_title('Stability Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        self._finalize_fig(fig)
        return self._savefig(fig, filename)

    def plot_stability_domain(self, nx_range: list, dt_range: list,
                               stability_data: Dict[str, np.ndarray],
                               title: str = '',
                               filename: str = 'stability_domain.png') -> str:
        """
        Create stability diagram showing stable/unstable regions for each method.

        stability_data: {method_name: 2D array of bool or error values}
        """
        fig, axes = plt.subplots(1, len(stability_data),
                                  figsize=(5*len(stability_data), 5),
                                  squeeze=False)
        axes = axes.flatten()

        for idx, (method_name, data) in enumerate(stability_data.items()):
            im = axes[idx].imshow(data.T, origin='lower',
                                   aspect='auto',
                                   extent=[nx_range[0], nx_range[-1],
                                          dt_range[0], dt_range[-1]],
                                   cmap='RdYlGn_r')
            axes[idx].set_xlabel('Nx')
            axes[idx].set_ylabel('dt')
            axes[idx].set_title(f'{method_name} Stability')
            plt.colorbar(im, ax=axes[idx], label='Error/Status')

        fig.suptitle(title, fontsize=14)
        self._finalize_fig(fig)
        return self._savefig(fig, filename)

    def plot_phase_space(self, times: List[float],
                          x_exp: List[float], p_exp: List[float],
                          title: str = '',
                          filename: str = 'phase_space.png') -> str:
        """
        Plot phase space trajectory (<x>, <p>) with time coloring.
        """
        # Protect against inf/nan from unstable methods like FTCS
        x_clean = np.nan_to_num(np.array(x_exp, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        p_clean = np.nan_to_num(np.array(p_exp, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        t_clean = np.array(times, dtype=float)

        fig, ax = plt.subplots(figsize=(8, 7))

        # Color by time
        points = np.array([x_clean, p_clean]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        from matplotlib.collections import LineCollection
        norm = plt.Normalize(min(t_clean), max(t_clean))
        lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2)
        lc.set_array(t_clean[:-1])
        ax.add_collection(lc)

        # Mark start and end
        ax.plot(x_clean[0], p_clean[0], 'go', ms=12, label=f'Start (t={t_clean[0]:.2f})', zorder=5)
        ax.plot(x_clean[-1], p_clean[-1], 'rs', ms=12, label=f'End (t={t_clean[-1]:.2f})', zorder=5)

        ax.set_xlabel('<x>')
        ax.set_ylabel('<p>')
        ax.set_title(f'{title} - Phase Space Trajectory')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

        cbar = fig.colorbar(lc, ax=ax)
        cbar.set_label('Time t')

        self._finalize_fig(fig)
        return self._savefig(fig, filename)

    def create_summary_figure(self, model_name: str, all_results: Dict) -> str:
        """
        Placeholder summary figure.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.close(fig)
        return ''
