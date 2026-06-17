"""
GIF Animation generation for TDSE time evolution simulations.
Uses PillowWriter (no FFMpeg required).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
import os
from typing import Dict, List, Optional


class AnimationGenerator:
    """Generate GIF animations of TDSE wavefunction evolution."""

    def __init__(self, output_dir: str = 'outputs/animations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _save_animation(self, anim, filename: str, fps: int = 10) -> str:
        """Save animation as GIF file."""
        filepath = os.path.join(self.output_dir, filename)
        writer = PillowWriter(fps=fps)
        anim.save(filepath, writer=writer, dpi=100)
        print(f"  Saved animation: {filepath}")
        plt.close(anim._fig)
        return filepath

    def create_evolution_animation(self, x: np.ndarray,
                                    psi_evolution: List[np.ndarray],
                                    psi_analytical_evolution: Optional[List[np.ndarray]] = None,
                                    times: Optional[List[float]] = None,
                                    V: Optional[np.ndarray] = None,
                                    title: str = 'TDSE Evolution',
                                    filename: str = 'evolution.gif',
                                    fps: int = 10,
                                    n_frames: int = None) -> str:
        """
        Create animation of wavefunction time evolution.

        Three-panel layout:
        - Top-left: |psi|^2 (probability density), numerical + analytical
        - Top-right: Real and imaginary parts
        - Bottom: Error/difference plot

        Args:
            x: Spatial grid
            psi_evolution: List of wavefunction arrays at each time step
            psi_analytical_evolution: Optional list of analytical solutions
            times: Corresponding time values
            V: Potential (for background shading)
            title: Figure title
            filename: Output filename (.gif)
            fps: Frames per second
            n_frames: Max number of frames (subsample if needed)
        """
        if n_frames and len(psi_evolution) > n_frames:
            step = len(psi_evolution) // n_frames
            indices = list(range(0, len(psi_evolution), step))[:n_frames]
        else:
            indices = list(range(len(psi_evolution)))

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1])

        ax1 = fig.add_subplot(gs[0, 0])  # Probability density
        ax2 = fig.add_subplot(gs[0, 1])  # Real/Imaginary parts
        ax3 = fig.add_subplot(gs[1, :])   # Error

        # Setup axes
        x_min, x_max = x.min(), x.max()

        # Panel 1: Probability density
        line_num, = ax1.plot(x, np.abs(psi_evolution[0])**2, 'b-', lw=2,
                             label='Numerical')
        line_ana = None
        if psi_analytical_evolution:
            line_ana, = ax1.plot(x, np.abs(psi_analytical_evolution[0])**2,
                                 'r--', lw=1.5, label='Analytical', alpha=0.7)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(0, max(np.max(np.abs(psi_evolution[0])**2) * 1.3, 0.1))
        ax1.set_xlabel('x')
        ax1.set_ylabel('|ψ|²')
        ax1.set_title('Probability Density')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Shade potential in background
        if V is not None:
            V_normalized = V / (np.max(np.abs(V)) + 1e-10)
            V_scaled = V_normalized * ax1.get_ylim()[1] * 0.3
            ax1.fill_between(x, 0, V_scaled, alpha=0.15, color='gray')
            ax1.text(0.02, 0.95, 'V(x) shaded', transform=ax1.transAxes,
                    fontsize=8, va='top', color='gray')

        # Panel 2: Real and Imaginary parts
        line_real, = ax2.plot(x, np.real(psi_evolution[0]), 'b-', lw=1.5,
                              label='Re(ψ)')
        line_imag, = ax2.plot(x, np.imag(psi_evolution[0]), 'r-', lw=1.5,
                              label='Im(ψ)')
        ax2.set_xlim(x_min, x_max)
        y_max = max(np.max(np.abs(psi_evolution[0])) * 1.3, 0.3)
        ax2.set_ylim(-y_max, y_max)
        ax2.set_xlabel('x')
        ax2.set_ylabel('ψ')
        ax2.set_title('Wavefunction Components')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Panel 3: Error
        err_data = np.zeros_like(x)
        line_err, = ax3.plot(x, err_data, 'g-', lw=1.5)
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(0, 0.1)
        ax3.set_xlabel('x')
        ax3.set_ylabel('|ψ_num - ψ_ana|')
        ax3.set_title('Pointwise Error')
        ax3.grid(True, alpha=0.3)

        # Time annotation
        time_text = fig.suptitle(f'{title}  |  t = 0.000', fontsize=14)

        def update(frame_idx):
            idx = indices[frame_idx]
            psi_n = psi_evolution[idx]

            # Update probability density
            prob = np.abs(psi_n)**2
            line_num.set_ydata(prob)

            # Update analytical if available
            if psi_analytical_evolution and line_ana is not None:
                psi_a = psi_analytical_evolution[idx]
                line_ana.set_ydata(np.abs(psi_a)**2)

                # Update error
                err = np.abs(psi_n - psi_a)
                line_err.set_ydata(err)

                # Auto-scale error axis
                err_max = max(np.max(err) * 1.2, 1e-6)
                ax3.set_ylim(0, err_max)

            # Update real/imag parts
            line_real.set_ydata(np.real(psi_n))
            line_imag.set_ydata(np.imag(psi_n))

            # Update time
            if times:
                t_val = times[idx]
            else:
                t_val = idx
            time_text.set_text(f'{title}  |  t = {t_val:.3f}')

            return line_num, line_real, line_imag, line_err, time_text

        anim = FuncAnimation(fig, update, frames=len(indices),
                             interval=1000//fps, blit=False)

        return self._save_animation(anim, filename, fps=fps)

    def create_method_comparison_animation(self, x: np.ndarray,
                                            method_psi_dict: Dict[str, List[np.ndarray]],
                                            psi_analytical: List[np.ndarray],
                                            times: List[float],
                                            title: str = 'Method Comparison',
                                            filename: str = 'method_comparison.gif',
                                            fps: int = 8) -> str:
        """
        Create animation comparing multiple methods simultaneously.

        Shows |psi|^2 for each method plus analytical solution on one plot.
        """
        n_methods = len(method_psi_dict)
        fig, axes = plt.subplots(n_methods, 1, figsize=(14, 4*n_methods+2),
                                  sharex=True)
        if n_methods == 1:
            axes = [axes]

        x_min, x_max = x.min(), x.max()
        lines = {}
        lines_ana = {}

        colors = ['blue', 'red', 'green']

        for i, (method_name, psi_list) in enumerate(method_psi_dict.items()):
            color = colors[i % len(colors)]
            lines[method_name], = axes[i].plot(
                x, np.abs(psi_list[0])**2, '-', color=color, lw=2,
                label=method_name)
            lines_ana[method_name], = axes[i].plot(
                x, np.abs(psi_analytical[0])**2, '--', color='black', lw=1,
                label='Analytical', alpha=0.6)

            axes[i].set_xlim(x_min, x_max)
            axes[i].set_ylim(0, max(np.max(np.abs(psi_list[0])**2)*1.3, 0.05))
            axes[i].set_ylabel('|ψ|²')
            axes[i].set_title(method_name)
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)

        axes[-1].set_xlabel('x')
        time_text = fig.suptitle(f'{title}  |  t = 0.000', fontsize=14)

        n_frames = min(len(times), 80)
        frame_step = max(1, len(times) // n_frames)
        frame_indices = list(range(0, len(times), frame_step))[:n_frames]

        def update(frame_idx):
            idx = frame_indices[frame_idx]
            for method_name, psi_list in method_psi_dict.items():
                lines[method_name].set_ydata(np.abs(psi_list[idx])**2)
                lines_ana[method_name].set_ydata(np.abs(psi_analytical[idx])**2)
            time_text.set_text(f'{title}  |  t = {times[idx]:.3f}')
            return list(lines.values()) + list(lines_ana.values()) + [time_text]

        anim = FuncAnimation(fig, update, frames=len(frame_indices),
                             interval=1000//fps, blit=False)

        return self._save_animation(anim, filename, fps=fps)

    def create_probability_density_animation(self, x: np.ndarray,
                                              psi_evolution: List[np.ndarray],
                                              times: List[float],
                                              title: str = '',
                                              filename: str = 'density.gif',
                                              fps: int = 12) -> str:
        """
        Simple animation focusing only on probability density evolution.
        Good for quick visualization of wavepacket dynamics.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        x_min, x_max = x.min(), x.max()

        line, = ax.plot(x, np.abs(psi_evolution[0])**2, 'b-', lw=2)
        fill = ax.fill_between(x, 0, np.abs(psi_evolution[0])**2,
                               alpha=0.3, color='blue')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, np.max(np.abs(psi_evolution[0])**2) * 1.3)
        ax.set_xlabel('x')
        ax.set_ylabel('|ψ|²')
        ax.grid(True, alpha=0.3)

        title_text = ax.set_title(f'{title}  |  t = {times[0]:.3f}' if title else f't = {times[0]:.3f}')

        n_frames = min(len(psi_evolution), 120)
        step = max(1, len(psi_evolution) // n_frames)

        fill_container = {'fill': fill}

        def update(frame):
            idx = frame * step
            if idx >= len(psi_evolution):
                idx = len(psi_evolution) - 1

            prob = np.abs(psi_evolution[idx])**2
            line.set_ydata(prob)

            # Update fill
            fill_container['fill'].remove()
            fill_container['fill'] = ax.fill_between(x, 0, prob, alpha=0.3, color='blue')

            # Dynamic y-axis scaling
            ymax = max(np.max(prob) * 1.2, 0.01)
            ax.set_ylim(0, ymax)

            if times:
                title_text.set_text(f'{title}  |  t = {times[idx]:.3f}' if title else f't = {times[idx]:.3f}')

            return line, fill_container['fill'], title_text

        n_update_frames = len(psi_evolution) // step
        anim = FuncAnimation(fig, update, frames=n_update_frames,
                             interval=1000//fps, blit=False)

        return self._save_animation(anim, filename, fps=fps)
