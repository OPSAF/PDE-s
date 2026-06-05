"""
TDSE Visualization Module
=========================

Visualization and animation tools for TDSE simulations.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from tdse.config import TDSEConfig, RunConfig
from tdse.potentials import Array

# setup_plot_style is canonically defined in tdse.config and re-exported here
# for backward compatibility with code that imports from tdse.visualization.
from tdse.config import setup_plot_style  # noqa: F401


# =============================================================================
# Visualization Class
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
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

        ax.plot(x, np.abs(psi) ** 2, lw=2.5, label="Numerical", color='#2E86AB')

        if show_exact is not None:
            ax.plot(x, np.abs(show_exact) ** 2, "k--", lw=2.0,
                   label="Exact", alpha=0.8)

        ax.set_xlabel(r"Position $x$", labelpad=12, fontsize=12)
        ax.set_ylabel(r"Probability Density $|\psi|^2$", labelpad=12, fontsize=12)
        ax.set_title(title, fontweight='bold', pad=20, fontsize=14)
        ax.legend(fontsize=11, loc='upper right', framealpha=0.95, bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3)
        
        # Adjust tick parameters
        ax.tick_params(axis='both', labelsize=10, pad=8)

        if filename:
            plt.savefig(os.path.join(self.config.outdir, filename),
                       dpi=self.config.dpi, bbox_inches='tight')

        plt.close(fig)

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
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)

        ax.plot(x, np.abs(psi) ** 2, lw=2.5, label=r"$|\psi|^2$", color='#2E86AB')

        if potential is not None:
            vmax = np.max(potential)
            if vmax > 0:
                ax.fill_between(x, 0, potential / vmax * 0.15,
                              color="#C73E1D", alpha=0.4, label="Potential")

        ax.set_xlabel(r"Position $x$", labelpad=12, fontsize=12)
        ax.set_ylabel(r"Probability Density $|\psi|^2$", labelpad=12, fontsize=12)
        ax.set_title(title, fontweight='bold', pad=20, fontsize=14)
        ax.legend(fontsize=11, loc='upper right', framealpha=0.95, bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3)
        
        # Adjust tick parameters
        ax.tick_params(axis='both', labelsize=10, pad=8)

        if filename:
            plt.savefig(os.path.join(self.config.outdir, filename),
                       dpi=self.config.dpi, bbox_inches='tight')

        plt.close(fig)

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

        cbar = plt.colorbar(im, ax=ax, shrink=0.85, label=r"$|\psi|^2$", pad=0.08)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(r"$|\psi|^2$", fontsize=11)

        if obstacle is not None:
            cx, cy, cr = obstacle
            theta = np.linspace(0.0, 2.0 * np.pi, 100)
            ax.plot(cx + cr * np.cos(theta), cy + cr * np.sin(theta),
                   "w--", lw=2.5, label=f"Obstacle R={cr:.2f}")
            ax.legend(fontsize=11, loc='upper right', framealpha=0.95)

        ax.set_xlabel(r"$x$", labelpad=12, fontsize=12)
        ax.set_ylabel(r"$y$", labelpad=12, fontsize=12)
        ax.set_title(title, fontweight='bold', pad=20, fontsize=14)
        
        # Adjust tick parameters
        ax.tick_params(axis='both', labelsize=10, pad=8)

        if filename:
            plt.savefig(os.path.join(self.config.outdir, filename),
                       dpi=self.config.dpi, bbox_inches='tight')

        plt.close(fig)

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
        fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#95C623', '#4A90A4', '#6B4423']

        for idx, (name, psi) in enumerate(data.items()):
            ax.plot(x, np.abs(psi) ** 2, lw=2.5, label=name,
                   color=colors[idx % len(colors)])

        ax.set_xlabel(r"Position $x$", labelpad=12, fontsize=12)
        ax.set_ylabel(r"Probability Density $|\psi|^2$", labelpad=12, fontsize=12)
        ax.set_title(title, fontweight='bold', pad=20, fontsize=14)
        ax.legend(fontsize=11, loc='upper right', framealpha=0.95, bbox_to_anchor=(1.02, 1), ncol=1)
        ax.grid(True, alpha=0.3)
        
        # Adjust tick parameters
        ax.tick_params(axis='both', labelsize=10, pad=8)

        if filename:
            plt.savefig(os.path.join(self.config.outdir, filename),
                       dpi=self.config.dpi, bbox_inches='tight')

        plt.close(fig)


# =============================================================================
# Animation Functions
# =============================================================================

def save_wavepacket_animation(cfg: RunConfig, x: Array, psi_hist: Array, 
                              saved_t: Array, filename: str = "wavepacket_animation.gif") -> None:
    """Save 1D wave packet animation."""
    # Reduce frames for faster saving
    max_frames = 100
    if len(saved_t) > max_frames:
        step = len(saved_t) // max_frames
        indices = np.arange(0, len(saved_t), step)[:max_frames]
        saved_t = saved_t[indices]
        psi_hist = np.array(psi_hist)[indices]

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, np.max(np.abs(psi_hist) ** 2) * 1.1)
    ax.set_xlabel("x")
    ax.set_ylabel("|psi|^2")

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame: int):
        line.set_data(x, np.abs(psi_hist[frame]) ** 2)
        ax.set_title(f"Wave Packet, t = {saved_t[frame]:.2f}")
        return (line,)

    ani = FuncAnimation(fig, update, frames=len(saved_t), init_func=init, blit=True)
    ani.save(os.path.join(cfg.outdir, filename), writer=PillowWriter(fps=20), dpi=150)
    plt.close(fig)


def save_tunneling_animation(cfg: RunConfig, x: Array, v: Array, 
                             saved_t: Array, hist: Array, label: str) -> None:
    """Save tunneling animation."""
    # Reduce frames for faster saving
    max_frames = 100
    if len(saved_t) > max_frames:
        step = len(saved_t) // max_frames
        hist = np.array(hist)
        saved_t = np.array(saved_t)
        indices = np.arange(0, len(saved_t), step)[:max_frames]
        saved_t = saved_t[indices].tolist()
        hist = hist[indices]

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    line, = ax.plot([], [], lw=2)
    ax.fill_between(x, 0, v / max(np.max(v), 1e-12) * 0.08, 
                    color="#C73E1D", alpha=0.3)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, np.max(np.abs(hist) ** 2) * 1.1)
    ax.set_xlabel("x")
    ax.set_ylabel("|psi|^2")

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame: int):
        line.set_data(x, np.abs(hist[frame]) ** 2)
        ax.set_title(f"Tunneling, {label}, t = {saved_t[frame]:.2f}")
        return (line,)

    ani = FuncAnimation(fig, update, frames=len(saved_t), init_func=init, blit=True)
    ani.save(os.path.join(cfg.outdir, "figure6_tunneling_animation.gif"), 
            writer=PillowWriter(fps=20), dpi=150)
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
    max_frames = 60
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

