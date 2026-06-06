"""
TDSE Configuration Module
========================

Unified configuration system for TDSE simulations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt


@dataclass
class TDSEConfig:
    """
    Unified configuration for all TDSE simulations.

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

    outdir: str = "tdse_outputs"
    quick: bool = False
    save_gif: bool = True
    dpi: int = 600
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
    outdir: str = "tdse_outputs"
    quick: bool = False
    save_gif: bool = True
    dpi: int = 600


def ensure_outdir(outdir: str) -> None:
    """Ensure output directory exists."""
    os.makedirs(outdir, exist_ok=True)


def setup_plot_style(dpi: int = 600) -> None:
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
    # Prefer vector-friendly output for publication; PNG still supported
    plt.rcParams['savefig.format'] = 'png'
    # Ensure PDF/PS output embeds fonts as Type42 (good for publishers)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Image display settings
    plt.rcParams['image.cmap'] = 'inferno'
    plt.rcParams['image.interpolation'] = 'bilinear'

    # Grid styling - subtle but visible
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.color'] = '#CCCCCC'

    # Professional color palette with good distinguishability (8 colors)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(
        'color', [
            '#2E86AB', '#A23B72', '#F18F01', '#C73E1D',
            '#3B1F2B', '#95C623', '#6B4C9A', '#1B998B',
        ]
    )

    # Spine styling - clean look
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    # Tick styling
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0

    # Legend styling
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = '#CCCCCC'

    # Layout adjustments - prevent label cutoff and text overlap
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['figure.constrained_layout.h_pad'] = 0.5
    plt.rcParams['figure.constrained_layout.w_pad'] = 0.5
    plt.rcParams['figure.constrained_layout.hspace'] = 0.2
    plt.rcParams['figure.constrained_layout.wspace'] = 0.2

