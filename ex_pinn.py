"""
ex_pinn.py -- PINN solver for 1D time-dependent Schrodinger equation
===================================================================

Physics-Informed Neural Network (PINN) approach to solve:
    i * psi_t = -1/2 * psi_xx + V(x) * psi

Compares PINN accuracy against traditional numerical methods
(Crank-Nicolson, Split-Step FFT) from the tdse package.

Requirements:  torch (with CUDA recommended)

This file is self-contained and does NOT affect the main program demo2.py.
"""

from __future__ import annotations

import os
import time
import sys
import warnings
from typing import Tuple, Optional

# Ignore PyTorch lr_scheduler verbose deprecation warning
warnings.filterwarnings("ignore", message="The verbose parameter is deprecated")

# Fix OMP conflict between numpy (from scipy/tdse) and PyTorch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import tdse package for numerical method comparisons and utilities
try:
    from tdse.config import RunConfig, ensure_outdir, setup_plot_style
    from tdse.potentials import (
        grid, gaussian_wavepacket, exact_free_gaussian, potential_free,
        probability_mass, l1_l2_linf_error, print_section, normalize,
    )
    from tdse.solvers import solve
except ImportError:
    print("ERROR: tdse package not found. Run from the project root directory.")
    sys.exit(1)

# PyTorch is required for PINN (imported inside main to allow graceful failure)
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# PINN Network Architecture
# =============================================================================

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for 1D TDSE.

    Takes (x, t) as input and outputs (psi_real, psi_imag).
    Uses a feedforward network with tanh activations and
    residual connections for better gradient flow.

    Architecture:
        Input: (x, t) -> [2]
        Hidden layers with residual skip connections
        Output: (u, v) -> [2]  (real and imaginary parts of psi)
    """

    def __init__(
        self,
        hidden_layers: int = 6,
        neurons: int = 128,
        activation: str = "tanh",
    ):
        super().__init__()

        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sin":
            self.activation = SinActivation()
        else:
            self.activation = nn.Tanh()

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(2, neurons),
            self.activation,
        )

        # Hidden layers with residual connections
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(neurons, neurons),
                    self.activation,
                )
            )

        # Output layer
        self.output_layer = nn.Linear(neurons, 2)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Spatial coordinates, shape (N, 1)
            t: Time coordinates, shape (N, 1)

        Returns:
            u: Real part of psi, shape (N, 1)
            v: Imaginary part of psi, shape (N, 1)
        """
        inputs = torch.cat([x, t], dim=1)

        h = self.input_layer(inputs)
        for layer in self.hidden_layers:
            h = h + layer(h)  # Residual connection

        output = self.output_layer(h)
        u = output[:, 0:1]
        v = output[:, 1:2]
        return u, v


class SinActivation(nn.Module):
    """Sine activation function (useful for periodic/oscillatory PDEs)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


# =============================================================================
# PINN Solver
# =============================================================================

class PINNSolver:
    """
    PINN-based solver for the 1D TDSE.

    Solves:  i * psi_t = -1/2 * psi_xx + V(x) * psi

    The network predicts psi = u + i*v where u and v are real-valued.
    PDE residuals are enforced through automatic differentiation.

    Example:
        >>> solver = PINNSolver(device="cuda")
        >>> solver.train(x_domain, t_domain, x_ic, psi0, V_func, epochs=5000)
        >>> psi_pred = solver.predict(x_test, t_test)
    """

    def __init__(
        self,
        hidden_layers: int = 6,
        neurons: int = 128,
        learning_rate: float = 1e-3,
        device: str = "cuda",
    ):
        self.device = torch.device(
            device if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        print(f"  PINN device: {self.device}")

        self.model = PINN(hidden_layers=hidden_layers, neurons=neurons).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=500, verbose=False
        )

        self.loss_history: dict = {"pde": [], "ic": [], "bc": [], "total": []}

    def compute_pde_residual(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        V_func: Optional[callable] = None,
    ) -> torch.Tensor:
        """
        Compute the PDE residual for the TDSE.

        TDSE: i * psi_t = -1/2 * psi_xx + V * psi

        In terms of u, v (psi = u + iv):
            f_R = u_t + 1/2 * v_xx - V * v = 0
            f_I = v_t - 1/2 * u_xx + V * u = 0
        """
        x.requires_grad_(True)
        t.requires_grad_(True)

        u, v = self.model(x, t)

        # First derivatives
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v),
                                   create_graph=True, retain_graph=True)[0]

        # Second derivatives in x
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True, retain_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                                   create_graph=True, retain_graph=True)[0]

        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True, retain_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
                                    create_graph=True, retain_graph=True)[0]

        # Potential term
        if V_func is not None:
            V_val = V_func(x.detach().cpu().numpy())
            V_val = torch.tensor(V_val, dtype=torch.float32, device=self.device)
            if V_val.ndim == 0:
                V_val = V_val.unsqueeze(0)
            V_val = V_val.view(-1, 1)
        else:
            V_val = torch.zeros_like(u)

        # PDE residuals
        f_real = u_t + 0.5 * v_xx - V_val * v
        f_imag = v_t - 0.5 * u_xx + V_val * u

        return torch.mean(f_real ** 2) + torch.mean(f_imag ** 2)

    def compute_ic_loss(
        self,
        x_ic: torch.Tensor,
        t_ic: torch.Tensor,
        psi0_real: torch.Tensor,
        psi0_imag: torch.Tensor,
    ) -> torch.Tensor:
        """Compute initial condition loss: ||psi_pred(x, 0) - psi0(x)||^2."""
        u_pred, v_pred = self.model(x_ic, t_ic)
        loss = torch.mean((u_pred - psi0_real) ** 2 + (v_pred - psi0_imag) ** 2)
        return loss

    def compute_bc_loss(
        self,
        x_bc: torch.Tensor,
        t_bc: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary condition loss (Dirichlet: psi=0 at boundaries)."""
        u_pred, v_pred = self.model(x_bc, t_bc)
        return torch.mean(u_pred ** 2 + v_pred ** 2)

    def train(
        self,
        x_domain: np.ndarray,
        t_domain: np.ndarray,
        x_ic: np.ndarray,
        psi0: np.ndarray,
        V_func: Optional[callable] = None,
        epochs: int = 2000,
        batch_size: int = 2048,
        lambda_pde: float = 1.0,
        lambda_ic: float = 10.0,
        lambda_bc: float = 5.0,
        verbose: bool = True,
    ) -> dict:
        """
        Train the PINN.

        Args:
            x_domain: Interior domain x points for PDE loss
            t_domain: Interior domain t points for PDE loss
            x_ic: x points for initial condition
            psi0: Initial wave function psi0(x_ic)
            V_func: Potential function V(x) or None for free particle
            epochs: Number of training epochs
            batch_size: Batch size for training
            lambda_pde: Weight for PDE loss
            lambda_ic: Weight for IC loss
            lambda_bc: Weight for BC loss
            verbose: Print progress

        Returns:
            Loss history dictionary
        """
        x_min, x_max = float(x_domain.min()), float(x_domain.max())
        t_min, t_max = 0.0, float(t_domain.max())

        # Prepare initial condition data
        x_ic_tensor = torch.tensor(x_ic.reshape(-1, 1), dtype=torch.float32, device=self.device)
        t_ic_tensor = torch.zeros_like(x_ic_tensor, device=self.device)
        psi0_real = torch.tensor(np.real(psi0).reshape(-1, 1), dtype=torch.float32, device=self.device)
        psi0_imag = torch.tensor(np.imag(psi0).reshape(-1, 1), dtype=torch.float32, device=self.device)

        if verbose:
            print(f"\n  Training PINN: {epochs} epochs, batch_size={batch_size}")
            print(f"  Domain: x=[{x_min:.1f}, {x_max:.1f}], t=[{t_min:.1f}, {t_max:.1f}]")
            print(f"  Architecture: {self.model.input_layer[0].in_features} -> "
                  f"{len(self.model.hidden_layers)+1} layers x "
                  f"{self.model.input_layer[0].out_features} neurons")
            print(f"  {'='*50}")
            print(f"  {'Epoch':>8} {'Loss':>12} {'PDE':>12} {'IC':>12} {'BC':>12}")
            print(f"  {'-'*50}")

        start_time = time.perf_counter()
        log_interval = max(1, epochs // 20)  # Log every 5% of training

        for epoch in range(epochs):
            # Sample collocation points for PDE loss
            x_pde = torch.rand(batch_size, 1, device=self.device) * (x_max - x_min) + x_min
            t_pde = torch.rand(batch_size, 1, device=self.device) * (t_max - t_min) + t_min

            # Sample boundary points
            n_bc = batch_size // 4
            x_bc_left = torch.full((n_bc // 2, 1), x_min, device=self.device)
            x_bc_right = torch.full((n_bc // 2, 1), x_max, device=self.device)
            x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
            t_bc = torch.rand(n_bc, 1, device=self.device) * (t_max - t_min) + t_min

            # Sample initial condition points (random subset)
            n_ic = min(batch_size // 4, len(x_ic))
            idx = torch.randperm(len(x_ic), device=self.device)[:n_ic]
            x_ic_batch = x_ic_tensor[idx]
            t_ic_batch = t_ic_tensor[idx]
            psi0_r_batch = psi0_real[idx]
            psi0_i_batch = psi0_imag[idx]

            loss_pde = self.compute_pde_residual(x_pde, t_pde, V_func)
            loss_ic = self.compute_ic_loss(x_ic_batch, t_ic_batch, psi0_r_batch, psi0_i_batch)
            loss_bc = self.compute_bc_loss(x_bc, t_bc)

            total_loss = lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_bc * loss_bc

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step(total_loss)

            self.loss_history["pde"].append(float(loss_pde))
            self.loss_history["ic"].append(float(loss_ic))
            self.loss_history["bc"].append(float(loss_bc))
            self.loss_history["total"].append(float(total_loss))

            if verbose and (epoch % log_interval == 0 or epoch == epochs - 1):
                print(f"  {epoch:6d}/{epochs}  "
                      f"{total_loss.item():12.2e}  "
                      f"{loss_pde.item():12.2e}  "
                      f"{loss_ic.item():12.2e}  "
                      f"{loss_bc.item():12.2e}")

        self.train_time = time.perf_counter() - start_time

        if verbose:
            print(f"  {'='*50}")
            print(f"\n  Training complete: {self.train_time:.1f}s")
            print(f"  Final losses - PDE: {self.loss_history['pde'][-1]:.2e}, "
                  f"IC: {self.loss_history['ic'][-1]:.2e}, "
                  f"BC: {self.loss_history['bc'][-1]:.2e}")

        return self.loss_history

    def predict(self, x: np.ndarray, t_val: float) -> np.ndarray:
        """
        Predict psi at spatial points x and time t.

        Args:
            x: Spatial grid, shape (N,)
            t_val: Time value

        Returns:
            psi: Complex wave function, shape (N,)
        """
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=self.device)
            t_tensor = torch.full_like(x_tensor, t_val, device=self.device)
            u, v = self.model(x_tensor, t_tensor)
            u_np = u.cpu().numpy().flatten()
            v_np = v.cpu().numpy().flatten()
            return u_np + 1j * v_np


# =============================================================================
# Potential function wrapper for PINN
# =============================================================================

def make_potential_func(x_array: np.ndarray, V_array: np.ndarray):
    """
    Create a callable potential function for the PINN from sampled data.

    Uses linear interpolation to provide V(x) for arbitrary x values
    requested by the PINN during training.
    """
    from scipy.interpolate import interp1d
    interp = interp1d(x_array, V_array, kind='linear',
                      bounds_error=False, fill_value=(V_array[0], V_array[-1]))
    return lambda x_vals: interp(x_vals)


# =============================================================================
# Experiment: Free Gaussian - PINN vs Numerical Methods
# =============================================================================

def experiment_pinn_free_gaussian(
    cfg: RunConfig,
    hidden_layers: int = 6,
    neurons: int = 128,
    epochs: int = 2000,
) -> None:
    """
    Compare PINN with traditional numerical methods for a free Gaussian.

    Uses the exact free-particle solution as the ground truth.
    Compares PINN against Crank-Nicolson and Split-Step FFT.
    """
    print_section("PINN Experiment: Free Gaussian Wave Packet")
    
    if not HAS_TORCH:
        print("ERROR: PyTorch is required for PINN. Install with: pip install torch")
        return
    
    outdir = os.path.join(cfg.outdir, "pinn_outputs")
    ensure_outdir(outdir)
    setup_plot_style(cfg.dpi)

    # Setup problem
    n = 256
    x, dx = grid(-15.0, 15.0, n)
    t_end = 1.5
    dt = 0.002
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    x0, sigma, k0 = -4.0, 0.8, 3.0

    psi0 = gaussian_wavepacket(x, x0, sigma, k0, dx)
    v = potential_free(x)

    # Exact solution
    psi_exact = exact_free_gaussian(x, t_end, x0, sigma, k0, dx)

    # ---- 1. Traditional numerical methods ----
    print("\n  Running traditional numerical methods...")
    methods = ["FTCS", "Backward-Euler", "Crank-Nicolson", "RK4", "Split-Step-FFT"]
    num_results = {}

    for method in methods:
        start = time.perf_counter()
        _, psi_hist = solve(method, psi0, v, x, t, dx, dt, store_every=len(t) - 1)
        runtime = time.perf_counter() - start
        psi_num = psi_hist[-1]
        l1, l2, linf = l1_l2_linf_error(psi_num, psi_exact, dx)
        mass = probability_mass(psi_num, dx)
        num_results[method] = {
            "psi": psi_num, "runtime": runtime, "L1": l1, "L2": l2, "Linf": linf,
            "mass": mass, "mass_err": abs(mass - 1.0),
        }
        print(f"    {method:<18s}  L2={l2:.4e}  mass_err={abs(mass-1.0):.2e}  "
              f"time={runtime:.4f}s")

    # ---- 2. PINN ----
    print("\n  Training PINN...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sample interior points for PDE loss
    n_pde = 5000
    x_pde_samples = np.random.uniform(x[0], x[-1], n_pde)
    t_pde_samples = np.random.uniform(0.0, t_end, n_pde)

    # Initial condition points
    x_ic = x.copy()

    # Set up PINN
    pinn = PINNSolver(
        hidden_layers=hidden_layers,
        neurons=neurons,
        learning_rate=1e-3,
        device=device,
    )

    # Train
    loss_hist = pinn.train(
        x_domain=x_pde_samples,
        t_domain=t_pde_samples,
        x_ic=x_ic,
        psi0=psi0,
        V_func=None,  # Free particle
        epochs=epochs,
        batch_size=2048,
        lambda_pde=1.0,
        lambda_ic=10.0,
        lambda_bc=5.0,
    )

    # Predict at final time
    psi_pinn = pinn.predict(x, t_end)

    # Compute errors
    l1, l2, linf = l1_l2_linf_error(psi_pinn, psi_exact, dx)
    mass = probability_mass(psi_pinn, dx)

    print(f"\n  PINN results:")
    print(f"    L2 error vs exact:   {l2:.4e}")
    print(f"    Mass conservation:   {mass:.6f}  (error: {abs(mass-1.0):.2e})")
    print(f"    Training time:       {pinn.train_time:.1f}s")

    # ---- 3. Plotting (5 separate high-DPI figures) ----
    method_colors = {
        "FTCS": "#E63946",
        "Backward-Euler": "#457B9D",
        "Crank-Nicolson": "#2A9D8F",
        "RK4": "#E9C46A",
        "Split-Step-FFT": "#F4A261",
        "PINN": "#2E86AB",
    }

    # Figure 1: Wave Function Comparison (Density, Real, Imag)
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    ax = axes1[0]
    ax.plot(x, np.abs(psi_exact) ** 2, "k--", lw=3.0, label="Exact", alpha=0.9)
    ax.plot(x, np.abs(psi_pinn) ** 2, lw=2.5, label=f"PINN", color=method_colors["PINN"])
    for method in methods:
        ax.plot(x, np.abs(num_results[method]["psi"]) ** 2, lw=1.8,
               label=method, color=method_colors[method], alpha=0.75)
    ax.set_xlabel("Position x", labelpad=8)
    ax.set_ylabel("|psi|^2", labelpad=8)
    ax.set_title("Probability Density Comparison", fontweight='bold', pad=10)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.95, ncol=2)
    ax.set_xlim(-10, 5)
    ax.grid(True, alpha=0.3)

    ax = axes1[1]
    ax.plot(x, np.real(psi_exact), "k--", lw=3.0, label="Exact", alpha=0.9)
    ax.plot(x, np.real(psi_pinn), lw=2.5, label="PINN", color=method_colors["PINN"])
    for method in methods:
        ax.plot(x, np.real(num_results[method]["psi"]), lw=1.8,
               label=method, color=method_colors[method], alpha=0.7)
    ax.set_xlabel("Position x", labelpad=8)
    ax.set_ylabel("Re[psi]", labelpad=8)
    ax.set_title("Real Part Comparison", fontweight='bold', pad=10)
    ax.legend(fontsize=7, framealpha=0.95)
    ax.set_xlim(-10, 5)
    ax.grid(True, alpha=0.3)

    ax = axes1[2]
    ax.plot(x, np.imag(psi_exact), "k--", lw=3.0, label="Exact", alpha=0.9)
    ax.plot(x, np.imag(psi_pinn), lw=2.5, label="PINN", color=method_colors["PINN"])
    for method in methods:
        ax.plot(x, np.imag(num_results[method]["psi"]), lw=1.8,
               label=method, color=method_colors[method], alpha=0.7)
    ax.set_xlabel("Position x", labelpad=8)
    ax.set_ylabel("Im[psi]", labelpad=8)
    ax.set_title("Imaginary Part Comparison", fontweight='bold', pad=10)
    ax.legend(fontsize=7, framealpha=0.95)
    ax.set_xlim(-10, 5)
    ax.grid(True, alpha=0.3)

    fig1.suptitle("PINN vs Numerical Methods: Wave Function", fontsize=14, fontweight='bold', y=1.02)
    fig1.savefig(os.path.join(outdir, "figure1_wave_function.png"), dpi=cfg.dpi, bbox_inches='tight')
    plt.close(fig1)

    # Figure 2: Error Analysis (L2, Mass, Absolute Error, Efficiency)
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    ax = axes2[0, 0]
    methods_all = ["PINN"] + methods
    l2_errors = [l2] + [num_results[m]["L2"] for m in methods]
    colors = [method_colors["PINN"]] + [method_colors[m] for m in methods]
    bars = ax.bar(methods_all, l2_errors, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yscale("log")
    ax.set_title("L2 Error (log scale)", fontweight='bold', pad=10)
    ax.set_xlabel("Method", labelpad=8)
    ax.set_ylabel("L2 Error", labelpad=8)
    ax.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, l2_errors):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.5, f"{val:.1e}",
               ha='center', va='bottom', fontsize=8, rotation=45)

    ax = axes2[0, 1]
    mass_errors = [abs(mass - 1.0)] + [num_results[m]["mass_err"] for m in methods]
    bars = ax.bar(methods_all, mass_errors, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yscale("log")
    ax.set_title("Mass Error (log scale)", fontweight='bold', pad=10)
    ax.set_xlabel("Method", labelpad=8)
    ax.set_ylabel("Mass Error", labelpad=8)
    ax.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, mass_errors):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.5, f"{val:.1e}",
               ha='center', va='bottom', fontsize=8, rotation=45)

    ax = axes2[1, 0]
    ax.plot(x, np.abs(psi_pinn - psi_exact), lw=2.0, label="PINN", color=method_colors["PINN"])
    for method in methods:
        ax.plot(x, np.abs(num_results[method]["psi"] - psi_exact), lw=1.5,
               label=method, color=method_colors[method], alpha=0.7)
    ax.set_xlabel("Position x", labelpad=8)
    ax.set_ylabel("|psi - psi_exact|", labelpad=8)
    ax.set_title("Absolute Error vs Exact", fontweight='bold', pad=10)
    ax.set_xlim(-10, 5)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes2[1, 1]
    runtimes_all = [pinn.train_time] + [num_results[m]["runtime"] for m in methods]
    efficiencies = [l2_err / rt if rt > 0 else float('inf') for l2_err, rt in zip(l2_errors, runtimes_all)]
    bars = ax.bar(methods_all, efficiencies, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yscale("log")
    ax.set_title("Efficiency (L2 Error / Runtime, lower is better)", fontweight='bold', pad=10)
    ax.set_xlabel("Method", labelpad=8)
    ax.set_ylabel("Efficiency (Error/Time)", labelpad=8)
    ax.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, efficiencies):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.5, f"{val:.1e}",
               ha='center', va='bottom', fontsize=8, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    fig2.suptitle("PINN vs Numerical Methods: Error Analysis", fontsize=14, fontweight='bold', y=1.02)
    fig2.savefig(os.path.join(outdir, "figure2_errors.png"), dpi=cfg.dpi, bbox_inches='tight')
    plt.close(fig2)

    # Figure 3: Runtime Comparison
    fig3, ax3 = plt.subplots(figsize=(10, 6), constrained_layout=True)
    runtimes = [pinn.train_time] + [num_results[m]["runtime"] for m in methods]
    bars = ax3.bar(methods_all, runtimes, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax3.set_title("Runtime Comparison", fontweight='bold', pad=10)
    ax3.set_xlabel("Method", labelpad=8)
    ax3.set_ylabel("Runtime (seconds)", labelpad=8)
    ax3.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, runtimes):
        if val < 0.01:
            label = f"{val*1000:.1f}ms"
        else:
            label = f"{val:.3f}s"
        ax3.text(bar.get_x() + bar.get_width() / 2, val * 1.1, label,
                ha='center', va='bottom', fontsize=9, rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    fig3.savefig(os.path.join(outdir, "figure3_runtime.png"), dpi=cfg.dpi, bbox_inches='tight')
    plt.close(fig3)

    # Figure 4: Training Loss History
    fig4, axes4 = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    epochs_arr = np.arange(1, len(loss_hist["total"]) + 1)

    ax = axes4[0]
    ax.semilogy(epochs_arr, loss_hist["total"], lw=2.0, label="Total Loss", color=method_colors["PINN"])
    ax.semilogy(epochs_arr, loss_hist["pde"], lw=1.5, label="PDE Loss", color="#A23B72", alpha=0.8)
    ax.semilogy(epochs_arr, loss_hist["ic"], lw=1.5, label="IC Loss", color="#F18F01", alpha=0.8)
    ax.semilogy(epochs_arr, loss_hist["bc"], lw=1.5, label="BC Loss", color="#95C623", alpha=0.8)
    ax.set_xlabel("Epoch", labelpad=8)
    ax.set_ylabel("Loss (log scale)", labelpad=8)
    ax.set_title("PINN Training Loss History", fontweight='bold', pad=10)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    ax = axes4[1]
    ax.plot(epochs_arr, loss_hist["total"], lw=2.0, label="Total Loss", color=method_colors["PINN"])
    ax.plot(epochs_arr, loss_hist["pde"], lw=1.5, label="PDE Loss", color="#A23B72", alpha=0.8)
    ax.plot(epochs_arr, loss_hist["ic"], lw=1.5, label="IC Loss", color="#F18F01", alpha=0.8)
    ax.plot(epochs_arr, loss_hist["bc"], lw=1.5, label="BC Loss", color="#95C623", alpha=0.8)
    ax.set_xlabel("Epoch", labelpad=8)
    ax.set_ylabel("Loss (linear scale)", labelpad=8)
    ax.set_title("PINN Training Loss (Linear Scale)", fontweight='bold', pad=10)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)

    fig4.suptitle("PINN Training Progress", fontsize=14, fontweight='bold', y=1.02)
    fig4.savefig(os.path.join(outdir, "figure4_training_loss.png"), dpi=cfg.dpi, bbox_inches='tight')
    plt.close(fig4)

    # Figure 5: Phase Portraits
    fig5, axes5 = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    ax = axes5[0, 0]
    ax.plot(np.real(psi_exact), np.imag(psi_exact), "k--", lw=2.5, label="Exact", alpha=0.8)
    ax.plot(np.real(psi_pinn), np.imag(psi_pinn), lw=2.0, label="PINN", color=method_colors["PINN"])
    idx = np.linspace(0, len(x) - 1, 20, dtype=int)
    ax.scatter(np.real(psi_exact)[idx], np.imag(psi_exact)[idx], c=x[idx], cmap="viridis", s=40, zorder=5)
    ax.set_xlabel("Re[psi]", labelpad=8)
    ax.set_ylabel("Im[psi]", labelpad=8)
    ax.set_title("PINN Phase Portrait", fontweight='bold', pad=10)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    phase_methods = ["Split-Step-FFT", "Crank-Nicolson", "RK4", "Backward-Euler", "FTCS"]
    for idx, method in enumerate(phase_methods):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        ax = axes5[row, col]
        ax.plot(np.real(psi_exact), np.imag(psi_exact), "k--", lw=2.0, label="Exact", alpha=0.7)
        ax.plot(np.real(num_results[method]["psi"]), np.imag(num_results[method]["psi"]),
               lw=1.8, label=method, color=method_colors[method])
        scatter_idx = np.linspace(0, len(x) - 1, 15, dtype=int)
        ax.scatter(np.real(psi_exact)[scatter_idx], np.imag(psi_exact)[scatter_idx],
                  c=x[scatter_idx], cmap="viridis", s=30, zorder=5, alpha=0.7)
        ax.set_xlabel("Re[psi]", labelpad=8)
        ax.set_ylabel("Im[psi]", labelpad=8)
        ax.set_title(f"{method} Phase Portrait", fontweight='bold', pad=10)
        ax.legend(fontsize=8, framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    fig5.suptitle("Phase Portraits: PINN vs Numerical Methods", fontsize=14, fontweight='bold', y=1.02)
    fig5.savefig(os.path.join(outdir, "figure5_phase_portraits.png"), dpi=cfg.dpi, bbox_inches='tight')
    plt.close(fig5)

    # ---- 5. Print summary ----
    print("\n" + "=" * 90)
    print("PINN EXPERIMENT SUMMARY")
    print("=" * 90)
    print(f"  {'Method':<20s} {'L2 Error':<14s} {'Mass Err':<14s} {'Runtime':<12s} {'Efficiency':<14s}")
    print(f"  {'-'*88}")
    pinn_efficiency = l2 / pinn.train_time if pinn.train_time > 0 else float('inf')
    print(f"  {'PINN':<20s} {l2:<14.4e} {abs(mass-1.0):<14.4e} "
          f"{pinn.train_time:<10.1f}s   {pinn_efficiency:<14.4e}")
    for m in methods:
        r = num_results[m]
        eff = r['L2'] / r['runtime'] if r['runtime'] > 0 else float('inf')
        print(f"  {m:<20s} {r['L2']:<14.4e} {r['mass_err']:<14.4e} "
              f"{r['runtime']:<10.4f}s   {eff:<14.4e}")
    print(f"\n  Efficiency = L2 Error / Runtime (lower is better)")
    print(f"\n  Output saved to: {os.path.abspath(outdir)}/")

    return {
        "pinn_l2": l2,
        "pinn_runtime": pinn.train_time,
        "numerical_results": num_results,
    }


# =============================================================================
# Experiment: PINN with Barrier Potential
# =============================================================================

def experiment_pinn_barrier(
    cfg: RunConfig,
    epochs: int = 2000,
) -> None:
    """
    PINN for a Gaussian wave packet scattering off a rectangular barrier.

    Compares with SSF numerical solution.
    """
    print_section("PINN Experiment: Barrier Scattering")

    if not HAS_TORCH:
        print("ERROR: PyTorch is required for PINN. Install with: pip install torch")
        return

    outdir = os.path.join(cfg.outdir, "pinn_outputs")
    ensure_outdir(outdir)
    setup_plot_style(cfg.dpi)

    n = 200
    x, dx = grid(-12.0, 12.0, n)
    t_end = 1.5
    dt = 0.002
    t = np.arange(0.0, t_end + 0.5 * dt, dt)

    barrier_left, barrier_right = -0.3, 0.3
    V0 = 2.0

    psi0 = gaussian_wavepacket(x, -5.0, 1.0, 3.0, dx)
    v = np.where((x > barrier_left) & (x < barrier_right), V0, 0.0).astype(float)

    V_func = make_potential_func(x, v)

    # Reference numerical solution (SSF)
    print("\n  Computing reference solution (Split-Step FFT)...")
    start = time.perf_counter()
    _, psi_hist = solve("Split-Step-FFT", psi0, v, x, t, dx, dt, store_every=len(t) - 1)
    psi_ref = psi_hist[-1]
    ref_time = time.perf_counter() - start
    print(f"    Done: {ref_time:.4f}s")

    # PINN
    print("\n  Training PINN...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_pde = 5000
    x_pde = np.random.uniform(x[0], x[-1], n_pde)
    t_pde = np.random.uniform(0.0, t_end, n_pde)

    pinn = PINNSolver(
        hidden_layers=6,
        neurons=128,
        learning_rate=1e-3,
        device=device,
    )

    _ = pinn.train(
        x_domain=x_pde,
        t_domain=t_pde,
        x_ic=x.copy(),
        psi0=psi0,
        V_func=V_func,
        epochs=epochs,
        batch_size=2048,
        lambda_pde=1.0,
        lambda_ic=10.0,
        lambda_bc=5.0,
    )

    psi_pinn = pinn.predict(x, t_end)

    l1, l2, linf = l1_l2_linf_error(psi_pinn, psi_ref, dx)
    mass = probability_mass(psi_pinn, dx)

    print(f"\n  PINN vs SSF comparison:")
    print(f"    L2 error vs SSF:     {l2:.4e}")
    print(f"    Mass conservation:   {mass:.6f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(x, np.abs(psi_ref) ** 2, lw=2.5, label="Split-Step FFT (ref)",
            color="#F18F01")
    ax.plot(x, np.abs(psi_pinn) ** 2, lw=2.0, label="PINN", color="#2E86AB")
    ax.fill_between(x, 0, v / V0 * 0.15, color="#C73E1D", alpha=0.3,
                     label="Barrier")
    ax.set_xlabel("Position x", labelpad=8)
    ax.set_ylabel("|psi|^2", labelpad=8)
    ax.set_title("Barrier Scattering: Density", fontweight='bold', pad=10)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.set_xlim(-10, 10)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x, np.real(psi_ref), lw=2.5, label="SSF", color="#F18F01")
    ax.plot(x, np.real(psi_pinn), lw=2.0, label="PINN", color="#2E86AB")
    ax.set_xlabel("Position x", labelpad=8)
    ax.set_ylabel("Re[psi]", labelpad=8)
    ax.set_title("Real Part Comparison", fontweight='bold', pad=10)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.set_xlim(-10, 10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, np.imag(psi_ref), lw=2.5, label="SSF", color="#F18F01")
    ax.plot(x, np.imag(psi_pinn), lw=2.0, label="PINN", color="#C73E1D")
    ax.set_xlabel("Position x", labelpad=8)
    ax.set_ylabel("Im[psi]", labelpad=8)
    ax.set_title("Imaginary Part Comparison", fontweight='bold', pad=10)
    ax.legend(fontsize=9, framealpha=0.95)
    ax.set_xlim(-10, 10)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    abs_err = np.abs(psi_pinn - psi_ref)
    ax.plot(x, abs_err, lw=1.8, color="#C73E1D")
    ax.fill_between(x, 0, abs_err, alpha=0.15, color="#C73E1D")
    ax.set_xlabel("Position x", labelpad=8)
    ax.set_ylabel("|psi_PINN - psi_SSF|", labelpad=8)
    ax.set_title("Absolute Error vs SSF", fontweight='bold', pad=10)
    ax.set_xlim(-10, 10)
    ax.grid(True, alpha=0.3)

    fig.suptitle("PINN: Barrier Scattering Comparison",
                 fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(os.path.join(outdir, "figure_pinn_barrier.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(f"\n  Output saved to: {os.path.abspath(outdir)}/")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run PINN experiments for 1D TDSE."""

    if not HAS_TORCH:
        print("=" * 70)
        print("ERROR: PyTorch is required for PINN experiments.")
        print("Install with: pip install torch torchvision torchaudio")
        print("=" * 70)
        sys.exit(1)

    import argparse

    parser = argparse.ArgumentParser(
        description="PINN solver for 1D time-dependent Schrodinger equation"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer training epochs for quick testing")
    parser.add_argument("--epochs", type=int, default=2000,
                        help="Number of training epochs (default: 2000)")
    parser.add_argument("--neurons", type=int, default=128,
                        help="Number of neurons per hidden layer (default: 128)")
    parser.add_argument("--layers", type=int, default=6,
                        help="Number of hidden layers (default: 6)")
    parser.add_argument("--barrier", action="store_true",
                        help="Run barrier scattering experiment")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Plot DPI (default: 300)")

    args = parser.parse_args()

    epochs = 1000 if args.quick else args.epochs

    cfg = RunConfig(outdir="tdse_outputs", dpi=args.dpi)
    ensure_outdir(cfg.outdir)
    setup_plot_style(cfg.dpi)

    print("=" * 70)
    print(" " * 12 + "PINN FOR 1D SCHRODINGER EQUATION")
    print("=" * 70)
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device:     {torch.cuda.get_device_name(0)}")
    print(f"  Training epochs: {epochs}")
    print(f"  Network:         {args.layers} layers x {args.neurons} neurons")
    print(f"  Output DPI:      {args.dpi}")

    t0 = time.perf_counter()

    # Free Gaussian experiment (always run)
    experiment_pinn_free_gaussian(
        cfg,
        hidden_layers=args.layers,
        neurons=args.neurons,
        epochs=epochs,
    )

    # Barrier scattering experiment (optional)
    if args.barrier:
        experiment_pinn_barrier(cfg, epochs=epochs)

    total_time = time.perf_counter() - t0
    print(f"\nTotal runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
