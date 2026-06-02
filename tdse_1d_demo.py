"""
Time-Dependent Schrodinger Equation (1D) research-grade single-file demo.

Dimensionless model:
    i psi_t = -1/2 psi_xx + V(x) psi

Implemented methods:
    FTCS, Backward Euler, Crank-Nicolson, Split-Step Fourier, RK4.

Outputs are written to ./tdse_outputs:
    Figure 1: analytic vs numerical
    Figure 2: error convergence
    Figure 3: stability map
    Figure 4: performance table
    Figure 5: wavepacket animation
    Figure 6: tunneling simulation

Dependencies:
    numpy scipy matplotlib pandas tqdm
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.linalg import solve_banded
from scipy.sparse import diags
from tqdm import tqdm


Array = np.ndarray


# =============================================================================
# Configuration and utility routines
# =============================================================================


@dataclass
class RunConfig:
    """Small collection of runtime controls for the demo."""

    outdir: str = "tdse_outputs"
    quick: bool = False
    save_gif: bool = True
    dpi: int = 150


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def normalize(psi: Array, dx: float) -> Array:
    """Normalize a wave function in L2."""

    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if norm == 0:
        raise ValueError("Cannot normalize a zero wave function.")
    return psi / norm


def probability_mass(psi: Array, dx: float) -> float:
    return float(np.sum(np.abs(psi) ** 2) * dx)


def l1_l2_linf_error(psi_num: Array, psi_ref: Array, dx: float) -> Tuple[float, float, float]:
    """Errors for complex wave functions, after global phase alignment."""

    psi_aligned = align_global_phase(psi_num, psi_ref, dx)
    diff = psi_aligned - psi_ref
    l1 = float(np.sum(np.abs(diff)) * dx)
    l2 = float(np.sqrt(np.sum(np.abs(diff) ** 2) * dx))
    linf = float(np.max(np.abs(diff)))
    return l1, l2, linf


def align_global_phase(psi: Array, reference: Array, dx: float) -> Array:
    """Remove irrelevant global phase before comparing wave functions."""

    inner = np.sum(np.conj(reference) * psi) * dx
    if np.abs(inner) < 1e-14:
        return psi
    return psi * np.exp(-1j * np.angle(inner))


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def grid(xmin: float, xmax: float, n: int) -> Tuple[Array, float]:
    x = np.linspace(xmin, xmax, n, endpoint=False)
    dx = float(x[1] - x[0])
    return x, dx


def dt_warning(method: str, dx: float, dt: float, vmax: float = 0.0) -> str:
    """Heuristic stability/accuracy note for dimensionless TDSE."""

    r = dt / dx**2
    method = method.lower()
    if method == "ftcs":
        return (
            f"FTCS has amplification > 1 for Schrodinger dynamics; r=dt/dx^2={r:.3g}. "
            "Use only as an instability demonstration."
        )
    if method == "rk4":
        return (
            f"RK4 with centered Laplacian is conditionally stable. "
            f"Heuristic r=dt/dx^2={r:.3g}; keep it small, e.g. r <= 0.25."
        )
    if method in ("cn", "crank-nicolson", "backward-euler"):
        return (
            f"{method} is unconditionally stable in norm/boundedness sense for this linear problem, "
            "but dt still controls phase accuracy."
        )
    if method in ("split-step", "split-step-fft", "fft"):
        return (
            "Split-Step FFT is unitary for real V with periodic boundary assumptions; "
            f"accuracy depends on dt and spectral resolution. max(V)={vmax:.3g}."
        )
    return "Unknown method."


# =============================================================================
# Physical models and analytic references
# =============================================================================


def potential_free(x: Array) -> Array:
    return np.zeros_like(x, dtype=float)


def potential_harmonic(x: Array, omega: float = 1.0) -> Array:
    return 0.5 * omega**2 * x**2


def potential_rect_barrier(x: Array, v0: float = 1.0, a: float = -0.5, b: float = 0.5) -> Array:
    return np.where((x > a) & (x < b), v0, 0.0).astype(float)


def potential_infinite_well_mask(x: Array, a: float = -5.0, b: float = 5.0) -> Array:
    """Finite computational representation of an infinite well domain mask."""

    return ((x > a) & (x < b)).astype(float)


def gaussian_wavepacket(x: Array, x0: float, sigma: float, k0: float, dx: float) -> Array:
    psi = np.exp(-((x - x0) ** 2) / (2.0 * sigma**2)) * np.exp(1j * k0 * x)
    return normalize(psi.astype(complex), dx)


def exact_free_gaussian(x: Array, t: float, x0: float, sigma: float, k0: float, dx: float) -> Array:
    """
    Exact free-particle Gaussian propagation for initial
        exp(-(x-x0)^2/(2 sigma^2)) exp(i k0 x)
    in dimensionless units. The expression is normalized numerically to match
    the discrete grid used by the demo.
    """

    denom = sigma**2 + 1j * t
    prefactor = sigma / np.sqrt(denom)
    phase = np.exp(1j * k0 * x - 0.5j * k0**2 * t)
    envelope = np.exp(-((x - x0 - k0 * t) ** 2) / (2.0 * denom))
    return normalize((prefactor * envelope * phase).astype(complex), dx)


def infinite_well_eigenstate(x: Array, n: int, a: float = -5.0, b: float = 5.0) -> Tuple[Array, float]:
    """Analytic eigenstate and energy for an infinite well on [a,b]."""

    length = b - a
    psi = np.zeros_like(x, dtype=complex)
    inside = (x >= a) & (x <= b)
    psi[inside] = np.sqrt(2.0 / length) * np.sin(n * np.pi * (x[inside] - a) / length)
    energy = 0.5 * (n * np.pi / length) ** 2
    return psi, energy


def harmonic_eigenstate_low(x: Array, n: int = 0, omega: float = 1.0, dx: float = 1.0) -> Tuple[Array, float]:
    """
    Low-lying analytic harmonic oscillator states for n=0,1,2.
    H = -1/2 dxx + 1/2 omega^2 x^2, E_n = omega(n+1/2).
    """

    xi = np.sqrt(omega) * x
    gaussian = (omega / np.pi) ** 0.25 * np.exp(-0.5 * xi**2)
    if n == 0:
        psi = gaussian
    elif n == 1:
        psi = np.sqrt(2.0) * xi * gaussian
    elif n == 2:
        psi = (1.0 / np.sqrt(2.0)) * (2.0 * xi**2 - 1.0) * gaussian
    else:
        raise ValueError("This compact demo implements harmonic oscillator states n=0,1,2.")
    energy = omega * (n + 0.5)
    return normalize(psi.astype(complex), dx), energy


# =============================================================================
# Finite-difference operators and solvers
# =============================================================================


def laplacian_dirichlet(psi: Array, dx: float) -> Array:
    """Second-order centered Laplacian with homogeneous Dirichlet endpoints."""

    out = np.zeros_like(psi, dtype=complex)
    out[1:-1] = (psi[:-2] - 2.0 * psi[1:-1] + psi[2:]) / dx**2
    return out


def hamiltonian_apply(psi: Array, v: Array, dx: float) -> Array:
    return -0.5 * laplacian_dirichlet(psi, dx) + v * psi


def banded_hamiltonian(v: Array, dx: float) -> Array:
    """Banded matrix for H = -1/2 Dxx + V with Dirichlet endpoints."""

    n = len(v)
    main = np.ones(n) / dx**2 + v
    off = -0.5 * np.ones(n - 1) / dx**2
    ab = np.zeros((3, n), dtype=complex)
    ab[0, 1:] = off
    ab[1, :] = main
    ab[2, :-1] = off
    return ab


def apply_tridiagonal(ab: Array, psi: Array) -> Array:
    """Apply a tridiagonal banded matrix stored in scipy solve_banded layout."""

    out = ab[1] * psi
    out[1:] += ab[2, :-1] * psi[:-1]
    out[:-1] += ab[0, 1:] * psi[1:]
    return out


def step_ftcs(psi: Array, v: Array, dx: float, dt: float) -> Array:
    """Explicit forward Euler in time, centered second difference in space."""

    return psi - 1j * dt * hamiltonian_apply(psi, v, dx)


def step_backward_euler(psi: Array, v: Array, dx: float, dt: float) -> Array:
    """Implicit backward Euler: (I + i dt H) psi^{n+1} = psi^n."""

    h = banded_hamiltonian(v, dx)
    a = h.copy()
    a[1] = 1.0 + 1j * dt * h[1]
    a[0] = 1j * dt * h[0]
    a[2] = 1j * dt * h[2]
    return solve_banded((1, 1), a, psi)


def step_crank_nicolson(psi: Array, v: Array, dx: float, dt: float) -> Array:
    """Crank-Nicolson: (I+i dt H/2) psi^{n+1}=(I-i dt H/2) psi^n."""

    h = banded_hamiltonian(v, dx)
    left = h.copy()
    left[1] = 1.0 + 0.5j * dt * h[1]
    left[0] = 0.5j * dt * h[0]
    left[2] = 0.5j * dt * h[2]

    right = h.copy()
    right[1] = 1.0 - 0.5j * dt * h[1]
    right[0] = -0.5j * dt * h[0]
    right[2] = -0.5j * dt * h[2]
    rhs = apply_tridiagonal(right, psi)
    return solve_banded((1, 1), left, rhs)


def step_rk4(psi: Array, v: Array, dx: float, dt: float) -> Array:
    """Classical RK4 for psi_t = -i H psi with second-order spatial stencil."""

    def f(y: Array) -> Array:
        return -1j * hamiltonian_apply(y, v, dx)

    k1 = f(psi)
    k2 = f(psi + 0.5 * dt * k1)
    k3 = f(psi + 0.5 * dt * k2)
    k4 = f(psi + dt * k3)
    return psi + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def step_split_step_fft(psi: Array, v: Array, x: Array, dx: float, dt: float) -> Array:
    """
    Strang split-step Fourier:
        exp(-i V dt/2) -> FFT -> exp(-i k^2 dt/2) -> IFFT -> exp(-i V dt/2)
    because H_kin = k^2/2 in Fourier space.
    """

    k = 2.0 * np.pi * np.fft.fftfreq(len(x), d=dx)
    psi_half = np.exp(-0.5j * v * dt) * psi
    psi_k = np.fft.fft(psi_half)
    psi_k *= np.exp(-0.5j * k**2 * dt)
    psi_new = np.fft.ifft(psi_k)
    psi_new *= np.exp(-0.5j * v * dt)
    return psi_new


def solve(
    method: str,
    psi0: Array,
    v: Array,
    x: Array,
    t: Array,
    dx: float,
    dt: float,
    store_every: int = 1,
    renormalize: bool = False,
) -> Tuple[Array, Array]:
    """
    Unified solver interface required by the prompt.

    Returns:
        saved_t, saved_psi
    """

    method_key = method.lower()
    psi = psi0.astype(complex).copy()
    saved_t = [float(t[0])]
    saved_psi = [psi.copy()]

    for n in range(1, len(t)):
        if method_key == "ftcs":
            psi = step_ftcs(psi, v, dx, dt)
        elif method_key in ("backward-euler", "be"):
            psi = step_backward_euler(psi, v, dx, dt)
        elif method_key in ("crank-nicolson", "cn"):
            psi = step_crank_nicolson(psi, v, dx, dt)
        elif method_key in ("split-step", "split-step-fft", "fft", "ssf"):
            psi = step_split_step_fft(psi, v, x, dx, dt)
        elif method_key == "rk4":
            psi = step_rk4(psi, v, dx, dt)
        else:
            raise ValueError(f"Unknown method: {method}")

        if renormalize and method_key not in ("ftcs",):
            psi = normalize(psi, dx)
        if n % store_every == 0 or n == len(t) - 1:
            saved_t.append(float(t[n]))
            saved_psi.append(psi.copy())

    return np.array(saved_t), np.array(saved_psi)


# =============================================================================
# Experiments
# =============================================================================


def experiment_analytic_vs_numerical(cfg: RunConfig) -> pd.DataFrame:
    """Figure 1 and a compact analytic comparison table."""

    print_section("Figure 1: analytic vs numerical")
    n = 384 if cfg.quick else 512
    x, dx = grid(-30.0, 30.0, n)
    t_end = 2.0
    dt = 0.0025 if cfg.quick else 0.002
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    x0, sigma, k0 = -8.0, 1.2, 2.0
    psi0 = gaussian_wavepacket(x, x0, sigma, k0, dx)
    v = potential_free(x)
    methods = ["FTCS", "Backward-Euler", "Crank-Nicolson", "Split-Step-FFT", "RK4"]
    rows = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    psi_exact = exact_free_gaussian(x, t_end, x0, sigma, k0, dx)
    axes[0, 0].plot(x, np.abs(psi_exact) ** 2, "k--", lw=2, label="exact")

    final_solutions = {}
    for method in tqdm(methods, desc="analytic comparison"):
        start = time.perf_counter()
        _, psi_hist = solve(method, psi0, v, x, t, dx, dt, store_every=len(t) - 1)
        runtime = time.perf_counter() - start
        psi_num = psi_hist[-1]
        final_solutions[method] = psi_num
        l1, l2, linf = l1_l2_linf_error(psi_num, psi_exact, dx)
        mass = probability_mass(psi_num, dx)
        rows.append(
            {
                "method": method,
                "N": n,
                "dx": dx,
                "dt": dt,
                "runtime_s": runtime,
                "L1": l1,
                "L2": l2,
                "Linf": linf,
                "mass": mass,
            }
        )
        if method != "FTCS":
            axes[0, 0].plot(x, np.abs(psi_num) ** 2, lw=1.2, label=method)

    axes[0, 0].set_title("Free Gaussian: |psi|^2 at final time")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("|psi|^2")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_xlim(-15, 5)

    l2_values = [r["L2"] for r in rows]
    axes[0, 1].bar([r["method"] for r in rows], l2_values)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("L2 error vs exact Gaussian")
    axes[0, 1].tick_params(axis="x", rotation=35)

    # Infinite well analytic eigenstate: CN should preserve phase and density.
    xw, dxw = grid(-5.0, 5.0, n)
    psi_well0, e_well = infinite_well_eigenstate(xw, n=2, a=-5.0, b=5.0)
    tw_end, dtw = 1.5, 0.0025
    tw = np.arange(0.0, tw_end + 0.5 * dtw, dtw)
    vw = np.zeros_like(xw)
    _, psi_well_hist = solve("Crank-Nicolson", psi_well0, vw, xw, tw, dxw, dtw, store_every=len(tw) - 1)
    psi_well_exact = psi_well0 * np.exp(-1j * e_well * tw_end)
    axes[1, 0].plot(xw, np.abs(psi_well_exact) ** 2, "k--", lw=2, label="well exact n=2")
    axes[1, 0].plot(xw, np.abs(psi_well_hist[-1]) ** 2, lw=1.5, label="CN")
    axes[1, 0].set_title("Infinite well eigenstate")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("|psi|^2")
    axes[1, 0].legend(fontsize=8)

    # Harmonic oscillator low state: compare against analytic phase evolution.
    xh, dxh = grid(-10.0, 10.0, n)
    omega = 1.0
    psi_h0, e_h = harmonic_eigenstate_low(xh, n=1, omega=omega, dx=dxh)
    vh = potential_harmonic(xh, omega)
    th_end, dth = 1.0, 0.002
    th = np.arange(0.0, th_end + 0.5 * dth, dth)
    _, psi_h_hist = solve("Crank-Nicolson", psi_h0, vh, xh, th, dxh, dth, store_every=len(th) - 1)
    psi_h_exact = psi_h0 * np.exp(-1j * e_h * th_end)
    axes[1, 1].plot(xh, np.abs(psi_h_exact) ** 2, "k--", lw=2, label="HO exact n=1")
    axes[1, 1].plot(xh, np.abs(psi_h_hist[-1]) ** 2, lw=1.5, label="CN")
    axes[1, 1].set_title("Harmonic oscillator low state")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("|psi|^2")
    axes[1, 1].legend(fontsize=8)

    fig.savefig(os.path.join(cfg.outdir, "figure1_analytic_vs_numerical.png"), dpi=cfg.dpi)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "analytic_error_table.csv"), index=False)
    print(df.to_string(index=False))
    return df


def experiment_convergence(cfg: RunConfig) -> pd.DataFrame:
    """Figure 2: error vs dx and fitted log-log convergence order."""

    print_section("Figure 2: convergence study")
    ns = [50, 100, 200, 400]
    methods = ["Crank-Nicolson", "Split-Step-FFT", "RK4"]
    rows = []
    xmin, xmax = -30.0, 30.0
    t_end = 0.8
    x0, sigma, k0 = -8.0, 1.0, 1.5

    for n in tqdm(ns, desc="convergence grids"):
        x, dx = grid(xmin, xmax, n)
        dt = min(0.0015, 0.04 * dx**2)
        if cfg.quick:
            dt = min(0.003, 0.06 * dx**2)
        t = np.arange(0.0, t_end + 0.5 * dt, dt)
        psi0 = gaussian_wavepacket(x, x0, sigma, k0, dx)
        v = potential_free(x)
        psi_ref = exact_free_gaussian(x, t[-1], x0, sigma, k0, dx)
        for method in methods:
            _, hist = solve(method, psi0, v, x, t, dx, dt, store_every=len(t) - 1)
            l1, l2, linf = l1_l2_linf_error(hist[-1], psi_ref, dx)
            rows.append({"method": method, "N": n, "dx": dx, "dt": dt, "L1": l1, "L2": l2, "Linf": linf})

    df = pd.DataFrame(rows)
    orders = []
    for method in methods:
        sub = df[df["method"] == method].sort_values("dx")
        coeff = np.polyfit(np.log(sub["dx"]), np.log(sub["L2"]), 1)
        order = float(coeff[0])
        orders.append({"method": method, "fitted_L2_order": order})
    order_df = pd.DataFrame(orders)
    df.to_csv(os.path.join(cfg.outdir, "convergence_errors.csv"), index=False)
    order_df.to_csv(os.path.join(cfg.outdir, "convergence_orders.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for method in methods:
        sub = df[df["method"] == method].sort_values("dx")
        label = f"{method}, p={order_df[order_df.method == method].fitted_L2_order.iloc[0]:.2f}"
        ax.loglog(sub["dx"], sub["L2"], "o-", label=label)
    ax.invert_xaxis()
    ax.set_xlabel("dx")
    ax.set_ylabel("L2 error")
    ax.set_title("Error convergence against exact free Gaussian")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(cfg.outdir, "figure2_error_convergence.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(df.to_string(index=False))
    print("\nFitted log-log convergence orders:")
    print(order_df.to_string(index=False))
    return df


def experiment_stability(cfg: RunConfig) -> pd.DataFrame:
    """Figure 3: stability map over dt and dx."""

    print_section("Figure 3: stability scan")
    methods = ["FTCS", "Backward-Euler", "Crank-Nicolson", "Split-Step-FFT", "RK4"]
    ns = [64, 96, 128, 192] if not cfg.quick else [64, 96, 128]
    dts = [0.001, 0.004, 0.010, 0.020, 0.040, 0.080] if not cfg.quick else [0.004, 0.020, 0.080]
    rows = []
    t_end = 1.5 if not cfg.quick else 1.0
    for n in tqdm(ns, desc="stability grids"):
        x, dx = grid(-20.0, 20.0, n)
        psi0 = gaussian_wavepacket(x, -5.0, 1.0, 2.0, dx)
        v = potential_free(x)
        for dt in dts:
            t = np.arange(0.0, t_end + 0.5 * dt, dt)
            for method in methods:
                try:
                    _, hist = solve(method, psi0, v, x, t, dx, dt, store_every=len(t) - 1)
                    mass = probability_mass(hist[-1], dx)
                    max_amp = float(np.max(np.abs(hist[-1])))
                    stable = np.isfinite(mass) and 0.2 < mass < 1.8 and max_amp < 8.0
                except Exception:
                    mass = np.nan
                    max_amp = np.nan
                    stable = False
                rows.append(
                    {
                        "method": method,
                        "N": n,
                        "dx": dx,
                        "dt": dt,
                        "r_dt_over_dx2": dt / dx**2,
                        "mass": mass,
                        "max_amp": max_amp,
                        "stable": stable,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "stability_scan.csv"), index=False)

    fig, axes = plt.subplots(1, len(methods), figsize=(16, 3.5), constrained_layout=True)
    for ax, method in zip(axes, methods):
        sub = df[df["method"] == method]
        pivot = sub.pivot(index="dt", columns="N", values="stable").astype(int)
        im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(method)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:g}" for v in pivot.index])
        ax.set_xlabel("N")
        if ax is axes[0]:
            ax.set_ylabel("dt")
    fig.colorbar(im, ax=axes, label="stable=1, divergent=0", shrink=0.85)
    fig.savefig(os.path.join(cfg.outdir, "figure3_stability_map.png"), dpi=cfg.dpi)
    plt.close(fig)

    summary = df.groupby("method")["stable"].agg(["sum", "count"])
    summary["verdict"] = np.where(summary["sum"] == summary["count"], "stable in scan", "conditional/divergent cases")
    print(summary.to_string())
    print("\nStability notes:")
    for method in methods:
        print(f"- {method}: {dt_warning(method, 0.2, 0.001)}")
    return df


def transmission_reflection(psi: Array, x: Array, dx: float, barrier_center: float = 0.0, buffer: float = 0.0) -> Tuple[float, float]:
    left = x < barrier_center - buffer
    right = x > barrier_center + buffer
    r = float(np.sum(np.abs(psi[left]) ** 2) * dx)
    t = float(np.sum(np.abs(psi[right]) ** 2) * dx)
    return t, r


def experiment_tunneling(cfg: RunConfig) -> pd.DataFrame:
    """Figure 6: rectangular barrier tunneling, with R+T approximately conserved."""

    print_section("Figure 6: tunneling simulation")
    n = 768 if cfg.quick else 1024
    x, dx = grid(-140.0, 140.0, n)
    v0 = 1.0
    a, b = -1.0, 1.0
    v = potential_rect_barrier(x, v0=v0, a=a, b=b)
    sigma, x0 = 4.0, -55.0
    dt = 0.02
    t_end = 70.0 if cfg.quick else 85.0
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    cases = [
        ("E<V0", np.sqrt(2.0 * 0.55 * v0)),
        ("E~V0", np.sqrt(2.0 * 1.00 * v0)),
        ("E>V0", np.sqrt(2.0 * 1.80 * v0)),
    ]
    rows = []

    fig, axes = plt.subplots(len(cases), 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    animation_hist = None
    animation_t = None
    animation_label = None
    for ax, (label, k0) in zip(axes, cases):
        psi0 = gaussian_wavepacket(x, x0, sigma, k0, dx)
        energy = 0.5 * k0**2
        store_every = max(1, len(t) // 220)
        saved_t, hist = solve("Split-Step-FFT", psi0, v, x, t, dx, dt, store_every=store_every)
        psi_final = hist[-1]
        tprob, rprob = transmission_reflection(psi_final, x, dx, barrier_center=0.0, buffer=0.0)
        rows.append({"case": label, "E": energy, "V0": v0, "T": tprob, "R": rprob, "R_plus_T": rprob + tprob})
        ax.plot(x, np.abs(psi0) ** 2, color="0.75", lw=1, label="initial")
        ax.plot(x, np.abs(psi_final) ** 2, lw=1.5, label="final")
        ax.fill_between(x, 0, v / max(v0, 1e-12) * 0.05, color="tab:red", alpha=0.3, label="barrier")
        ax.set_xlim(-95, 95)
        ax.set_ylabel("|psi|^2")
        ax.set_title(f"{label}: E={energy:.2f}, T={tprob:.3f}, R={rprob:.3f}, R+T={rprob+tprob:.3f}")
        ax.legend(fontsize=8, loc="upper right")
        if label == "E<V0":
            animation_hist = hist
            animation_t = saved_t
            animation_label = label
    axes[-1].set_xlabel("x")
    fig.savefig(os.path.join(cfg.outdir, "figure6_tunneling_simulation.png"), dpi=cfg.dpi)
    plt.close(fig)

    if cfg.save_gif and animation_hist is not None:
        save_tunneling_animation(cfg, x, v, animation_t, animation_hist, animation_label)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "tunneling_RT.csv"), index=False)
    print(df.to_string(index=False))
    return df


def experiment_performance(cfg: RunConfig) -> pd.DataFrame:
    """Figure 4: runtime comparison table for FTCS, CN and Split-Step FFT."""

    print_section("Figure 4: performance comparison")
    ns = [256, 512, 1024] if not cfg.quick else [192, 384]
    methods = ["FTCS", "Crank-Nicolson", "Split-Step-FFT"]
    rows = []
    for n in tqdm(ns, desc="performance"):
        x, dx = grid(-30.0, 30.0, n)
        dt = 0.002
        t_end = 0.35
        t = np.arange(0.0, t_end + 0.5 * dt, dt)
        psi0 = gaussian_wavepacket(x, -8.0, 1.2, 2.0, dx)
        v = potential_harmonic(x, omega=0.08)
        for method in methods:
            start = time.perf_counter()
            _, hist = solve(method, psi0, v, x, t, dx, dt, store_every=len(t) - 1)
            runtime = time.perf_counter() - start
            rows.append(
                {
                    "method": method,
                    "grid_size": n,
                    "dx": dx,
                    "dt": dt,
                    "steps": len(t) - 1,
                    "runtime_s": runtime,
                    "mass_final": probability_mass(hist[-1], dx),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "performance_table.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    ax.axis("off")
    shown = df.copy()
    shown["runtime_s"] = shown["runtime_s"].map(lambda z: f"{z:.4f}")
    shown["dx"] = shown["dx"].map(lambda z: f"{z:.4g}")
    shown["mass_final"] = shown["mass_final"].map(lambda z: f"{z:.6f}")
    table = ax.table(cellText=shown.values, colLabels=shown.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    ax.set_title("Performance table")
    fig.savefig(os.path.join(cfg.outdir, "figure4_performance_table.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(df.to_string(index=False))
    return df


def experiment_method_comparison(cfg: RunConfig) -> None:
    """Additional method comparison plot required by visualization section."""

    print_section("Method comparison plot")
    n = 512 if not cfg.quick else 384
    x, dx = grid(-30.0, 30.0, n)
    dt, t_end = 0.002, 1.5
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    psi0 = gaussian_wavepacket(x, -8.0, 1.2, 2.0, dx)
    v = potential_harmonic(x, omega=0.1)
    methods = ["FTCS", "Backward-Euler", "Crank-Nicolson", "Split-Step-FFT", "RK4"]

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for method in methods:
        _, hist = solve(method, psi0, v, x, t, dx, dt, store_every=len(t) - 1)
        psi = hist[-1]
        if np.max(np.abs(psi)) > 5:
            psi = psi / np.max(np.abs(psi)) * np.sqrt(probability_mass(psi0, dx))
        ax.plot(x, np.abs(psi) ** 2, lw=1.2, label=method)
    ax.set_xlim(-15, 5)
    ax.set_xlabel("x")
    ax.set_ylabel("|psi|^2")
    ax.set_title("Method comparison from same initial condition")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(cfg.outdir, "method_comparison.png"), dpi=cfg.dpi)
    plt.close(fig)


# =============================================================================
# Animations
# =============================================================================


def save_wavepacket_animation(cfg: RunConfig) -> None:
    """Figure 5: free Gaussian wavepacket animation."""

    print_section("Figure 5: wavepacket animation")
    n = 512 if not cfg.quick else 384
    x, dx = grid(-30.0, 30.0, n)
    dt, t_end = 0.004, 3.0
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    psi0 = gaussian_wavepacket(x, -10.0, 1.1, 2.0, dx)
    v = potential_free(x)
    saved_t, hist = solve("Split-Step-FFT", psi0, v, x, t, dx, dt, store_every=max(1, len(t) // 140))

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    line, = ax.plot([], [], lw=2, label="Split-Step FFT")
    exact_line, = ax.plot([], [], "k--", lw=1, label="exact")
    ax.set_xlim(-20, 8)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("x")
    ax.set_ylabel("|psi|^2")
    ax.set_title("Figure 5: free wavepacket propagation")
    ax.legend(fontsize=8)

    def init():
        line.set_data([], [])
        exact_line.set_data([], [])
        return line, exact_line

    def update(frame: int):
        tt = saved_t[frame]
        line.set_data(x, np.abs(hist[frame]) ** 2)
        exact = exact_free_gaussian(x, tt, -10.0, 1.1, 2.0, dx)
        exact_line.set_data(x, np.abs(exact) ** 2)
        ax.set_title(f"Figure 5: free wavepacket propagation, t={tt:.2f}")
        return line, exact_line

    if cfg.save_gif:
        ani = FuncAnimation(fig, update, frames=len(saved_t), init_func=init, blit=True)
        ani.save(os.path.join(cfg.outdir, "figure5_wavepacket_animation.gif"), writer=PillowWriter(fps=24), dpi=cfg.dpi)
    fig.savefig(os.path.join(cfg.outdir, "figure5_wavepacket_last_frame.png"), dpi=cfg.dpi)
    plt.close(fig)


def save_tunneling_animation(cfg: RunConfig, x: Array, v: Array, saved_t: Array, hist: Array, label: str) -> None:
    """Animated rectangular barrier tunneling for one representative case."""

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    line, = ax.plot([], [], lw=2)
    barrier = v / max(np.max(v), 1e-12) * 0.08
    ax.fill_between(x, 0, barrier, color="tab:red", alpha=0.3)
    ax.set_xlim(-95, 95)
    ax.set_ylim(0, max(0.18, np.max(np.abs(hist) ** 2) * 1.15))
    ax.set_xlabel("x")
    ax.set_ylabel("|psi|^2")
    ax.set_title(f"Figure 6: tunneling animation {label}")

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame: int):
        line.set_data(x, np.abs(hist[frame]) ** 2)
        ax.set_title(f"Figure 6: tunneling animation {label}, t={saved_t[frame]:.2f}")
        return (line,)

    ani = FuncAnimation(fig, update, frames=len(saved_t), init_func=init, blit=True)
    ani.save(os.path.join(cfg.outdir, "figure6_tunneling_animation.gif"), writer=PillowWriter(fps=24), dpi=cfg.dpi)
    plt.close(fig)


# =============================================================================
# Main driver
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="1D TDSE numerical PDE demo in dimensionless units.")
    parser.add_argument("--outdir", default="tdse_outputs", help="Output directory.")
    parser.add_argument("--quick", action="store_true", help="Run a smaller/faster version.")
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation.")
    args = parser.parse_args()

    cfg = RunConfig(outdir=args.outdir, quick=args.quick, save_gif=not args.no_gif)
    ensure_outdir(cfg.outdir)
    warnings.filterwarnings("ignore", category=UserWarning)

    print_section("Dimensionless 1D Time-Dependent Schrodinger Equation Demo")
    print("Equation: i psi_t = -1/2 psi_xx + V(x) psi")
    print("All quantities are dimensionless.")
    print(f"Output directory: {os.path.abspath(cfg.outdir)}")
    print("\nNumerical stability reminders:")
    print("- FTCS is included because it is requested, but it is intrinsically unstable for TDSE.")
    print("- Crank-Nicolson uses scipy.linalg.solve_banded on a tridiagonal system.")
    print("- Split-Step Fourier uses numpy.fft and assumes periodic spectral propagation.")
    print("- RK4 + centered differences is conditionally stable; keep dt/dx^2 small.")

    total_start = time.perf_counter()
    experiment_analytic_vs_numerical(cfg)
    experiment_convergence(cfg)
    experiment_stability(cfg)
    experiment_performance(cfg)
    experiment_method_comparison(cfg)
    if cfg.save_gif:
        save_wavepacket_animation(cfg)
    else:
        print_section("Figure 5: GIF skipped by --no-gif")
    experiment_tunneling(cfg)
    total_runtime = time.perf_counter() - total_start

    print_section("Done")
    print(f"Total runtime: {total_runtime:.2f} s")
    print("Generated files:")
    for name in sorted(os.listdir(cfg.outdir)):
        print(f"- {name}")


if __name__ == "__main__":
    main()
