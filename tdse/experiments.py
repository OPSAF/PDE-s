"""
TDSE Experiments Module
=======================

Pre-built experiment suite for 1D and 2D time-dependent Schrödinger equation
simulations. Each experiment function takes a RunConfig and produces standardized
outputs (figures, CSV data, and optional animations) in the configured output
directory.

All functions import from the tdse package's submodules (config, potentials,
solvers, visualization) — no inline implementations are duplicated from demo2.py.

Usage:
    from tdse.config import RunConfig
    from tdse.experiments import experiment_analytic_vs_numerical

    cfg = RunConfig(outdir="results", quick=True, save_gif=False)
    df = experiment_analytic_vs_numerical(cfg)
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

# Relative imports from sibling modules
from .config import RunConfig
from .potentials import (
    Array,
    grid,
    make_2d_grid,
    normalize,
    probability_mass,
    mass_2d,
    l1_l2_linf_error,
    align_global_phase,
    dt_warning,
    gaussian_wavepacket,
    gaussian_wavepacket_2d,
    exact_free_gaussian,
    exact_free_gaussian_2d,
    potential_free,
    potential_harmonic,
    potential_rect_barrier,
    potential_circle_2d,
    potential_waveguide_2d,
    absorbing_potential_2d,
    infinite_well_eigenstate,
    harmonic_eigenstate_low,
    transmission_reflection,
    print_section,
)
from .solvers import (
    solve,
    step_split_step_fft_2d,
)
from .visualization import (
    save_tunneling_animation,
    save_2d_animation,
)


# =============================================================================
# 1D Experiments
# =============================================================================


def experiment_analytic_vs_numerical(cfg: RunConfig) -> pd.DataFrame:
    """Figure 1: analytic vs numerical comparison across five 1D methods.

    Compares free Gaussian wave packet propagation using FTCS, Backward Euler,
    Crank-Nicolson, Split-Step FFT, and RK4 against the exact analytic solution.
    Also verifies infinite well and harmonic oscillator eigenstate evolution.
    """
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    psi_exact = exact_free_gaussian(x, t_end, x0, sigma, k0, dx)
    axes[0, 0].plot(x, np.abs(psi_exact) ** 2, "k--", lw=2.5, label="Exact", alpha=0.8)

    for method in tqdm(methods, desc="analytic comparison"):
        start = time.perf_counter()
        _, psi_hist = solve(method, psi0, v, x, t, dx, dt, store_every=len(t) - 1)
        runtime = time.perf_counter() - start
        psi_num = psi_hist[-1]
        l1, l2, linf = l1_l2_linf_error(psi_num, psi_exact, dx)
        mass = probability_mass(psi_num, dx)
        rows.append({
            "method": method, "N": n, "dx": dx, "dt": dt,
            "runtime_s": runtime, "L1": l1, "L2": l2, "Linf": linf, "mass": mass,
        })
        if method != "FTCS":
            axes[0, 0].plot(x, np.abs(psi_num) ** 2, lw=1.8, label=method)

    axes[0, 0].set_title("Free Gaussian: |ψ|² at Final Time", fontweight='bold', pad=10)
    axes[0, 0].set_xlabel("Position x", labelpad=8)
    axes[0, 0].set_ylabel("Probability Density |ψ|²", labelpad=8)
    axes[0, 0].legend(fontsize=9, loc='upper right', framealpha=0.95)
    axes[0, 0].set_xlim(-15, 5)
    axes[0, 0].grid(True, alpha=0.3)

    l2_values = [r["L2"] for r in rows]
    colors = ['#C73E1D', '#A23B72', '#2E86AB', '#F18F01', '#3B1F2B']
    axes[0, 1].bar([r["method"] for r in rows], l2_values, color=colors,
                   alpha=0.8, edgecolor='white', linewidth=0.5)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("L₂ Error vs Exact Gaussian", fontweight='bold', pad=10)
    axes[0, 1].set_xlabel("Numerical Method", labelpad=8)
    axes[0, 1].set_ylabel("L₂ Error (log scale)", labelpad=8)
    axes[0, 1].tick_params(axis="x", rotation=35)

    # Infinite well eigenstate verification
    xw, dxw = grid(-5.0, 5.0, n)
    psi_well0, e_well = infinite_well_eigenstate(xw, n=2, a=-5.0, b=5.0)
    tw_end, dtw = 1.5, 0.0025
    tw = np.arange(0.0, tw_end + 0.5 * dtw, dtw)
    vw = np.zeros_like(xw)
    _, psi_well_hist = solve("Crank-Nicolson", psi_well0, vw, xw, tw, dxw, dtw,
                             store_every=len(tw) - 1)
    psi_well_exact = psi_well0 * np.exp(-1j * e_well * tw_end)
    axes[1, 0].plot(xw, np.abs(psi_well_exact) ** 2, "k--", lw=2.5,
                    label="Exact n=2", alpha=0.8)
    axes[1, 0].plot(xw, np.abs(psi_well_hist[-1]) ** 2, lw=1.8,
                    label="Crank-Nicolson")
    axes[1, 0].set_title("Infinite Well Eigenstate (n=2)", fontweight='bold', pad=10)
    axes[1, 0].set_xlabel("Position x", labelpad=8)
    axes[1, 0].set_ylabel("Probability Density |ψ|²", labelpad=8)
    axes[1, 0].legend(fontsize=9, loc='upper right', framealpha=0.95)

    # Harmonic oscillator eigenstate verification
    xh, dxh = grid(-10.0, 10.0, n)
    omega = 1.0
    psi_h0, e_h = harmonic_eigenstate_low(xh, n=1, omega=omega, dx=dxh)
    vh = potential_harmonic(xh, omega)
    th_end, dth = 1.0, 0.002
    th = np.arange(0.0, th_end + 0.5 * dth, dth)
    _, psi_h_hist = solve("Crank-Nicolson", psi_h0, vh, xh, th, dxh, dth,
                          store_every=len(th) - 1)
    psi_h_exact = psi_h0 * np.exp(-1j * e_h * th_end)
    axes[1, 1].plot(xh, np.abs(psi_h_exact) ** 2, "k--", lw=2.5,
                    label="Exact n=1", alpha=0.8)
    axes[1, 1].plot(xh, np.abs(psi_h_hist[-1]) ** 2, lw=1.8,
                    label="Crank-Nicolson")
    axes[1, 1].set_title("Harmonic Oscillator (n=1)", fontweight='bold', pad=10)
    axes[1, 1].set_xlabel("Position x", labelpad=8)
    axes[1, 1].set_ylabel("Probability Density |ψ|²", labelpad=8)
    axes[1, 1].legend(fontsize=9, loc='upper right', framealpha=0.95)

    fig.suptitle("Figure 1: Analytical vs Numerical Solutions",
                 fontsize=16, fontweight='bold', y=1.02)
    fig.savefig(os.path.join(cfg.outdir, "figure1_analytic_vs_numerical.png"),
                dpi=cfg.dpi)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "analytic_error_table.csv"), index=False)
    print(df.to_string(index=False))
    return df


def experiment_convergence(cfg: RunConfig) -> pd.DataFrame:
    """Figure 2: error vs dx convergence with fitted log-log convergence order."""
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
            rows.append({"method": method, "N": n, "dx": dx, "dt": dt,
                         "L1": l1, "L2": l2, "Linf": linf})

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

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for method in methods:
        sub = df[df["method"] == method].sort_values("dx")
        p = order_df[order_df.method == method].fitted_L2_order.iloc[0]
        label = f"{method}, p={p:.2f}"
        ax.loglog(sub["dx"], sub["L2"], "o-", label=label, markersize=8, linewidth=2)
    ax.invert_xaxis()
    ax.set_xlabel("Grid Spacing Δx", labelpad=10)
    ax.set_ylabel("L₂ Error", labelpad=10)
    ax.set_title("Error Convergence vs Exact Free Gaussian", fontweight='bold', pad=15)
    ax.grid(True, which="both", alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
    fig.savefig(os.path.join(cfg.outdir, "figure2_error_convergence.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(df.to_string(index=False))
    print("\nFitted log-log convergence orders:")
    print(order_df.to_string(index=False))
    return df


def experiment_stability(cfg: RunConfig) -> pd.DataFrame:
    """Figure 3: stability map across grid sizes and time steps for five methods."""
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
        for dt_val in dts:
            t = np.arange(0.0, t_end + 0.5 * dt_val, dt_val)
            for method in methods:
                try:
                    _, hist = solve(method, psi0, v, x, t, dx, dt_val,
                                    store_every=len(t) - 1)
                    mass = probability_mass(hist[-1], dx)
                    max_amp = float(np.max(np.abs(hist[-1])))
                    stable = np.isfinite(mass) and 0.2 < mass < 1.8 and max_amp < 8.0
                except Exception:
                    mass = np.nan
                    max_amp = np.nan
                    stable = False
                rows.append({
                    "method": method, "N": n, "dx": dx, "dt": dt_val,
                    "r_dt_over_dx2": dt_val / dx**2,
                    "mass": mass, "max_amp": max_amp, "stable": stable,
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "stability_scan.csv"), index=False)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    axes_flat = axes.flatten()
    summary = df.groupby("method")["stable"].agg(["sum", "count"])
    summary["verdict"] = np.where(
        summary["sum"] == summary["count"],
        "stable in scan", "conditional/divergent cases"
    )

    for idx, (ax, method) in enumerate(zip(axes_flat, methods)):
        sub = df[df["method"] == method]
        pivot = sub.pivot(index="dt", columns="N", values="stable").astype(int)
        im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f"{method}", fontweight='bold', fontsize=12, pad=10)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:g}" for v in pivot.index], fontsize=10)
        ax.set_xlabel("Grid Points N", labelpad=8)
        ax.set_ylabel("Time Step Δt", labelpad=8)

    ax_summary = axes_flat[-1]
    stable_counts = summary["sum"].values
    bar_colors = ['#C73E1D' if s < 9 else '#2E86AB' for s in stable_counts]
    ax_summary.bar(range(len(methods)), stable_counts, color=bar_colors,
                   edgecolor='white', linewidth=1.5)
    ax_summary.set_xticks(range(len(methods)))
    ax_summary.set_xticklabels(methods, rotation=35, ha='right', fontsize=10)
    ax_summary.set_ylabel("Stable Configurations", labelpad=8)
    ax_summary.set_title("Stability Summary", fontweight='bold', fontsize=12, pad=10)
    ax_summary.set_ylim(0, 10)
    ax_summary.grid(True, alpha=0.3, axis='y')

    cbar = fig.colorbar(im, ax=axes_flat[:-1],
                        label="Stable (1) / Divergent (0)", shrink=0.75, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    fig.suptitle("Figure 3: Stability Map Across Methods",
                 fontsize=16, fontweight='bold', y=0.99)
    fig.savefig(os.path.join(cfg.outdir, "figure3_stability_map.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(summary.to_string())
    print("\nStability notes:")
    for method in methods:
        print(f"- {method}: {dt_warning(method, 0.2, 0.001)}")
    return df


def experiment_tunneling(cfg: RunConfig) -> pd.DataFrame:
    """Figure 6: quantum tunneling through rectangular barrier with T/R analysis."""
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

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes_flat = axes.flatten()
    animation_hist = None
    animation_t = None
    animation_label = None
    results = []

    for ax, (label, k0) in zip(axes_flat[:3], cases):
        psi0 = gaussian_wavepacket(x, x0, sigma, k0, dx)
        energy = 0.5 * k0**2
        store_every = max(1, len(t) // 220)
        saved_t, hist = solve("Split-Step-FFT", psi0, v, x, t, dx, dt,
                              store_every=store_every)
        psi_final = hist[-1]
        tprob, rprob = transmission_reflection(psi_final, x, dx,
                                               barrier_center=0.0, buffer=0.0)
        rows.append({"case": label, "E": energy, "V0": v0,
                     "T": tprob, "R": rprob, "R_plus_T": rprob + tprob})
        results.append({"label": label, "T": tprob, "R": rprob})

        ax.plot(x, np.abs(psi0) ** 2, color="gray", lw=1.2, label="Initial",
                alpha=0.7, linestyle='--')
        ax.plot(x, np.abs(psi_final) ** 2, lw=2.0, label="Final", color='#2E86AB')
        ax.fill_between(x, 0, v / max(v0, 1e-12) * 0.05,
                        color="#C73E1D", alpha=0.3, label=f"Barrier V₀={v0}")
        ax.set_xlim(-95, 95)
        ax.set_ylabel("Probability Density |ψ|²", labelpad=8)
        ax.set_title(f"{label}: E={energy:.2f}, T={tprob:.3f}, R={rprob:.3f}",
                     fontweight='bold', fontsize=11, pad=8)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        if label == "E<V0":
            animation_hist = hist
            animation_t = saved_t
            animation_label = label

    ax_summary = axes_flat[-1]
    labels = [r["label"] for r in results]
    T_vals = [r["T"] for r in results]
    R_vals = [r["R"] for r in results]
    x_pos = np.arange(len(labels))
    width = 0.35
    ax_summary.bar(x_pos - width/2, T_vals, width, label='Transmission',
                   color='#2E86AB', edgecolor='white')
    ax_summary.bar(x_pos + width/2, R_vals, width, label='Reflection',
                   color='#A23B72', edgecolor='white')
    ax_summary.set_xticks(x_pos)
    ax_summary.set_xticklabels(labels, fontsize=10)
    ax_summary.set_ylabel("Probability", labelpad=8)
    ax_summary.set_title("Transmission vs Reflection Comparison",
                         fontweight='bold', fontsize=12, pad=10)
    ax_summary.legend(fontsize=9)
    ax_summary.set_ylim(0, 1.1)
    ax_summary.grid(True, alpha=0.3, axis='y')

    axes_flat[2].set_xlabel("Position x", labelpad=8)
    axes_flat[3].set_xlabel("Case", labelpad=8)
    fig.suptitle("Figure 6: Quantum Tunneling Through Rectangular Barrier",
                 fontsize=15, fontweight='bold', y=0.98)
    fig.savefig(os.path.join(cfg.outdir, "figure6_tunneling_simulation.png"),
                dpi=cfg.dpi)
    plt.close(fig)

    if cfg.save_gif and animation_hist is not None:
        save_tunneling_animation(cfg, x, v, animation_t,
                                 animation_hist, animation_label)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "tunneling_RT.csv"), index=False)
    print(df.to_string(index=False))
    return df


def experiment_performance(cfg: RunConfig) -> pd.DataFrame:
    """Figure 4: runtime comparison table for FTCS, CN, and Split-Step FFT."""
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
            _, hist = solve(method, psi0, v, x, t, dx, dt,
                            store_every=len(t) - 1)
            runtime = time.perf_counter() - start
            rows.append({
                "method": method, "grid_size": n, "dx": dx, "dt": dt,
                "steps": len(t) - 1, "runtime_s": runtime,
                "mass_final": probability_mass(hist[-1], dx),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "performance_table.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    ax.axis("off")
    shown = df.copy()
    shown["runtime_s"] = shown["runtime_s"].map(lambda z: f"{z:.4f}")
    shown["dx"] = shown["dx"].map(lambda z: f"{z:.4g}")
    shown["mass_final"] = shown["mass_final"].map(lambda z: f"{z:.6f}")
    table = ax.table(cellText=shown.values, colLabels=shown.columns,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    ax.set_title("Performance table")
    fig.savefig(os.path.join(cfg.outdir, "figure4_performance_table.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(df.to_string(index=False))
    return df


def experiment_method_comparison(cfg: RunConfig) -> None:
    """Method comparison plot: all five 1D methods from same initial condition."""
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
# 2D Experiments
# =============================================================================


def experiment_2d_free_propagation(cfg: RunConfig) -> None:
    """Figure 7: 2D free Gaussian wave packet propagation and mass conservation."""
    print_section("Figure 7: 2D free Gaussian propagation (Split-Step FFT)")
    nx = ny = 96 if cfg.quick else 128
    X, Y, x, y, dx, dy, KX, KY = make_2d_grid(nx, ny, -15.0, 15.0, -15.0, 15.0)

    psi = gaussian_wavepacket_2d(
        X, Y, x0=-5.0, y0=-2.0, sigma=1.2, kx0=2.0, ky0=1.0, dx=dx, dy=dy
    )
    V = np.zeros_like(X)
    dt = 0.01
    n_steps = 150 if cfg.quick else 200
    snap_at = {0, n_steps // 2, n_steps}

    psi_cur = psi.copy()
    snaps: List[Tuple[float, Array]] = []
    mass_hist: List[float] = []
    t_hist: List[float] = []

    for step in range(n_steps + 1):
        if step in snap_at:
            snaps.append((step * dt, psi_cur.copy()))
        if step % 5 == 0:
            mass_hist.append(mass_2d(psi_cur, dx, dy))
            t_hist.append(step * dt)
        if step < n_steps:
            psi_cur = step_split_step_fft_2d(psi_cur, V, KX, KY, dt)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes_flat = axes.flatten()

    for idx, (t_snap, psi_snap) in enumerate(snaps):
        ax = axes_flat[idx]
        im = ax.imshow(
            np.abs(psi_snap).T ** 2, origin="lower", aspect="equal",
            extent=[x[0], x[-1], y[0], y[-1]], cmap="inferno",
            interpolation='bilinear'
        )
        cbar = plt.colorbar(im, ax=ax, shrink=0.82, label="|ψ|²", pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        ax.set_title(f"Time t = {t_snap:.2f}", fontweight='bold', fontsize=12, pad=10)
        ax.set_xlabel("x", labelpad=8)
        ax.set_ylabel("y", labelpad=8)
        ax.tick_params(labelsize=10)

    ax_mass = axes_flat[-1]
    ax_mass.plot(t_hist, mass_hist, lw=2.0, color='#2E86AB',
                 marker='o', markersize=4, markevery=5)
    ax_mass.axhline(y=1.0, color='#C73E1D', linestyle='--', lw=1.5, alpha=0.7,
                    label='Expected (M=1)')
    ax_mass.set_xlabel("Time t", labelpad=10)
    ax_mass.set_ylabel("Total Mass ∫|ψ|² dxdy", labelpad=10)
    ax_mass.set_title("Mass Conservation", fontweight='bold', fontsize=12, pad=10)
    ax_mass.set_ylim(0.95, 1.05)
    ax_mass.legend(fontsize=10, loc='best', framealpha=0.95)
    ax_mass.grid(True, alpha=0.3)

    fig.suptitle("Figure 7: 2D Free Gaussian Wave Packet (Split-Step FFT)",
                 fontsize=15, fontweight='bold', y=0.98)
    fig.savefig(os.path.join(cfg.outdir, "figure7_2d_free_propagation.png"),
                dpi=cfg.dpi)
    plt.close(fig)

    print(f"  Initial mass={mass_hist[0]:.6f}  Final mass={mass_hist[-1]:.6f}")


def experiment_2d_circular_obstacle_with_animation(cfg: RunConfig) -> None:
    """Figure 8: 2D circular obstacle scattering with absorbing boundaries."""
    print_section("2D Circular Obstacle (with Animation)")
    nx = ny = 96 if cfg.quick else 128
    X, Y, x, y, dx, dy, KX, KY = make_2d_grid(nx, ny, -14.0, 14.0, -10.0, 10.0)

    kx0 = 3.0
    psi0 = gaussian_wavepacket_2d(
        X, Y, x0=-7.0, y0=0.0, sigma=1.2, kx0=kx0, ky0=0.0, dx=dx, dy=dy
    )
    V = potential_circle_2d(X, Y, xc=2.0, yc=0.0, R=1.5, v0=20.0).astype(complex)
    V += absorbing_potential_2d(X, Y, width=3.0, strength=2.0)

    dt = 0.01
    n_steps = 250 if cfg.quick else 350
    store_every = max(1, n_steps // 80)
    saved_t = []
    saved_psi = []

    psi_cur = psi0.copy()
    for step in range(n_steps + 1):
        if step % store_every == 0 or step == n_steps:
            saved_t.append(step * dt)
            saved_psi.append(psi_cur.copy())
        if step < n_steps:
            psi_cur = step_split_step_fft_2d(psi_cur, V, KX, KY, dt)

    t_approach = 1.5
    t_scatter = 3.5
    step_approach = min(int(round(t_approach / dt)), n_steps)
    step_scatter = min(int(round(t_scatter / dt)), n_steps)

    def find_closest_idx(target_step):
        for i, t_val in enumerate(saved_t):
            if t_val >= target_step * dt:
                return i
        return len(saved_t) - 1

    idx0 = 0
    idx_approach = find_closest_idx(step_approach)
    idx_scatter = find_closest_idx(step_scatter)
    indices = [idx0, idx_approach, idx_scatter]
    titles = {
        0: "Initial (t=0)",
        step_approach: f"Approaching (t={t_approach:.1f})",
        step_scatter: f"Scattered (t={t_scatter:.1f})",
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes_flat = axes.flatten()
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    cx, cy, cr = 2.0, 0.0, 1.5

    for idx, (ax, i) in enumerate(zip(axes_flat[:3], indices)):
        im = ax.imshow(
            np.abs(saved_psi[i]).T ** 2, origin="lower", aspect="equal",
            extent=[x[0], x[-1], y[0], y[-1]], cmap="inferno",
            interpolation='bilinear'
        )
        cbar = plt.colorbar(im, ax=ax, shrink=0.82, label="|ψ|²", pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        ax.plot(cx + cr * np.cos(theta), cy + cr * np.sin(theta),
                "w--", lw=2.5)
        title_key = 0 if i == idx0 else (step_approach if i == idx_approach else step_scatter)
        ax.set_title(titles.get(title_key, f"t={saved_t[i]:.1f}"),
                     fontweight='bold', fontsize=13, pad=10)
        ax.set_xlabel("x", labelpad=8)
        ax.set_ylabel("y", labelpad=8)
        ax.tick_params(labelsize=10)

    ax_scatter = axes_flat[-1]
    psi_final = saved_psi[idx_scatter]
    radial_dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = radial_dist > cr + 0.5
    angles = np.arctan2(Y - cy, X - cx)
    hist_vals, bins = np.histogram(
        angles[mask], bins=36, weights=np.abs(psi_final[mask])**2, density=True
    )
    ax_scatter.plot(bins[:-1], hist_vals, lw=2.5, color='#F18F01')
    ax_scatter.fill_between(bins[:-1], hist_vals, alpha=0.3, color='#F18F01')
    ax_scatter.set_xlabel("Scattering Angle (rad)", labelpad=8)
    ax_scatter.set_ylabel("Normalized Intensity", labelpad=8)
    ax_scatter.set_title("Scattering Angular Distribution",
                         fontweight='bold', fontsize=12, pad=10)
    ax_scatter.set_xlim(-np.pi, np.pi)
    ax_scatter.grid(True, alpha=0.3)

    fig.suptitle("Figure 8: 2D Circular Obstacle Scattering with Absorbing Boundaries",
                 fontsize=14, fontweight='bold', y=0.98)
    fig.savefig(os.path.join(cfg.outdir, "figure8_2d_circular_obstacle_absorbing.png"),
                dpi=cfg.dpi)
    plt.close(fig)

    if cfg.save_gif:
        save_2d_animation(
            cfg, X, Y, saved_t, saved_psi,
            "figure8_circular_obstacle_animation.gif",
            "2D Circular Obstacle Scattering",
            obstacle_plot=(cx, cy, cr)
        )

    print(f"  Final mass={mass_2d(psi_cur, dx, dy):.6f}")


def experiment_2d_waveguide(cfg: RunConfig) -> None:
    """Figure 9: 2D waveguide vs free propagation with beam profile analysis."""
    print_section("Figure 9: 2D waveguide vs free propagation (Split-Step FFT)")
    nx = ny = 96 if cfg.quick else 128
    X, Y, x, y, dx, dy, KX, KY = make_2d_grid(nx, ny, -10.0, 16.0, -8.0, 8.0)

    psi0 = gaussian_wavepacket_2d(
        X, Y, x0=-6.0, y0=0.0, sigma=1.0, kx0=2.5, ky0=0.0, dx=dx, dy=dy
    )
    V_free = np.zeros_like(X)
    V_guide = potential_waveguide_2d(X, Y, alpha=0.4)
    dt = 0.01
    n_steps = 180 if cfg.quick else 250
    store_every = max(1, n_steps // 80)
    saved_t = []
    saved_psi_guide = []

    psi_free = psi0.copy()
    psi_guide = psi0.copy()
    for step in range(n_steps + 1):
        if step % store_every == 0 or step == n_steps:
            saved_t.append(step * dt)
            saved_psi_guide.append(psi_guide.copy())
        if step < n_steps:
            psi_free = step_split_step_fft_2d(psi_free, V_free, KX, KY, dt)
            psi_guide = step_split_step_fft_2d(psi_guide, V_guide, KX, KY, dt)

    t_final = n_steps * dt
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    cases = [
        ("Free Propagation", psi_free),
        (f"Waveguide V = {0.4}y²", psi_guide),
    ]
    for col, (label, psi_plot) in enumerate(cases):
        im = axes[0, col].imshow(
            np.abs(psi_plot).T ** 2, origin="lower", aspect="equal",
            extent=[x[0], x[-1], y[0], y[-1]], cmap="inferno",
            interpolation='bilinear'
        )
        cbar = plt.colorbar(im, ax=axes[0, col], shrink=0.82, label="|ψ|²", pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        axes[0, col].set_title(f"{label}\n|ψ|² at t = {t_final:.1f}",
                               fontweight='bold', fontsize=11, pad=8)
        axes[0, col].set_xlabel("x", labelpad=6)
        axes[0, col].set_ylabel("y", labelpad=6)

        marginal_y = np.sum(np.abs(psi_plot) ** 2, axis=0) * dx
        axes[1, col].fill_between(y, marginal_y, alpha=0.3, color='#2E86AB')
        axes[1, col].plot(y, marginal_y, lw=2.0, color='#2E86AB')
        axes[1, col].set_xlabel("Transverse Position y", labelpad=8)
        axes[1, col].set_ylabel("Marginal ∫|ψ|² dx", labelpad=8)
        axes[1, col].set_title(f"{label}: Beam Profile",
                               fontweight='bold', fontsize=11, pad=8)
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle("Figure 9: 2D Waveguide vs Free Propagation (Split-Step FFT)",
                 fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(os.path.join(cfg.outdir, "figure9_2d_waveguide.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(f"  Free mass={mass_2d(psi_free, dx, dy):.6f}  "
          f"Waveguide mass={mass_2d(psi_guide, dx, dy):.6f}")

    if cfg.save_gif:
        save_2d_animation(
            cfg, X, Y, saved_t, saved_psi_guide,
            "figure9_waveguide_animation.gif",
            "2D Waveguide Propagation"
        )


# =============================================================================
# Analysis Experiments
# =============================================================================


def experiment_circular_obstacle_radius_sweep(cfg: RunConfig) -> pd.DataFrame:
    """Parameter sweep: circular obstacle radius vs transmitted/scattered mass."""
    print_section("Parameter Sweep: Circular Obstacle Radius")
    nx = ny = 80 if cfg.quick else 100
    X, Y, x, y, dx, dy, KX, KY = make_2d_grid(nx, ny, -14.0, 14.0, -10.0, 10.0)
    kx0 = 3.0
    psi0 = gaussian_wavepacket_2d(
        X, Y, x0=-7.0, y0=0.0, sigma=1.2, kx0=kx0, ky0=0.0, dx=dx, dy=dy
    )
    dt = 0.01
    n_steps = 150 if cfg.quick else 200
    radii = [0.5, 1.0, 1.5, 2.0, 2.5]
    rows = []

    for R in radii:
        V = potential_circle_2d(X, Y, xc=2.0, yc=0.0, R=R, v0=20.0)
        psi = psi0.copy()
        for _ in range(n_steps):
            psi = step_split_step_fft_2d(psi, V, KX, KY, dt)

        transmitted_mask = X > 6.0
        scattered_mask = ((X > -2.0) & (X < 6.0) &
                          (np.sqrt((X - 2.0)**2 + (Y - 0.0)**2) > R + 0.5))
        transmitted = mass_2d(psi * transmitted_mask.astype(float), dx, dy)
        scattered = mass_2d(psi * scattered_mask.astype(float), dx, dy)
        total = mass_2d(psi, dx, dy)

        rows.append({
            "radius": R,
            "transmitted_mass": transmitted,
            "scattered_mass": scattered,
            "total_mass": total,
        })
        print(f"  R={R:.2f}: transmitted={transmitted:.4f}, "
              f"scattered={scattered:.4f}, total={total:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "circular_obstacle_sweep.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(df["radius"], df["transmitted_mass"], "o-",
            label="Transmitted Mass", linewidth=2.5, markersize=10, color='#2E86AB')
    ax.plot(df["radius"], df["scattered_mass"], "s-",
            label="Scattered Mass", linewidth=2.5, markersize=10, color='#C73E1D')
    ax.set_xlabel("Obstacle Radius R", labelpad=10)
    ax.set_ylabel("Probability Mass", labelpad=10)
    ax.set_title("Circular Obstacle Radius vs Transmitted/Scattered Mass",
                 fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)
    fig.savefig(os.path.join(cfg.outdir, "figure_circular_sweep.png"), dpi=cfg.dpi)
    plt.close(fig)

    return df


def experiment_waveguide_strength_sweep(cfg: RunConfig) -> pd.DataFrame:
    """Parameter sweep: waveguide strength (alpha) vs beam spreading."""
    print_section("Parameter Sweep: Waveguide Strength")
    nx = ny = 80 if cfg.quick else 100
    X, Y, x, y, dx, dy, KX, KY = make_2d_grid(nx, ny, -10.0, 16.0, -8.0, 8.0)
    psi0 = gaussian_wavepacket_2d(
        X, Y, x0=-6.0, y0=0.0, sigma=1.0, kx0=2.5, ky0=0.0, dx=dx, dy=dy
    )
    dt = 0.01
    n_steps = 150 if cfg.quick else 200
    alphas = [0.1, 0.2, 0.4, 0.6, 1.0]
    rows = []

    for alpha in alphas:
        V = potential_waveguide_2d(X, Y, alpha=alpha)
        psi = psi0.copy()
        for _ in range(n_steps):
            psi = step_split_step_fft_2d(psi, V, KX, KY, dt)

        marginal_y = np.sum(np.abs(psi) ** 2, axis=0) * dx
        mean_y = np.sum(y * marginal_y) * dy
        std_y = np.sqrt(np.sum((y - mean_y)**2 * marginal_y) * dy)
        rows.append({"alpha": alpha, "beam_width": std_y})
        print(f"  alpha={alpha:.2f}: beam_width={std_y:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "waveguide_sweep.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.plot(df["alpha"], df["beam_width"], "o-", linewidth=2.5, markersize=10,
            color='#F18F01', markeredgecolor='white', markeredgewidth=1.5)
    ax.set_xlabel("Waveguide Strength α", labelpad=10)
    ax.set_ylabel("Beam Width σ_y (Standard Deviation)", labelpad=10)
    ax.set_title("Waveguide Strength vs Beam Spreading", fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)
    fig.savefig(os.path.join(cfg.outdir, "figure_waveguide_sweep.png"), dpi=cfg.dpi)
    plt.close(fig)

    return df


def experiment_1d_conservation_analysis(cfg: RunConfig) -> None:
    """Conservation analysis: mass error vs time for FTCS, CN, and Split-Step FFT."""
    print_section("Conservation Analysis: Mass Error vs Time")
    n = 256 if cfg.quick else 384
    x, dx = grid(-30.0, 30.0, n)
    t_end = 1.0
    dt = 0.002
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    psi0 = gaussian_wavepacket(x, -8.0, 1.2, 2.0, dx)
    v = potential_free(x)
    methods = ["FTCS", "Crank-Nicolson", "Split-Step-FFT"]

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    for method in methods:
        saved_t, saved_psi = solve(method, psi0, v, x, t, dx, dt)
        initial_mass = probability_mass(saved_psi[0], dx)
        mass_errors = [abs(probability_mass(psi, dx) - initial_mass)
                       for psi in saved_psi]
        ax.semilogy(saved_t, mass_errors, lw=2.5, label=method,
                    markersize=4, markevery=len(saved_t) // 10)

    ax.set_xlabel("Time t", labelpad=12)
    ax.set_ylabel("Mass Error |M(t) - M(0)|", labelpad=12)
    ax.set_title("Mass Conservation Analysis: Error vs Time",
                 fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.tick_params(labelsize=10)
    fig.savefig(os.path.join(cfg.outdir, "figure_conservation_analysis.png"),
                dpi=cfg.dpi)
    plt.close(fig)


def experiment_runtime_comparison(cfg: RunConfig) -> pd.DataFrame:
    """Runtime comparison table for FTCS, CN, and Split-Step FFT."""
    print_section("Runtime Comparison")
    n = 256 if cfg.quick else 384
    x, dx = grid(-30.0, 30.0, n)
    t_end = 1.0
    dt = 0.002
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    psi0 = gaussian_wavepacket(x, -8.0, 1.2, 2.0, dx)
    v = potential_free(x)
    methods = ["FTCS", "Crank-Nicolson", "Split-Step-FFT"]
    rows = []

    for method in methods:
        start = time.perf_counter()
        saved_t, saved_psi = solve(method, psi0, v, x, t, dx, dt)
        runtime = time.perf_counter() - start
        initial_mass = probability_mass(saved_psi[0], dx)
        final_mass = probability_mass(saved_psi[-1], dx)
        mass_error = abs(final_mass - initial_mass)
        rows.append({
            "method": method, "runtime_s": runtime,
            "initial_mass": initial_mass, "final_mass": final_mass,
            "mass_error": mass_error,
        })
        print(f"  {method}: runtime={runtime:.4f}s, mass_error={mass_error:.6e}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "runtime_comparison.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.axis("off")
    shown = df.copy()
    shown["runtime_s"] = shown["runtime_s"].map(lambda z: f"{z:.4f}")
    shown["initial_mass"] = shown["initial_mass"].map(lambda z: f"{z:.6f}")
    shown["final_mass"] = shown["final_mass"].map(lambda z: f"{z:.6f}")
    shown["mass_error"] = shown["mass_error"].map(lambda z: f"{z:.6e}")

    table = ax.table(
        cellText=shown.values,
        colLabels=["Method", "Runtime (s)", "Initial Mass", "Final Mass", "Mass Error"],
        cellLoc="center", loc="center",
        colColours=['#2E86AB'] * len(shown.columns),
        colWidths=[0.25, 0.18, 0.18, 0.18, 0.21],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#2E86AB')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#F5F5F5')

    ax.set_title("Performance Comparison Table", fontsize=14, fontweight='bold', pad=20)
    fig.savefig(os.path.join(cfg.outdir, "figure_runtime_comparison.png"), dpi=cfg.dpi)
    plt.close(fig)

    return df


def experiment_2d_error_heatmap(cfg: RunConfig) -> None:
    """2D error heatmap: numerical vs exact solution with pointwise error."""
    print_section("2D Error Heatmap")
    nx = ny = 80 if cfg.quick else 100
    X, Y, x, y, dx, dy, KX, KY = make_2d_grid(nx, ny, -15.0, 15.0, -15.0, 15.0)
    x0, y0, sigma_val, kx0, ky0 = -5.0, -2.0, 1.2, 2.0, 1.0
    psi0 = gaussian_wavepacket_2d(X, Y, x0, y0, sigma_val, kx0, ky0, dx, dy)
    V = np.zeros_like(X)
    dt = 0.01
    n_steps = 100 if cfg.quick else 150

    psi = psi0.copy()
    for _ in range(n_steps):
        psi = step_split_step_fft_2d(psi, V, KX, KY, dt)

    psi_exact = exact_free_gaussian_2d(
        X, Y, n_steps * dt, x0, y0, sigma_val, kx0, ky0, dx, dy
    )
    error = np.abs(psi - psi_exact)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    extent = [x[0], x[-1], y[0], y[-1]]

    im1 = axes[0].imshow(np.abs(psi).T ** 2, origin="lower", aspect="equal",
                         extent=extent, cmap="inferno", interpolation='bilinear')
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.82, label="|ψ|²")
    cbar1.ax.tick_params(labelsize=8)
    axes[0].set_title("Numerical Solution", fontweight='bold', fontsize=12, pad=10)

    im2 = axes[1].imshow(np.abs(psi_exact).T ** 2, origin="lower", aspect="equal",
                         extent=extent, cmap="inferno", interpolation='bilinear')
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.82, label="|ψ|²")
    cbar2.ax.tick_params(labelsize=8)
    axes[1].set_title("Exact Analytical Solution", fontweight='bold', fontsize=12, pad=10)

    im3 = axes[2].imshow(error.T, origin="lower", aspect="equal",
                         extent=extent, cmap="viridis", interpolation='bilinear')
    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.82, label="|ψ_num - ψ_exact|")
    cbar3.ax.tick_params(labelsize=8)
    axes[2].set_title("Absolute Error", fontweight='bold', fontsize=12, pad=10)

    for ax in axes:
        ax.set_xlabel("x", labelpad=8)
        ax.set_ylabel("y", labelpad=8)
        ax.tick_params(labelsize=9)

    fig.suptitle("Figure: 2D Numerical Error Analysis", fontsize=14, fontweight='bold',
                 y=1.02)
    fig.savefig(os.path.join(cfg.outdir, "figure_2d_error_heatmap.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(f"  Max error: {np.max(error):.6e}")


def experiment_2d_convergence(cfg: RunConfig) -> pd.DataFrame:
    """2D grid convergence study for Split-Step FFT vs exact Gaussian."""
    print_section("2D Convergence Study")
    grid_sizes = [40, 60, 80, 100] if not cfg.quick else [32, 48, 64]
    results = []

    for nx in tqdm(grid_sizes, desc="2D convergence"):
        ny = nx
        X, Y, x, y, dx, dy, KX, KY = make_2d_grid(
            nx, ny, -15.0, 15.0, -15.0, 15.0
        )
        x0, y0, sigma_val, kx0, ky0 = -5.0, -2.0, 1.2, 2.0, 1.0
        psi0 = gaussian_wavepacket_2d(X, Y, x0, y0, sigma_val, kx0, ky0, dx, dy)
        V = np.zeros_like(X)
        dt = 0.005
        n_steps = 100 if cfg.quick else 150
        t_final = n_steps * dt

        psi = psi0.copy()
        for _ in range(n_steps):
            psi = step_split_step_fft_2d(psi, V, KX, KY, dt)

        psi_exact = exact_free_gaussian_2d(
            X, Y, t_final, x0, y0, sigma_val, kx0, ky0, dx, dy
        )
        diff = psi - psi_exact
        l1 = float(np.sum(np.abs(diff)) * dx * dy)
        l2 = float(np.sqrt(np.sum(np.abs(diff) ** 2) * dx * dy))
        linf = float(np.max(np.abs(diff)))

        mass_num = mass_2d(psi, dx, dy)
        mass_exact = mass_2d(psi_exact, dx, dy)

        results.append({
            "grid_size": nx, "N": nx * ny, "dx": dx, "dy": dy,
            "dt": dt, "n_steps": n_steps,
            "L1_error": l1, "L2_error": l2, "Linf_error": linf,
            "mass_numerical": mass_num, "mass_exact": mass_exact,
            "mass_error": abs(mass_num - mass_exact),
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(cfg.outdir, "2d_convergence_study.csv"), index=False)

    if len(df) >= 2:
        df["log_N"] = np.log(df["N"])
        df["log_L2"] = np.log(df["L2_error"].replace(0, np.nan))
        valid = df.dropna(subset=["log_L2"])
        if len(valid) >= 2:
            coeffs = np.polyfit(valid["log_N"].values, valid["log_L2"].values, 1)
            df["convergence_order"] = -coeffs[0]
            print(f"\nFitted convergence order (L2 vs N): {-coeffs[0]:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axes[0].loglog(df["N"], df["L1_error"], "o-", lw=2, color="#2E86AB", label="L1")
    axes[0].loglog(df["N"], df["L2_error"], "s-", lw=2, color="#A23B72", label="L2")
    axes[0].loglog(df["N"], df["Linf_error"], "^-", lw=2, color="#F18F01", label="Linf")
    axes[0].set_xlabel("Number of Grid Points (N)", labelpad=8)
    axes[0].set_ylabel("Error", labelpad=8)
    axes[0].set_title("L1, L2, Linf Errors vs Grid Size", fontweight='bold',
                      fontsize=12, pad=10)
    axes[0].legend(fontsize=10, loc='best')
    axes[0].grid(True, alpha=0.3, which='both')

    axes[1].semilogy(df["grid_size"], df["mass_error"], "o-", lw=2, color="#2E86AB")
    axes[1].set_xlabel("Grid Size (nx = ny)", labelpad=8)
    axes[1].set_ylabel("Mass Conservation Error", labelpad=8)
    axes[1].set_title("Mass Conservation Error", fontweight='bold', fontsize=12, pad=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(range(len(df)), df["mass_numerical"], width=0.6,
                color="#2E86AB", alpha=0.7, label="Numerical")
    axes[2].axhline(y=1.0, color="#C73E1D", linestyle="--", lw=2,
                    label="Expected (M=1)")
    axes[2].set_xticks(range(len(df)))
    axes[2].set_xticklabels([str(n) for n in df["grid_size"]])
    axes[2].set_xlabel("Grid Size", labelpad=8)
    axes[2].set_ylabel("Mass", labelpad=8)
    axes[2].set_title("Mass Conservation Across Grid Sizes",
                      fontweight='bold', fontsize=12, pad=10)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_ylim(0.99, 1.01)

    fig.suptitle("Figure: 2D Convergence Analysis", fontsize=14, fontweight='bold',
                 y=1.02)
    fig.savefig(os.path.join(cfg.outdir, "figure_2d_convergence.png"), dpi=cfg.dpi)
    plt.close(fig)

    print(f"\n2D Convergence Results:")
    print(df.to_string(index=False))
    return df


# =============================================================================
# Wave packet animation experiment
# =============================================================================


def save_wavepacket_animation_experiment(cfg: RunConfig) -> None:
    """Figure 5: free Gaussian wavepacket propagation animation with exact solution.

    Note: This is the experiment-level wrapper (takes only cfg). Different from
    tdse.visualization.save_wavepacket_animation which is a low-level renderer.
    """
    print_section("Figure 5: wavepacket animation")
    n = 512 if not cfg.quick else 384
    x, dx = grid(-30.0, 30.0, n)
    dt, t_end = 0.004, 3.0
    t = np.arange(0.0, t_end + 0.5 * dt, dt)
    psi0 = gaussian_wavepacket(x, -10.0, 1.1, 2.0, dx)
    v = potential_free(x)
    saved_t, hist = solve("Split-Step-FFT", psi0, v, x, t, dx, dt,
                          store_every=max(1, len(t) // 140))

    max_frames = 30
    if len(saved_t) > max_frames:
        step = len(saved_t) // max_frames
        hist = np.array(hist)
        saved_t = np.array(saved_t)
        indices = np.arange(0, len(saved_t), step)[:max_frames]
        saved_t = saved_t[indices].tolist()
        hist = hist[indices]

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
        ani = FuncAnimation(fig, update, frames=len(saved_t),
                            init_func=init, blit=True)
        ani.save(os.path.join(cfg.outdir, "figure5_wavepacket_animation.gif"),
                 writer=PillowWriter(fps=20), dpi=150)
    fig.savefig(os.path.join(cfg.outdir, "figure5_wavepacket_last_frame.png"),
                dpi=cfg.dpi)
    plt.close(fig)
