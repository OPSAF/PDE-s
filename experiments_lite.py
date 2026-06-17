#!/usr/bin/env python3
"""
TDSE 数值方法对比实验 — 微分方程数值解课程专用

实验结构（按重要性排序）：
  实验①  一维无限深势阱（主实验，有解析解）
          → 五方法对比 | 解析解验证 | 完整误差分析 | 三值图 | 动图(更长+解析对比GIF)
          → 新增：误差演化图 | 期望值演化图 | 相空间轨迹图
          → 新增：newtdse风格并集画图（误差曲线/概率守恒/snapshot/汇总表）

  实验②  一维谐振子相干态（来自newtdse.py）
          → 全部5种方法 | 完整解析解验证 | L2/L∞/概率/<x>/<p>/方差分析
          → 收敛性研究 | 稳定性扫描 | 相空间轨迹 | newtdse风格GIF(3个)
          → 并集画图：误差曲线/概率守恒/snapshot/汇总表

  实验③  二维各向同性谐振子相干态（来自case7.py）
          → ADI+SSF双方法 | 解析解验证 | 轨迹对比 | case7风格GIF

  实验④  Von Neumann 稳定性扫描（详细版）
  实验⑤  Crank-Nicolson 收敛性验证（详细版）

误差概念整理：
  L1  = ∫|ψ_num − ψ_ref| dx              （平均绝对偏差）
  L2  = sqrt(∫|ψ_num − ψ_ref|² dx)        （均方根误差）
  L∞  = max|ψ_num − ψ_ref|                （最大点误差）
  相对L2 = L2 / ||ψ_ref||_2                 （归一化误差）
  Re/Im分解 = 分别计算实部和虚部的L2误差    （定位相位漂移 vs 振幅衰减）
"""

import os
import time
import shutil
import tempfile
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

from PIL import Image as PILImage
import imageio.v2 as imageio

from tdse.config import RunConfig
from tdse.potentials import (
    Array,
    grid,
    make_2d_grid,
    gaussian_wavepacket,
    gaussian_wavepacket_2d,
    potential_free,
    potential_rect_barrier,
    potential_infinite_well_mask,
    absorbing_potential_2d,
    exact_free_gaussian,
    exact_free_gaussian_2d,
    probability_mass,
    l1_l2_linf_error,
    l2_error_real_imag,
    normalize,
)
from tdse.solvers import solve, solve_2d


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def get_adaptive_dpi(num_subplots: int) -> int:
    """根据子图数量自适应设置DPI，最低600"""
    if num_subplots <= 3:
        return 600
    elif num_subplots <= 6:
        return 800
    else:
        return 1000


# ── 统一颜色方案（与newtdse.py一致）──
COLORS_UNION = {
    "FTCS": "#e84855",
    "Backward-Euler": "#f77f00",
    "BE": "#f77f00",
    "Crank-Nicolson": "#2176ae",
    "CN": "#2176ae",
    "RK4": "#06a77d",
    "Split-Step-FFT": "#9b2dca",
    "SSF": "#9b2dca",
    "SSFFT": "#9b2dca",
    "Exact": "#111111",
}

STYLES_UNION = {
    "FTCS": "--",
    "Backward-Euler": "-.",
    "BE": "-.",
    "Crank-Nicolson": "-",
    "CN": "-",
    "RK4": ":",
    "Split-Step-FFT": "-",
    "SSF": "-",
    "SSFFT": "-",
    "Exact": "-",
}


def run_1d_methods(methods, psi0, v, x, t, dx, dt):
    """统一运行一维5种方法"""
    results = {}
    for method in methods:
        try:
            _, hist = solve(method, psi0, v, x, t, dx, dt, store_every=len(t)-1)
            mass = probability_mass(hist[-1], dx)
            results[method] = {
                "psi": hist[-1],
                "stable": True,
                "mass": mass,
            }
        except Exception as e:
            print(f"  {method} failed: {e}")
            results[method] = {"psi": None, "stable": False, "mass": np.nan}
    return results


def run_1d_methods_with_history(methods, psi0, v, x, t, dx, dt, store_every=50):
    """Run 1D methods and return results WITH full history for error evolution."""
    results = {}
    for method in methods:
        try:
            t_hist, hist = solve(method, psi0, v, x, t, dx, dt, store_every=store_every)
            mass = probability_mass(hist[-1], dx)
            results[method] = {
                "psi": hist[-1],
                "stable": True,
                "mass": mass,
                "t_hist": t_hist,
                "hist": hist,
            }
        except Exception as e:
            print(f"  {method} failed: {e}")
            results[method] = {"psi": None, "stable": False, "mass": np.nan,
                               "t_hist": [], "hist": []}
    return results


def plot_1d_three_panel(x, results, methods, colors, labels, title_prefix,
                        outdir, filename, dpi=600,
                        show_barrier=None, show_v=None,
                        psi_exact=None):
    """标准三面板：|psi|^2, Re[psi], Im[psi]（可选含解析解）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    axes[0].set_title(r"$|\psi|^2$", fontsize=13)
    axes[1].set_title(r"$\mathrm{Re}[\psi]$", fontsize=13)
    axes[2].set_title(r"$\mathrm{Im}[\psi]$", fontsize=13)

    if psi_exact is not None:
        axes[0].plot(x, np.abs(psi_exact)**2, 'k--', lw=2.0, label='Exact', alpha=0.8, zorder=10)
        axes[1].plot(x, np.real(psi_exact), 'k--', lw=2.0, label='Exact', alpha=0.8, zorder=10)
        axes[2].plot(x, np.imag(psi_exact), 'k--', lw=2.0, label='Exact', alpha=0.8, zorder=10)

    for i, method in enumerate(methods):
        r = results[method]
        if r["stable"] and r["psi"] is not None:
            psi = r["psi"]
            axes[0].plot(x, np.abs(psi)**2, label=labels[i], color=colors[i], lw=1.5)
            axes[1].plot(x, np.real(psi), label=labels[i], color=colors[i], lw=1.5)
            axes[2].plot(x, np.imag(psi), label=labels[i], color=colors[i], lw=1.5)
        else:
            for ax in axes:
                ax.text(0.5, 0.95, f"[{labels[i]}: DIVERGED]",
                       transform=ax.transAxes, ha='center', va='top',
                       fontsize=9, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFE4E4', edgecolor='red'))
    if show_barrier:
        bl, br = show_barrier
        for ax in axes:
            ax.axvline(bl, color='k', linestyle='--', lw=1.2, alpha=0.7)
            ax.axvline(br, color='k', linestyle='--', lw=1.2, alpha=0.7)
    if show_v is not None:
        ax2 = axes[0].twinx()
        ax2.plot(x, show_v / (np.max(np.abs(show_v)) + 1e-30) * axes[0].get_ylim()[1] * 0.4,
                 'gray', ':', lw=1.0, alpha=0.6)
        ax2.set_yticks([])
    for ax in axes:
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3); ax.set_xlabel(r"$x$", fontsize=11)
    fig.savefig(os.path.join(outdir, filename), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ── Analysis Utilities (from tdse_project style) ──

def compute_expectation_values(psi, x, V, dx):
    """Compute quantum expectation values: <x>, <p>, <x^2>, <p^2>, energy."""
    prob = np.abs(psi)**2
    x_exp = np.sum(x * prob) * dx
    x2_exp = np.sum(x**2 * prob) * dx
    dpsi = np.zeros_like(psi, dtype=complex)
    dpsi[1:-1] = (psi[2:] - psi[:-2]) / (2*dx)
    dpsi[0] = (psi[1] - psi[0]) / dx
    dpsi[-1] = (psi[-1] - psi[-2]) / dx
    p_exp = np.sum(np.conj(psi) * (-1j) * dpsi) * dx
    p_exp = float(np.real(p_exp))
    d2psi = np.zeros_like(psi, dtype=complex)
    d2psi[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2
    p2_exp = float(np.real(np.sum(np.conj(psi) * (-1)*d2psi) * dx))
    V_exp = np.sum(V * prob) * dx
    energy = p2_exp / 2.0 + V_exp
    return {"x": x_exp, "p": p_exp, "x2": x2_exp, "p2": p2_exp, "energy": energy}


def plot_error_evolution(t_hist, hist, psi_exact_fn, x, dx, outdir, filename,
                         title_prefix="", dpi=600):
    """Plot error evolution over time: L2 Error + Max Error (semilogy), Probability Error."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    times = np.array(t_hist)
    l2_errors, max_errors, prob_errors = [], [], []
    for i, psi_t in enumerate(hist):
        psi_ex = psi_exact_fn(times[i]) if callable(psi_exact_fn) else None
        if psi_ex is not None:
            l2 = float(np.sqrt(np.sum(np.abs(psi_t - psi_ex)**2) * dx))
            mx = float(np.max(np.abs(psi_t - psi_ex)))
            l2_errors.append(l2); max_errors.append(mx)
        else:
            l2_errors.append(None); max_errors.append(None)
        prob_errors.append(abs(probability_mass(psi_t, dx) - 1.0))

    valid_l2 = [(t,e) for t,e in zip(times, l2_errors) if e is not None and e > 0]
    if valid_l2:
        vt, ve = zip(*valid_l2)
        ax1.semilogy(vt, ve, 'o-', color='#2E86AB', lw=1.5, markersize=3, label='L2 Error')
    valid_mx = [(t,e) for t,e in zip(times, max_errors) if e is not None and e > 0]
    if valid_mx:
        vt, ve = zip(*valid_mx)
        ax1.semilogy(vt, ve, 's-', color='#C73E1D', lw=1.5, markersize=3, label='Max Error')
    ax1.set_xlabel("Time"); ax1.set_ylabel("Error")
    ax1.set_title(f"{title_prefix}Error Evolution"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(times, prob_errors, '^-', color='#1B998B', lw=1.5, markersize=3)
    ax2.axhline(0, color='gray', ls=':', alpha=0.5)
    ax2.set_xlabel("Time"); ax2.set_ylabel("|M - 1|")
    ax2.set_title(f"{title_prefix}Probability Conservation"); ax2.grid(True, alpha=0.3)

    fig.savefig(os.path.join(outdir, filename), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_expectation_evolution(t_hist, hist, x, V, dx, outdir, filename,
                                title_prefix="", dpi=600):
    """Plot expectation values <x> and <p> evolution over time."""
    x_vals, p_vals, e_vals = [], [], []
    for psi_t in hist:
        ev = compute_expectation_values(psi_t, x, V, dx)
        x_vals.append(ev["x"]); p_vals.append(ev["p"]); e_vals.append(ev["energy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax1.plot(t_hist, x_vals, 'o-', color='#2E86AB', lw=1.5, markersize=3, label=r'$\langle x \rangle$')
    ax1.set_xlabel("Time"); ax1.set_ylabel(r"$\langle x \rangle$")
    ax1.set_title(f"{title_prefix}Position Expectation"); ax1.grid(True, alpha=0.3)

    ax2.plot(t_hist, p_vals, 'o-', color='#C73E1D', lw=1.5, markersize=3, label=r'$\langle p \rangle$')
    ax2.set_xlabel("Time"); ax2.set_ylabel(r"$\langle p \rangle$")
    ax2.set_title(f"{title_prefix}Momentum Expectation"); ax2.grid(True, alpha=0.3)

    fig.savefig(os.path.join(outdir, filename), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_phase_space(t_hist, hist, x, V, dx, outdir, filename, title="", dpi=600):
    """Plot phase space trajectory (<x>, <p>) with time coloring."""
    from matplotlib.collections import LineCollection

    x_vals, p_vals = [], []
    for psi_t in hist:
        ev = compute_expectation_values(psi_t, x, V, dx)
        x_vals.append(ev["x"]); p_vals.append(ev["p"])

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    points = np.array([x_vals, p_vals]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(t_hist[0], t_hist[-1])
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2, alpha=0.8)
    lc.set_array(np.array(t_hist))
    ax.add_collection(lc)

    ax.scatter(x_vals[0], p_vals[0], c='green', s=80, zorder=5, marker='o',
               label=f'Start t={t_hist[0]:.2f}', edgecolors='black')
    ax.scatter(x_vals[-1], p_vals[-1], c='red', s=80, zorder=5, marker='s',
               label=f'End t={t_hist[-1]:.2f}', edgecolors='black')

    cb = fig.colorbar(lc, ax=ax)
    cb.set_label("Time")
    ax.set_xlabel(r"$\langle x \rangle$"); ax.set_ylabel(r"$\langle p \rangle$")
    ax.set_title(title or "Phase Space Trajectory")
    ax.set_aspect('equal', adjustable='datalim'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(outdir, filename), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 可复用：newtdse风格并集画图函数（供Exp-01和Exp-02调用）
# ──────────────────────────────────────────────────────────────

def plot_newtdse_union_errors(results_hist, exact_fn, x, dx, outdir, filename_prefix,
                               title_prefix, T_final=None, dpi=600):
    """
    newtdse风格：全部方法的L2和Linf误差随时间曲线（两张图：L2和Linf）。
    左图：L2 Error vs Time（semilogy），右图：Linf Error vs Time（semilogy）
    所有方法在同一图上，不同颜色线。标注T_final相关参考线。
    """
    fig, (ax_l2, ax_linf) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for method_name, rdata in results_hist.items():
        if not rdata.get("stable", False) or len(rdata.get("hist", [])) == 0:
            continue
        t_arr = np.array(rdata["t_hist"])
        hist_arr = rdata["hist"]
        l2_list, linf_list = [], []
        for i, psi_t in enumerate(hist_arr):
            try:
                psi_ex = exact_fn(t_arr[i])
                l2_val = float(np.sqrt(np.sum(np.abs(psi_t - psi_ex)**2) * dx))
                linf_val = float(np.max(np.abs(psi_t - psi_ex)))
                if l2_val > 10:
                    l2_val = np.nan
                if linf_val > 10:
                    linf_val = np.nan
            except Exception:
                l2_val = np.nan
                linf_val = np.nan
            l2_list.append(l2_val)
            linf_list.append(linf_val)

        clr = COLORS_UNION.get(method_name, "#333333")
        ls = STYLES_UNION.get(method_name, "-")
        lbl = method_name.replace("Split-Step-FFT", "SSF").replace("Backward-Euler", "BE").replace("Crank-Nicolson", "CN")

        ax_l2.semilogy(t_arr, l2_list, color=clr, ls=ls, lw=1.8, label=lbl)
        ax_linf.semilogy(t_arr, linf_list, color=clr, ls=ls, lw=1.8, label=lbl)

    if T_final is not None:
        half_T = T_final / 2.0
        if half_T > 0:
            ax_l2.axvline(half_T, color="gray", ls=":", lw=1, alpha=0.6)
            ax_linf.axvline(half_T, color="gray", ls=":", lw=1, alpha=0.6)
        ax_l2.axvline(T_final, color="gray", ls="--", lw=1, alpha=0.6)
        ax_linf.axvline(T_final, color="gray", ls="--", lw=1, alpha=0.6)

    ax_l2.set_xlabel("t"); ax_l2.set_ylabel("L2 Error")
    ax_l2.set_title(f"{title_prefix} L2 Error vs Time"); ax_l2.legend(fontsize=8)
    ax_l2.grid(True, which="both", alpha=0.3)

    ax_linf.set_xlabel("t"); ax_linf.set_ylabel("Linf Error")
    ax_linf.set_title(f"{title_prefix} Linf Error vs Time"); ax_linf.legend(fontsize=8)
    ax_linf.grid(True, which="both", alpha=0.3)

    fig.suptitle(f"{title_prefix} Error Evolution — All Methods", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, f"{filename_prefix}_errors.png"), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [{filename_prefix}_errors] All-method error curves (L2 + Linf)")


def plot_newtdse_union_probability(results_hist, exact_fn, x, dx, outdir, filename,
                                    title_prefix, P0=None, dpi=600):
    """newtdse风格：全部方法的概率守恒图。int(|psi|^2)dx vs t，所有方法同图。"""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    for method_name, rdata in results_hist.items():
        if not rdata.get("stable", False) or len(rdata.get("hist", [])) == 0:
            continue
        t_arr = np.array(rdata["t_hist"])
        hist_arr = rdata["hist"]
        prob_list = []
        for psi_t in hist_arr:
            pval = probability_mass(psi_t, dx)
            if abs(pval) > 10:
                pval = np.nan
            prob_list.append(pval)

        clr = COLORS_UNION.get(method_name, "#333333")
        ls = STYLES_UNION.get(method_name, "-")
        lbl = method_name.replace("Split-Step-FFT", "SSF").replace("Backward-Euler", "BE").replace("Crank-Nicolson", "CN")

        ax.plot(t_arr, prob_list, color=clr, ls=ls, lw=1.8, label=lbl)

    if P0 is not None:
        ax.axhline(P0, color="k", ls="--", lw=1.2, alpha=0.5, label=f"P0={P0:.4f}")

    ax.set_xlabel("t"); ax.set_ylabel("int(|psi|^2) dx")
    ax.set_title(f"{title_prefix} Probability Conservation — All Methods", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, filename), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [{filename}] Probability conservation (all methods)")


def plot_newtdse_union_snapshot(x, results, results_hist, exact_fn, methods_stable,
                                 colors_dict, t_snap, outdir, filename, title,
                                 V_show=None, dpi=130):
    """
    newtdse风格：GridSpec 2×3快照对比。
    上排：Re(psi), Im(psi), |psi|^2（所有稳定方法+Exact叠加）
    下排：每个稳定方法的|error|逐点分布（semilogy）
    """
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)

    quants = [
        ("Re(psi)", lambda p: np.real(p), (-1.1, 1.1)),
        ("Im(psi)", lambda p: np.imag(p), (-1.1, 1.1)),
        ("|psi|^2", lambda p: np.abs(p)**2, (-0.05, 0.75)),
    ]

    best_method = None
    best_idx = None
    for m in methods_stable:
        if m in results_hist and results_hist[m].get("stable") and len(results_hist[m].get("t_hist", [])) > 0:
            t_arr = np.array(results_hist[m]["t_hist"])
            idx = int(np.argmin(np.abs(t_arr - t_snap)))
            best_method = m
            best_idx = idx
            break

    if best_method is None or best_idx is None:
        print(f"  Warning: no stable data for snapshot at t={t_snap}")
        plt.close(fig)
        return

    t_actual = results_hist[best_method]["t_hist"][best_idx]

    for col, (qtitle, qfn, yl) in enumerate(quants):
        ax = fig.add_subplot(gs[0, col])
        if V_show is not None:
            Vn = V_show / (np.max(np.abs(V_show)) + 1e-30) * yl[1] * 0.85
            ax.fill_between(x, yl[0], Vn, alpha=0.07, color="gray")

        psi_ex = exact_fn(t_actual)
        ax.plot(x, qfn(psi_ex), color=COLORS_UNION["Exact"], lw=2.2, label="Exact", zorder=6)

        for nm in methods_stable:
            if nm in results_hist and results_hist[nm].get("stable"):
                hist_nm = results_hist[nm]["hist"]
                if best_idx < len(hist_nm):
                    yv = qfn(hist_nm[best_idx])
                    if np.nanmax(np.abs(yv)) < 1e6:
                        c = colors_dict.get(nm, "#333333")
                        s = STYLES_UNION.get(nm, "-")
                        lbl = nm.replace("Split-Step-FFT", "SSF").replace("Backward-Euler", "BE").replace("Crank-Nicolson", "CN")
                        ax.plot(x, yv, color=c, ls=s, lw=1.6, alpha=0.85, label=lbl)

        ax.set_xlim(x[0], x[-1]); ax.set_ylim(*yl)
        ax.set_title(f"{qtitle}  (t = {t_actual:.3f})")
        ax.set_xlabel("x"); ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)

    n_bottom = min(len(methods_stable), 3)
    for col, nm in enumerate(methods_stable[:n_bottom]):
        ax = fig.add_subplot(gs[1, col])
        if nm in results_hist and results_hist[nm].get("stable"):
            hist_nm = results_hist[nm]["hist"]
            if best_idx < len(hist_nm):
                err_field = np.abs(hist_nm[best_idx] - psi_ex)
                c = colors_dict.get(nm, "#333333")
                ax.semilogy(x, err_field, color=c, lw=1.8)
                lbl = nm.replace("Split-Step-FFT", "SSF").replace("Backward-Euler", "BE").replace("Crank-Nicolson", "CN")
                ax.set_title(f"Pointwise error — {lbl}")
        ax.set_xlim(x[0], x[-1]); ax.set_xlabel("x")
        ax.set_ylabel("|psi_num - psi_exact|")
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(outdir, filename), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  [{filename}] Snapshot comparison at t={t_actual:.3f}")


def plot_newtdse_union_summary_table(results, results_hist, methods_all, colors_dict,
                                      outdir, filename, title, P0=None, dpi=130):
    """
    newtdse风格：汇总表格图（matplotlib.table）。
    列：Method, L2 Error, Linf Error, deltaP/P0, Runtime, Blow-up, Stability
    带颜色的header和行背景。
    """
    rows = []
    for nm in methods_all:
        r = results.get(nm, {})
        rh = results_hist.get(nm, {})

        l2_fin = linf_fin = dp_ratio = np.nan
        runtime = 0.0
        blow_str = "--"
        stability = "unstable"

        if r.get("stable", False) and r.get("psi") is not None:
            if rh.get("stable") and len(rh.get("l2", [])) > 0:
                l2_arr = np.array(rh["l2"])
                linf_arr = np.array(rh.get("linf", []))
                prob_arr = np.array(rh.get("prob", []))
                valid = ~np.isnan(l2_arr)
                if valid.any():
                    idx_last = int(np.where(valid)[0][-1])
                    l2_fin = l2_arr[idx_last]
                    linf_fin = linf_arr[idx_last]
                    pr_fin = prob_arr[idx_last] if idx_last < len(prob_arr) else P0 or 1.0
                    if P0 is not None and P0 != 0:
                        dp_ratio = abs(pr_fin - P0) / P0
            stability = "stable"

        blow_info = rh.get("blow", None)
        if blow_info is not None:
            blow_str = f"t~{blow_info:.3f}"

        rt = rh.get("runtime", 0.0)
        rows.append([
            nm.replace("Split-Step-FFT", "SSF").replace("Backward-Euler", "BE").replace("Crank-Nicolson", "CN"),
            f"{l2_fin:.3e}" if np.isfinite(l2_fin) else "diverged",
            f"{linf_fin:.3e}" if np.isfinite(linf_fin) else "diverged",
            f"{dp_ratio:.3e}" if np.isfinite(dp_ratio) else "--",
            f"{rt:.2f}s",
            blow_str,
            stability,
        ])

    col_labels = ["Method", "L2 Error", "Linf Error", "deltaP/P0", "Runtime", "Blow-up", "Stability"]

    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    tbl.scale(1.0, 2.1)

    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2b3a55")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    row_bg_map = {
        "FTCS": "#fde8e8",
        "Backward-Euler": "#fff3e0",
        "Crank-Nicolson": "#e3f2fd",
        "RK4": "#e8f5e9",
        "Split-Step-FFT": "#f3e5f5",
    }
    for i, row in enumerate(rows):
        nm_raw = methods_all[i] if i < len(methods_all) else row[0]
        bg = row_bg_map.get(nm_raw, "#fafafa")
        for j in range(len(col_labels)):
            tbl[(i+1, j)].set_facecolor(bg)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, filename), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  [{filename}] Summary table")


# ──────────────────────────────────────────────────────────────
# 无限深势阱解析解
# ──────────────────────────────────────────────────────────────

def _exact_infinite_well_gaussian(x, t, x0, sigma, k0, a, b, n_eigen=200):
    """无限深势阱中高斯波包的精确解（本征函数展开法）。"""
    L = b - a
    psi = np.zeros_like(x, dtype=complex)
    psi0 = gaussian_wavepacket(x, x0, sigma, k0, (x[1]-x[0]))
    psi0_norm = np.sqrt(np.sum(np.abs(psi0)**2) * (x[1]-x[0]))

    for n in range(1, n_eigen + 1):
        phi_n = np.sqrt(2.0 / L) * np.sin(n * np.pi * (x - a) / L)
        c_n = np.sum(np.conj(phi_n) * psi0) * (x[1] - x[0])
        E_n = (n * np.pi)**2 / (2.0 * L**2)
        psi += c_n * phi_n * np.exp(-1j * E_n * t)

    return normalize(psi.astype(complex), x[1]-x[0])


# ──────────────────────────────────────────────────────────────
# 实验①：一维无限深势阱（主实验，有解析解）
# ──────────────────────────────────────────────────────────────

def experiment_1d_infinite_well(cfg):
    """一维无限深势阱 — 五种方法 + 解析解完整对比（初值=本征态）"""
    print("=" * 60)
    print("Exp-01: 1D Infinite Well (MAIN — eigenstate initial condition)")
    print("=" * 60)

    n = 2048
    well_left, well_right = -10.0, 10.0
    L = well_right - well_left
    x, dx = grid(well_left, well_right, n)
    dt = 0.001
    t_end = 8.0
    t = np.arange(0.0, t_end + 0.5 * dt, dt)

    eigen_n = 3
    E_n = (eigen_n * np.pi)**2 / (2.0 * L**2)

    phi_n = np.sqrt(2.0 / L) * np.sin(eigen_n * np.pi * (x - well_left) / L)
    phi_n = phi_n.astype(complex)
    phi_n = normalize(phi_n, dx)
    psi0 = phi_n.copy()

    print(f"  Eigenstate n={eigen_n}, E_n={E_n:.4f}, n={n}, dt={dt}")
    print(f"  Domain: [{well_left}, {well_right}], Dirichlet BC enforced")

    v = np.zeros_like(x, dtype=float)

    methods_all = ["FTCS", "Backward-Euler", "Crank-Nicolson", "RK4", "Split-Step-FFT"]
    colors_all = ["#9B9B9B", "#2E86AB", "#1B998B", "#F18F01", "#C73E1D"]
    labels_all = ["FTCS", "BE", "CN", "RK4", "SSF"]

    results = run_1d_methods(methods_all, psi0, v, x, t, dx, dt)

    psi_exact = phi_n * np.exp(-1j * E_n * t_end)

    dpi = get_adaptive_dpi(3)

    # Fig1a: 全部5方法 + 解析解 三值对比
    plot_1d_three_panel(
        x, results, methods_all, colors_all, labels_all,
        "Infinite Well (all)", cfg.outdir, "fig1a_inf_well_all.png",
        dpi=dpi, show_barrier=(well_left, well_right), psi_exact=psi_exact)
    print("  [Fig1a] All 5 methods + Exact: three-value")

    # Fig1b: 稳定方法 + 解析解三值对比
    m_stable = ["Backward-Euler", "Crank-Nicolson", "RK4", "Split-Step-FFT"]
    c_stable = ["#2E86AB", "#1B998B", "#F18F01", "#C73E1D"]
    l_stable = ["BE", "CN", "RK4", "SSF"]
    plot_1d_three_panel(
        x, results, m_stable, c_stable, l_stable,
        "Infinite Well (stable)", cfg.outdir, "fig1b_inf_well_stable.png",
        dpi=dpi, show_barrier=(well_left, well_right), psi_exact=psi_exact)
    print("  [Fig1b] Stable methods + Exact: three-value")

    # Fig1c: 各方法与解析解单独对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    axes_flat = axes.flatten()
    for idx, method in enumerate(m_stable):
        ax = axes_flat[idx]
        r = results[method]
        if r["stable"] and r["psi"] is not None:
            ax.fill_between([well_left, well_right], [0], [np.max(np.abs(psi_exact)**2)*1.3],
                            color='gray', alpha=0.12)
            ax.plot(x, np.abs(psi_exact)**2, 'k--', lw=1.8, label='Exact', alpha=0.8)
            ax.plot(x, np.abs(r["psi"])**2, lw=1.5, label=method)
            l1, l2, linf = l1_l2_linf_error(r["psi"], psi_exact, dx)
            ax.set_title(f"{method}: L1={l1:.2e}  L2={l2:.2e}  Linf={linf:.2e}", fontsize=10)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlabel("x")
    fig.savefig(os.path.join(cfg.outdir, "fig1c_inf_well_vs_exact.png"),
                dpi=get_adaptive_dpi(4), bbox_inches='tight')
    plt.close(fig)
    print("  [Fig1c] Stable methods vs Exact (separate)")

    # Fig1d: 完整误差分析表
    err_rows = []
    for method in methods_all:
        r = results[method]
        row = {"Method": method}
        if r["stable"] and r["psi"] is not None:
            l1, l2, linf = l1_l2_linf_error(r["psi"], psi_exact, dx)
            l2_re, l2_im, _ = l2_error_real_imag(r["psi"], psi_exact, dx)
            ref_norm = float(np.sqrt(np.sum(np.abs(psi_exact)**2) * dx))
            rel_l2 = l2 / ref_norm if ref_norm > 0 else np.nan
            row.update({
                "Mass": r["mass"], "L1": l1, "L2": l2, "Linf": linf,
                "Rel_L2": rel_l2, "L2_Re": l2_re, "L2_Im": l2_im,
            })
        else:
            row.update({"Mass": np.nan, "L1": np.nan, "L2": np.nan,
                        "Linf": np.nan, "Rel_L2": np.nan,
                        "L2_Re": np.nan, "L2_Im": np.nan})
        err_rows.append(row)

    df_err = pd.DataFrame(err_rows)
    df_err.to_csv(os.path.join(cfg.outdir, "data_inf_well_errors.csv"), index=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    metrics = [
        ("L1 Error", "L1", True), ("L2 Error", "L2", True),
        ("Linf Error", "Linf", True), ("Relative L2", "Rel_L2", True),
        ("L2 Real Part", "L2_Re", True), ("L2 Imag Part", "L2_Im", True),
    ]
    valid_methods = []
    for m in methods_all:
        val = df_err.loc[df_err["Method"]==m, "L2"].values[0]
        if np.isfinite(val):
            valid_methods.append(m)
    bar_colors_dict = dict(zip(methods_all, colors_all))

    for idx, (title, col, use_log) in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        vals = [df_err.loc[df_err["Method"]==m, col].values[0] for m in valid_methods]
        clrs = [bar_colors_dict[m] for m in valid_methods]
        bars = ax.bar(range(len(valid_methods)), vals, color=clrs, edgecolor='white', lw=0.5)
        if use_log:
            vals_finite = [v for v in vals if np.isfinite(v) and v > 0]
            if vals_finite:
                ax.set_yscale('log'); v_min = min(vals_finite) * 0.5; v_max = max(vals_finite) * 5.0
                ax.set_ylim(v_min, v_max)
        ax.set_xticks(range(len(valid_methods))); ax.set_xticklabels(valid_methods, fontsize=9, rotation=10)
        ax.set_title(title, fontsize=11); ax.grid(True, alpha=0.3)
        for bi, (bar, val) in enumerate(zip(bars, vals)):
            if np.isfinite(val) and val > 0:
                bh = bar.get_height(); bx = bar.get_x() + bar.get_width() / 2.
                ax.text(bx, bh * 1.12, f'{val:.2e}', ha='center', va='bottom',
                       fontsize=7, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.15', facecolor=clrs[bi], alpha=0.8, edgecolor='none'),
                       zorder=10)

    fig.suptitle("Infinite Well - Complete Error Analysis vs Analytic Solution",
                 fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(cfg.outdir, "fig1d_inf_well_error_analysis.png"),
                dpi=get_adaptive_dpi(6), bbox_inches='tight')
    plt.close(fig)
    print("  [Fig1d] Complete error analysis (6 metrics)")
    print(df_err.round(8).to_string(index=False))

    # 动图
    if results["Crank-Nicolson"]["stable"]:
        print("  Generating animation (CN, eigenstate evolution)...")
        t_long = np.arange(0.0, 40.0 + 0.5 * dt, dt)
        _, hist_long = solve("Crank-Nicolson", psi0, v, x, t_long, dx, dt,
                             store_every=int(len(t_long)/120))
        t_snap = t_long[::int(len(t_long)//120)][:len(hist_long)]
        _save_lightweight_1d_gif_three_line(
            cfg.outdir, x, hist_long, t_snap,
            "gif_inf_well.gif",
            title=f"Infinite Well (Eigenstate n={eigen_n})",
            show_barrier=(well_left, well_right),
            fps=20
        )
        print("  [GIF1] Eigenstate animation (|psi|^2 should be constant!)")

        _save_lightweight_1d_gif_with_exact(
            cfg.outdir, x, hist_long, t_snap,
            lambda ti: phi_n * np.exp(-1j * E_n * ti),
            "gif_inf_well_exact_compare.gif",
            title=f"Infinite Well (Eigenstate n={eigen_n}) CN vs Exact",
            show_barrier=(well_left, well_right),
            fps=16
        )
        print("  [GIF1b] Eigenstate animation (CN + Exact overlaid)")

    # 误差演化、期望值、相空间
    if results["Crank-Nicolson"]["stable"]:
        print("  Computing error evolution (CN)...")
        t_cn, hist_cn = solve("Crank-Nicolson", psi0, v, x, t, dx, dt, store_every=max(1, len(t)//100))

        def exact_fn(ti):
            return phi_n * np.exp(-1j * E_n * ti)

        plot_error_evolution(t_cn, hist_cn, exact_fn, x, dx, cfg.outdir,
                             "fig1e_inf_well_error_evol.png",
                             title_prefix="Infinite Well (CN): ")
        print("  [Fig1e] Error evolution (CN)")

        plot_expectation_evolution(t_cn, hist_cn, x, v, dx, cfg.outdir,
                                   "fig1f_inf_well_expectations.png",
                                   title_prefix="Infinite Well (CN): ")
        print("  [Fig1f] Expectation values evolution (CN)")

        plot_phase_space(t_cn, hist_cn, x, v, dx, cfg.outdir,
                         "fig1g_inf_well_phase_space.png",
                         title="Infinite Well (Eigenstate n={}) Phase Space".format(eigen_n))
        print("  [Fig1g] Phase space trajectory (CN)")

    # ══════════════════════════════════════════════════
    # 新增：newtdse风格并集画图（Exp-01末尾添加）
    # ══════════════════════════════════════════════════
    if results["Crank-Nicolson"]["stable"]:
        print("  Generating newtdse-style union plots (Exp-01)...")

        results_hist_e1 = run_1d_methods_with_history(
            methods_all, psi0, v, x, t, dx, dt, store_every=max(1, len(t)//100))

        def e1_exact_fn(ti):
            return phi_n * np.exp(-1j * E_n * ti)

        P0_e1 = probability_mass(psi0, dx)
        for mname in methods_all:
            if mname in results_hist_e1 and results_hist_e1[mname]["stable"]:
                rh = results_hist_e1[mname]
                t_arr = np.array(rh["t_hist"])
                hist_arr = rh["hist"]
                l2list, linflist, prolist = [], [], []
                for i, psi_t in enumerate(hist_arr):
                    ti = t_arr[i] if i < len(t_arr) else t_end
                    try:
                        pe = e1_exact_fn(ti)
                        l2list.append(float(np.sqrt(np.sum(np.abs(psi_t - pe)**2) * dx)))
                        linflist.append(float(np.max(np.abs(psi_t - pe))))
                    except Exception:
                        l2list.append(np.nan); linflist.append(np.nan)
                    prolist.append(probability_mass(psi_t, dx))
                rh["l2"] = l2list; rh["linf"] = linflist; rh["prob"] = prolist
                rh["runtime"] = 0.0; rh["blow"] = None

        plot_newtdse_union_errors(
            results_hist_e1, e1_exact_fn, x, dx, cfg.outdir,
            "fig1h_newtdse", "Infinite Well (Exp-01):", T_final=t_end, dpi=600)

        plot_newtdse_union_probability(
            results_hist_e1, e1_exact_fn, x, dx, cfg.outdir,
            "fig1i_newtdse_probability.png", "Infinite Well (Exp-01):", P0=P0_e1, dpi=600)

        snap_t_e1 = t_end / 2.0
        e1_colors_dict = {m: COLORS_UNION.get(m, "#333333") for m in m_stable}
        plot_newtdse_union_snapshot(
            x, results, results_hist_e1, e1_exact_fn, m_stable,
            e1_colors_dict, snap_t_e1, cfg.outdir,
            "fig1j_newtdse_snapshot.png",
            "Infinite Well (Exp-01) Snapshot Comparison",
            V_show=None, dpi=130)

        plot_newtdse_union_summary_table(
            results, results_hist_e1, methods_all,
            {m: COLORS_UNION.get(m, "#333333") for m in methods_all},
            cfg.outdir, "fig1k_newtdse_summary.png",
            "Summary Table — Exp-01 Infinite Well Methods", P0=P0_e1, dpi=130)

        print("  [Fig1h-1k] newtdse union plots done (Exp-01)")

    print(f"  Grid: n={n}, dt={dt}, steps={len(t)-1}, Well width={well_right-well_left}")


# ──────────────────────────────────────────────────────────────
# 一维谐振子相干态精确解析解
# ──────────────────────────────────────────────────────────────

def _exact_ho_coherent(x, t, x0, k0):
    """一维谐振子相干态的精确解析解。V(x)=x^2/2"""
    xc = x0 * np.cos(t) + k0 * np.sin(t)
    pc = k0 * np.cos(t) - x0 * np.sin(t)
    return (np.pi**(-0.25)
            * np.exp(-0.5 * (x - xc)**2)
            * np.exp(1j * pc * x)
            * np.exp(-1j * xc * pc / 2.0)
            * np.exp(-1j * t / 2.0))


def _classical_traj_ho(t, x0, k0):
    """返回谐振子经典轨迹的位置和动量."""
    return x0 * np.cos(t) + k0 * np.sin(t), k0 * np.cos(t) - x0 * np.sin(t)


# ──────────────────────────────────────────────────────────────
# 实验②：一维谐振子相干态（来自newtdse.py）
# ──────────────────────────────────────────────────────────────

def experiment_1d_ho_coherent(cfg):
    """一维谐振子相干态 — 全部5种方法 + 完整解析解验证"""
    print("=" * 60)
    print("Exp-02: 1D Harmonic Oscillator Coherent State (from newtdse.py)")
    print("=" * 60)

    N_ho = 512
    L_ho = 12.0
    x_ho = np.linspace(-L_ho, L_ho, N_ho, endpoint=False)
    dx_ho = x_ho[1] - x_ho[0]
    x0_ho, k0_ho = 2.0, 2.0
    T_final_ho = 2.0 * np.pi
    dt_ho = 0.001
    Nt_ho = int(round(T_final_ho / dt_ho))
    t_ho = np.arange(0.0, T_final_ho + 0.5 * dt_ho, dt_ho)

    print(f"  Potential: V(x) = x^2/2 (Harmonic Oscillator)")
    print(f"  Grid: N={N_ho}, dx={dx_ho:.5f}, domain=[{-L_ho}, {L_ho}]")
    print(f"  Time: T={T_final_ho:.4f} (one HO period), dt={dt_ho}, steps={Nt_ho}")
    print(f"  IC: x0={x0_ho}, k0={k0_ho}")

    v_ho = 0.5 * x_ho**2
    psi0_ho = _exact_ho_coherent(x_ho, 0.0, x0_ho, k0_ho)
    P0_ho = float(np.sqrt(np.sum(np.abs(psi0_ho)**2) * dx_ho))
    print(f"  Initial norm: {P0_ho:.8f}")

    methods_all_ho = ["FTCS", "Backward-Euler", "Crank-Nicolson", "RK4", "Split-Step-FFT"]
    labels_all_ho = ["FTCS", "BE", "CN", "RK4", "SSF"]

    print("  Running all 5 methods...")
    results_ho = run_1d_methods(methods_all_ho, psi0_ho, v_ho, x_ho, t_ho, dx_ho, dt_ho)
    print("  Running all 5 methods with history...")
    results_hist_ho = run_1d_methods_with_history(
        methods_all_ho, psi0_ho, v_ho, x_ho, t_ho, dx_ho, dt_ho, store_every=max(1, Nt_ho//120))

    psi_exact_final_ho = _exact_ho_coherent(x_ho, T_final_ho, x0_ho, k0_ho)
    dpi_ho = get_adaptive_dpi(3)
    colors_ho = [COLORS_UNION.get(m, "#333333") for m in methods_all_ho]

    # Fig2a-2b: 三值对比图
    plot_1d_three_panel(x_ho, results_ho, methods_all_ho, colors_ho, labels_all_ho,
        "HO Coherent (all)", cfg.outdir, "fig2a_ho_all.png",
        dpi=dpi_ho, show_v=v_ho, psi_exact=psi_exact_final_ho)
    print("  [Fig2a] All 5 methods + Exact: three-value")

    m_stable_ho = ["Backward-Euler", "Crank-Nicolson", "RK4", "Split-Step-FFT"]
    c_stable_ho = [COLORS_UNION.get(m, "#333333") for m in m_stable_ho]
    l_stable_ho = ["BE", "CN", "RK4", "SSF"]
    plot_1d_three_panel(x_ho, results_ho, m_stable_ho, c_stable_ho, l_stable_ho,
        "HO Coherent (stable)", cfg.outdir, "fig2b_ho_stable.png",
        dpi=dpi_ho, show_v=v_ho, psi_exact=psi_exact_final_ho)
    print("  [Fig2b] Stable methods + Exact: three-value")

    # Fig2c: CN vs Exact 四面板
    if results_ho.get("Crank-Nicolson", {}).get("stable", False):
        psi_cn_ho = results_ho["Crank-Nicolson"]["psi"]
        l1_c, l2_c, linf_c = l1_l2_linf_error(psi_cn_ho, psi_exact_final_ho, dx_ho)
        l2_re_c, l2_im_c, _ = l2_error_real_imag(psi_cn_ho, psi_exact_final_ho, dx_ho)
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
        ax = axes[0,0]; Vn_s = v_ho/(np.max(v_ho)+1e-30)*np.max(np.abs(psi_exact_final_ho)**2)*0.3
        ax.fill_between(x_ho, 0, np.max(np.abs(psi_exact_final_ho)**2)*1.3, color='gray', alpha=0.08)
        ax.plot(x_ho, Vn_s, 'gray', lw=1.0, alpha=0.5, label='V(x)')
        ax.plot(x_ho, np.abs(psi_exact_final_ho)**2, 'k--', lw=1.8, label='Exact', alpha=0.8)
        ax.plot(x_ho, np.abs(psi_cn_ho)**2, COLORS_UNION["CN"], lw=1.5, label='CN')
        ax.set_xlabel("x"); ax.set_ylabel(r"$|\psi|^2$")
        ax.set_title(rf"CN vs Exact: $|\psi|^2$" + f"\nL2={l2_c:.2e}, Linf={linf_c:.2e}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax = axes[0,1]; ax.plot(x_ho, np.real(psi_exact_final_ho), 'k--', lw=1.5, label='Exact', alpha=0.7)
        ax.plot(x_ho, np.real(psi_cn_ho), COLORS_UNION["CN"], lw=1.5, label='CN')
        ax.set_xlabel("x"); ax.set_ylabel(r"Re[$\psi$]"); ax.set_title(f"Real Part: L2_Re={l2_re_c:.2e}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax = axes[1,0]; ax.plot(x_ho, np.imag(psi_exact_final_ho), 'k--', lw=1.5, label='Exact', alpha=0.7)
        ax.plot(x_ho, np.imag(psi_cn_ho), COLORS_UNION["RK4"], lw=1.5, label='CN')
        ax.set_xlabel("x"); ax.set_ylabel(r"Im[$\psi$]"); ax.set_title(f"Imag Part: L2_Im={l2_im_c:.2e}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax = axes[1,1]; err_c = np.abs(psi_cn_ho - psi_exact_final_ho)
        ax.semilogy(x_ho, err_c, COLORS_UNION["BE"], lw=1.0)
        ax.axhline(linf_c, color='red', ls=':', lw=1.0, alpha=0.7, label=f'Linf={linf_c:.2e}')
        ax.set_xlabel("x"); ax.set_ylabel(r"$|\psi_{num} - \psi_{exact}|$"); ax.set_title("Pointwise Error Distribution"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(cfg.outdir, "fig2c_ho_CN_vs_exact.png"), dpi=get_adaptive_dpi(4), bbox_inches='tight')
        plt.close(fig)
        print(f"  [Fig2c] CN vs Exact detail: L2={l2_c:.2e}, Linf={linf_c:.2e}")

    # Fig2d: 误差柱状图（只保留一个！）
    err_rows_ho = []
    for method in methods_all_ho:
        r = results_ho.get(method, {}); row = {"Method": method}
        if r.get("stable", False) and r.get("psi") is not None:
            l1_v, l2_v, linf_v = l1_l2_linf_error(r["psi"], psi_exact_final_ho, dx_ho)
            l2_re_v, l2_im_v, _ = l2_error_real_imag(r["psi"], psi_exact_final_ho, dx_ho)
            ref_n = float(np.sqrt(np.sum(np.abs(psi_exact_final_ho)**2) * dx_ho))
            row.update({"Mass": r["mass"], "L1": l1_v, "L2": l2_v, "Linf": linf_v,
                        "Rel_L2": l2_v/ref_n, "L2_Re": l2_re_v, "L2_Im": l2_im_v})
        else:
            row.update({"Mass": np.nan, "L1": np.nan, "L2": np.nan, "Linf": np.nan,
                        "Rel_L2": np.nan, "L2_Re": np.nan, "L2_Im": np.nan})
        err_rows_ho.append(row)
    df_err_ho = pd.DataFrame(err_rows_ho); df_err_ho.to_csv(os.path.join(cfg.outdir, "data_ho_errors.csv"), index=False)
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    metrics_ho = [("L1 Error","L1",True),("L2 Error","L2",True),("Linf Error","Linf",True),
                  ("Relative L2","Rel_L2",True),("L2 Real Part","L2_Re",True),("L2 Imag Part","L2_Im",True)]
    valid_m_ho = [m for m in methods_all_ho if np.isfinite(df_err_ho.loc[df_err_ho["Method"]==m,"L2"].values[0])]
    for idx,(title,col,ulog) in enumerate(metrics_ho):
        ax=axes[idx//3][idx%3]; vals=[df_err_ho.loc[df_err_ho["Method"]==m,col].values[0] for m in valid_m_ho]
        clrs=[COLORS_UNION.get(m,"#333") for m in valid_m_ho]; bars=ax.bar(range(len(valid_m_ho)),vals,color=clrs,edgecolor='white',lw=0.5)
        if ulog:
            vf=[v for v in vals if np.isfinite(v) and v>0]
            if vf: ax.set_yscale('log'); ax.set_ylim(min(vf)*0.5,max(vf)*5.0)
        ax.set_xticks(range(len(valid_m_ho))); ax.set_xticklabels(valid_m_ho,fontsize=9,rotation=10)
        ax.set_title(title,fontsize=11); ax.grid(True,alpha=0.3)
        for bi,(bar,val) in enumerate(zip(bars,vals)):
            if np.isfinite(val) and val>0:
                ax.text(bar.get_x()+bar.get_width()/2.,val*1.12,f'{val:.2e}',ha='center',va='bottom',
                       fontsize=6.5,fontweight='bold',bbox=dict(boxstyle='round,pad=0.1',facecolor=clrs[bi],alpha=0.7,edgecolor='none'))
    fig.suptitle("HO Coherent State - Complete Error Analysis vs Analytic Solution", fontsize=13,y=1.01)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(os.path.join(cfg.outdir,"fig2d_ho_error_analysis.png"),dpi=get_adaptive_dpi(6),bbox_inches='tight')
    plt.close(fig)
    print("  [Fig2d] Complete error analysis (6 metrics, single histogram)")

    # Fig2e-i: 误差演化、期望值、方差、相空间、时间序列
    def ho_exact_fn(ti): return _exact_ho_coherent(x_ho, ti, x0_ho, k0_ho)

    if "Crank-Nicolson" in results_hist_ho and results_hist_ho["Crank-Nicolson"].get("stable",False):
        cn_hist_ho = results_hist_ho["Crank-Nicolson"]["hist"]; cn_t_ho = results_hist_ho["Crank-Nicolson"]["t_hist"]
        plot_error_evolution(cn_t_ho, cn_hist_ho, ho_exact_fn, x_ho, dx_ho, cfg.outdir, "fig2e_ho_error_evol.png", title_prefix="HO Coherent (CN): ")
        print("  [Fig2e] Error evolution (CN)")
        x_vals_ho,p_vals_ho=[],[]
        for psi_t in cn_hist_ho:
            ev=compute_expectation_values(psi_t,x_ho,v_ho,dx_ho); x_vals_ho.append(ev["x"]); p_vals_ho.append(ev["p"])
        t_cl_ho=np.array(cn_t_ho); x_cl_ho=x0_ho*np.cos(t_cl_ho)+k0_ho*np.sin(t_cl_ho); p_cl_ho=k0_ho*np.cos(t_cl_ho)-x0_ho*np.sin(t_cl_ho)
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
        ax1.plot(cn_t_ho,x_vals_ho,'o-',color=COLORS_UNION["CN"],lw=1.5,markersize=3,label=r'Num $\langle x \rangle$')
        ax1.plot(t_cl_ho,x_cl_ho,'k--',lw=2.0,label=r'Classical $x_{cl}(t)$'); ax1.set_xlabel("Time"); ax1.set_ylabel("Position")
        ax1.set_title("HO Coherent: Position Expectation vs Classical"); ax1.legend(); ax1.grid(True,alpha=0.3)
        ax2.plot(cn_t_ho,p_vals_ho,'o-',color=COLORS_UNION["RK4"],lw=1.5,markersize=3,label=r'Num $\langle p \rangle$')
        ax2.plot(t_cl_ho,p_cl_ho,'k--',lw=2.0,label=r'Classical $p_{cl}(t)$'); ax2.set_xlabel("Time"); ax2.set_ylabel("Momentum")
        ax2.set_title("HO Coherent: Momentum Expectation vs Classical"); ax2.legend(); ax2.grid(True,alpha=0.3)
        fig.savefig(os.path.join(cfg.outdir,"fig2f_ho_expectations.png"),dpi=get_adaptive_dpi(2),bbox_inches='tight'); plt.close(fig)
        print("  [Fig2f] Expectation values (Numerical vs Classical elliptical orbit)")
        x_vars_ho,p_vars_ho=[],[]
        for psi_t in cn_hist_ho:
            prob=np.abs(psi_t)**2; xv=np.sum(x_ho*prob)*dx_ho; x_vars_ho.append(np.sum((x_ho-xv)**2*prob)*dx_ho)
            dpsi=np.zeros_like(psi_t,dtype=complex); dpsi[1:-1]=(psi_t[2:]-psi_t[:-2])/(2*dx_ho)
            pv=float(np.real(np.sum(np.conj(psi_t)*(-1j)*dpsi)*dx_ho))
            p_vars_ho.append(float(np.real(np.sum(np.conj(psi_t)*(1j*dpsi-pv)**2*psi_t)*dx_ho)))
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
        ax1.plot(cn_t_ho,x_vars_ho,'o-',color=COLORS_UNION["CN"],lw=1.5,markersize=3,label=r'Num $\langle (\Delta x)^2 \rangle$')
        ax1.axhline(0.5,color='k',ls='--',lw=1.5,alpha=0.7,label=r'Theory (= 1/2)'); ax1.set_xlabel("Time"); ax1.set_ylabel("Position Variance")
        ax1.set_title("Position Variance Evolution (Coherent State)"); ax1.legend(); ax1.grid(True,alpha=0.3)
        ax2.plot(cn_t_ho,p_vars_ho,'o-',color=COLORS_UNION["RK4"],lw=1.5,markersize=3,label=r'Num $\langle (\Delta p)^2 \rangle$')
        ax2.axhline(0.5,color='k',ls='--',lw=1.5,alpha=0.7,label=r'Theory (= 1/2)'); ax2.set_xlabel("Time"); ax2.set_ylabel("Momentum Variance")
        ax2.set_title("Momentum Variance Evolution (Coherent State)"); ax2.legend(); ax2.grid(True,alpha=0.3)
        fig.savefig(os.path.join(cfg.outdir,"fig2g_ho_variance.png"),dpi=get_adaptive_dpi(2),bbox_inches='tight'); plt.close(fig)
        print("  [Fig2g] Variance evolution (coherent state should be constant ~0.5)")
        from matplotlib.collections import LineCollection
        fig,ax=plt.subplots(figsize=(9,7),constrained_layout=True)
        points=np.array([x_vals_ho,p_vals_ho]).T.reshape(-1,1,2); segments=np.concatenate([points[:-1],points[1:]],axis=1)
        norm=plt.Normalize(cn_t_ho[0],cn_t_ho[-1]); lc=LineCollection(segments,cmap='viridis',norm=norm,lw=2,alpha=0.8); lc.set_array(np.array(cn_t_ho)); ax.add_collection(lc)
        tf=np.linspace(cn_t_ho[0],cn_t_ho[-1],200); ax.plot(x0_ho*np.cos(tf)+k0_ho*np.sin(tf),k0_ho*np.cos(tf)-x0_ho*np.sin(tf),'w--',lw=2.0,alpha=0.6,label='Classical ellipse')
        ax.scatter(x_vals_ho[0],p_vals_ho[0],c='lime',s=80,zorder=5,marker='o',edgecolors='black',label=f'Start (t={cn_t_ho[0]:.1f})')
        ax.scatter(x_vals_ho[-1],p_vals_ho[-1],c='red',s=80,zorder=5,marker='s',edgecolors='black',label=f'End (t={cn_t_ho[-1]:.1f})')
        cb=fig.colorbar(lc,ax=ax); cb.set_label("Time"); ax.set_xlabel(r"$\langle x \rangle$"); ax.set_ylabel(r"$\langle p \rangle$")
        ax.set_title("HO Coherent Phase Space (Elliptical Orbit)"); ax.legend(fontsize=9); ax.grid(True,alpha=0.3); ax.set_aspect('equal',adjustable='datalim')
        fig.savefig(os.path.join(cfg.outdir,"fig2h_ho_phase_space.png"),dpi=600,bbox_inches='tight'); plt.close(fig)
        print("  [Fig2h] Phase space trajectory (elliptical orbit!)")
        fig,axes=plt.subplots(3,3,figsize=(18,12),constrained_layout=True)
        n_snap_ho=min(9,len(cn_hist_ho)); snap_idx_ho=np.linspace(0,len(cn_hist_ho)-1,n_snap_ho,dtype=int)
        for i,si in enumerate(snap_idx_ho):
            ax=axes[i//3][i%3]; psi_s=cn_hist_ho[si]; psi_ex_s=_exact_ho_coherent(x_ho,cn_t_ho[si],x0_ho,k0_ho)
            ax.plot(x_ho,np.abs(psi_ex_s)**2,'k--',lw=1.2,alpha=0.6,label='Exact'); ax.plot(x_ho,np.abs(psi_s)**2,COLORS_UNION["CN"],lw=1.5,label='CN Num')
            xcl_s,_=_classical_traj_ho(cn_t_ho[si],x0_ho,k0_ho); ax.axvline(xcl_s,color='green',ls=':',lw=1.5,alpha=0.7)
            ax.set_xlim(-L_ho,L_ho); ax.set_title(f"t = {cn_t_ho[si]:.2f}",fontsize=10); ax.set_xlabel("x"); ax.set_ylabel(r"$|\psi|^2$"); ax.legend(fontsize=6); ax.grid(True,alpha=0.2)
        fig.suptitle("HO Coherent: Probability Density Evolution (CN vs Exact)",fontsize=14,y=1.02)
        fig.savefig(os.path.join(cfg.outdir,"fig2i_ho_time_series.png"),dpi=get_adaptive_dpi(9),bbox_inches='tight'); plt.close(fig)
        print("  [Fig2i] Probability density time series (9 snapshots)")

    # B组：newtdse风格图
    for mname in methods_all_ho:
        if mname in results_hist_ho and results_hist_ho[mname].get("stable",False):
            rh=results_hist_ho[mname]; t_arr=np.array(rh["t_hist"]); hist_arr=rh["hist"]
            l2list,linflist,prolist=[],[],[]
            for i,psi_t in enumerate(hist_arr):
                ti=t_arr[i] if i<len(t_arr) else T_final_ho
                try:
                    pe=ho_exact_fn(ti); l2list.append(float(np.sqrt(np.sum(np.abs(psi_t-pe)**2)*dx_ho)))
                    linflist.append(float(np.max(np.abs(psi_t-pe))))
                except Exception: l2list.append(np.nan); linflist.append(np.nan)
                prolist.append(probability_mass(psi_t,dx_ho))
            rh["l2"]=l2list; rh["linf"]=linflist; rh["prob"]=prolist; rh["runtime"]=0.0; rh["blow"]=None

    # Fig2j: 全方法误差曲线(semilogy)
    print("  Generating newtdse-style error curves...")
    fig,(ax_l2,ax_li)=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for mname in methods_all_ho:
        if not results_hist_ho.get(mname,{}).get("stable",False) or len(results_hist_ho[mname].get("hist",[]))==0: continue
        rh=results_hist_ho[mname]; t_arr=np.array(rh["t_hist"]); l2v=np.array(rh.get("l2",[])); liv=np.array(rh.get("linf",[]))
        clr=COLORS_UNION.get(mname,"#333"); ls=STYLES_UNION.get(mname,"-"); lbl=mname.replace("Split-Step-FFT","SSF").replace("Backward-Euler","BE").replace("Crank-Nicolson","CN")
        ax_l2.semilogy(t_arr,np.where(l2v>10,np.nan,l2v),color=clr,ls=ls,lw=1.8,label=lbl)
        ax_li.semilogy(t_arr,np.where(liv>10,np.nan,liv),color=clr,ls=ls,lw=1.8,label=lbl)
    for ax in [ax_l2,ax_li]:
        ax.axvline(np.pi,color="gray",ls=":",lw=1,alpha=0.6); ax.axvline(2*np.pi,color="gray",ls="--",lw=1,alpha=0.6)
    ax_l2.set_xlabel("t"); ax_l2.set_ylabel("L2 Error"); ax_l2.set_title("L2 Error vs Time — All Methods"); ax_l2.legend(fontsize=8); ax_l2.grid(True,which="both",alpha=0.3)
    ax_li.set_xlabel("t"); ax_li.set_ylabel("Linf Error"); ax_li.set_title("Linf Error vs Time — All Methods"); ax_li.legend(fontsize=8); ax_li.grid(True,which="both",alpha=0.3)
    fig.suptitle("HO Coherent: Error Over Time — All Methods",fontsize=12,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(os.path.join(cfg.outdir,"fig2j_newtdse_errors.png"),dpi=600,bbox_inches='tight'); plt.close(fig)
    print("  [Fig2j] All-method error curves (L2 + Linf, semilogy)")

    # Fig2k: 概率守恒图
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for mname in methods_all_ho:
        if not results_hist_ho.get(mname,{}).get("stable",False) or len(results_hist_ho[mname].get("hist",[]))==0: continue
        rh=results_hist_ho[mname]; t_arr=np.array(rh["t_hist"]); pv=np.array(rh.get("prob",[])); pvc=np.where(np.abs(pv)>10,np.nan,pv)
        clr=COLORS_UNION.get(mname,"#333"); ls=STYLES_UNION.get(mname,"-"); lbl=mname.replace("Split-Step-FFT","SSF").replace("Backward-Euler","BE").replace("Crank-Nicolson","CN")
        ax.plot(t_arr,pvc,color=clr,ls=ls,lw=1.8,label=lbl)
    ax.axhline(P0_ho,color="k",ls="--",lw=1.2,alpha=0.5,label=f"P0={P0_ho:.4f}")
    ax.set_xlabel("t"); ax.set_ylabel("int(|psi|^2) dx"); ax.set_title("HO Coherent: Probability Conservation — All Methods",fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3); ax.set_xlim(0,T_final_ho); fig.tight_layout()
    fig.savefig(os.path.join(cfg.outdir,"fig2k_newtdse_prob.png"),dpi=600,bbox_inches='tight'); plt.close(fig)
    print("  [Fig2k] Probability conservation (all methods)")

    # Fig2l: GridSpec 2×3快照对比 at t=pi
    print("  Generating snapshot comparison at t=pi...")
    t_snap_ho=np.pi; best_mi=None; best_ii=None
    for mname in m_stable_ho:
        if mname in results_hist_ho and results_hist_ho[mname].get("stable",False):
            ii=int(np.argmin(np.array(results_hist_ho[mname]["t_hist"])-t_snap_ho)); best_mi=mname; best_ii=ii; break
    if best_mi and best_ii is not None:
        t_snap_actual=results_hist_ho[best_mi]["t_hist"][best_ii]; psi_ex_snap=ho_exact_fn(t_snap_actual)
        fig=plt.figure(figsize=(14,9)); gs=GridSpec(2,3,figure=fig,hspace=0.38,wspace=0.30)
        quants_ho=[("Re(psi)",lambda p:np.real(p),(-1.1,1.1)),("Im(psi)",lambda p:np.imag(p),(-1.1,1.1)),("|psi|^2",lambda p:np.abs(p)**2,(-0.05,0.75))]
        for col,(qtitle,qfn,yl) in enumerate(quants_ho):
            ax=fig.add_subplot(gs[0,col]); Vn_bg=v_ho/(np.max(v_ho)+1e-30)*yl[1]*0.85; ax.fill_between(x_ho,yl[0],Vn_bg,alpha=0.07,color="gray")
            ax.plot(x_ho,qfn(psi_ex_snap),color=COLORS_UNION["Exact"],lw=2.2,label="Exact",zorder=6)
            for nm in m_stable_ho:
                if nm in results_hist_ho and results_hist_ho[nm].get("stable",False):
                    hist_nm=results_hist_ho[nm]["hist"]; yv=qfn(hist_nm[best_ii])
                    if np.nanmax(np.abs(yv))<1e6:
                        c=COLORS_UNION.get(nm,"#333"); s=STYLES_UNION.get(nm,"-"); nlbl=nm.replace("Split-Step-FFT","SSF").replace("Backward-Euler","BE").replace("Crank-Nicolson","CN")
                        ax.plot(x_ho,yv,color=c,ls=s,lw=1.6,alpha=0.85,label=nlbl)
            ax.set_xlim(-L_ho,L_ho); ax.set_ylim(*yl); ax.set_title(f"{qtitle}  (t = {t_snap_actual:.3f} ~= pi)"); ax.set_xlabel("x"); ax.legend(fontsize=7,loc="upper right"); ax.grid(True,alpha=0.25)
        for col,nm in enumerate(m_stable_ho[:3]):
            ax=fig.add_subplot(gs[1,col])
            if nm in results_hist_ho and results_hist_ho[nm].get("stable",False):
                err_field=np.abs(results_hist_ho[nm]["hist"][best_ii]-psi_ex_snap); c=COLORS_UNION.get(nm,"#333")
                ax.semilogy(x_ho,err_field,color=c,lw=1.8); nlbl=nm.replace("Split-Step-FFT","SSF").replace("Backward-Euler","BE").replace("Crank-Nicolson","CN"); ax.set_title(f"Pointwise error — {nlbl}")
            ax.set_xlim(-L_ho,L_ho); ax.set_xlabel("x"); ax.set_ylabel("|psi_num - psi_exact|"); ax.grid(True,which="both",alpha=0.25)
        fig.suptitle("Snapshot Comparison at t = pi (half period)",fontsize=13,fontweight="bold")
        fig.savefig(os.path.join(cfg.outdir,"fig2l_newtdse_snapshot.png"),bbox_inches="tight",dpi=130); plt.close(fig)
        print("  [Fig2l] Snapshot 2x3 at t~=pi")

    # Fig2m: 汇总表格
    print("  Generating summary table...")
    rows_tbl=[]
    for mname in methods_all_ho:
        rh=results_hist_ho.get(mname,{})
        l2_fin=linf_fin=dp_r=np.nan; stability_s="unstable"; blow_s="--"
        if rh.get("stable",False) and rh.get("l2") is not None:
            l2a=np.array(rh["l2"]); lifa=np.array(rh.get("linf",[])); pra=np.array(rh.get("prob",[])); valid=~np.isnan(l2a)
            if valid.any():
                il=int(np.where(valid)[0][-1]); l2_fin=l2a[il]; linf_fin=lifa[il] if il<lifa.shape[0] else np.nan
                pr_fin=pra[il] if il<pra.shape[0] else P0_ho
                if P0_ho>0: dp_r=abs(pr_fin-P0_ho)/P0_ho
            stability_s="stable"
        blow_info=rh.get("blow"); blow_s=f"t~{blow_info:.3f}" if blow_info is not None else "--"
        rows_tbl.append([mname.replace("Split-Step-FFT","SSF").replace("Backward-Euler","BE").replace("Crank-Nicolson","CN"),
            f"{l2_fin:.3e}" if np.isfinite(l2_fin) else "diverged", f"{linf_fin:.3e}" if np.isfinite(linf_fin) else "diverged",
            f"{dp_r:.3e}" if np.isfinite(dp_r) else "--", f"{rh.get('runtime',0.0):.2f}s", blow_s, stability_s])
    col_labels_tbl=["Method","L2 Error","Linf Error","deltaP/P0","Runtime","Blow-up","Stability"]
    fig,ax=plt.subplots(figsize=(13,3.4)); ax.axis("off")
    tbl=ax.table(cellText=rows_tbl,colLabels=col_labels_tbl,loc="center",cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.0,2.1)
    for j in range(len(col_labels_tbl)): tbl[(0,j)].set_facecolor("#2b3a55"); tbl[(0,j)].set_text_props(color="white",fontweight="bold")
    row_bg_tbl={"FTCS":"#fde8e8","Backward-Euler":"#fff3e0","Crank-Nicolson":"#e3f2fd","RK4":"#e8f5e9","Split-Step-FFT":"#f3e5f5"}
    for i,row in enumerate(rows_tbl):
        nm_key=methods_all_ho[i] if i<len(methods_all_ho) else row[0]; bg=row_bg_tbl.get(nm_key,"#fafafa")
        for j in range(len(col_labels_tbl)): tbl[(i+1,j)].set_facecolor(bg)
    fig.suptitle("Summary Table — HO Coherent State Methods",fontsize=13,fontweight="bold",y=0.97)
    fig.tight_layout(); fig.savefig(os.path.join(cfg.outdir,"fig2m_newtdse_summary.png"),bbox_inches="tight",dpi=130); plt.close(fig)
    print("  [Fig2m] Summary table")

    # C组：收敛性研究 (loglog L2 error vs dt, 在t=pi/2处测量)
    print("  Running convergence study (at t=pi/2)...")
    dt_list_conv=[0.020,0.010,0.005,0.002,0.001,0.0005]; T_conv=np.pi/2.0
    conv_methods=["Backward-Euler","Crank-Nicolson","RK4","Split-Step-FFT"]; conv_err_data={m:[] for m in conv_methods}
    ref_psi_conv=_exact_ho_coherent(x_ho,T_conv,x0_ho,k0_ho)
    for dt_c in dt_list_conv:
        Nt_c=int(round(T_conv/dt_c)); t_c=np.arange(0.0,T_conv+0.5*dt_c,dt_c)
        for m in conv_methods:
            try:
                _,hist_c=solve(m,psi0_ho,v_ho,x_ho,t_c,dx_ho,dt_c,store_every=len(t_c)-1)
                conv_err_data[m].append(float(np.sqrt(np.sum(np.abs(hist_c[-1]-ref_psi_conv)**2)*dx_ho)))
            except Exception: conv_err_data[m].append(np.nan)
        print(f"    dt={dt_c:.4f}  "+"  ".join(f"{m}={conv_err_data[m][-1]:.2e}" for m in conv_methods))
    fig,ax=plt.subplots(figsize=(8,5.5)); dt_arr_conv=np.array(dt_list_conv)
    conv_colors_map={"Backward-Euler":"#f77f00","Crank-Nicolson":"#2176ae","RK4":"#06a77d","Split-Step-FFT":"#9b2dca"}
    for m in conv_methods:
        err_arr=np.array(conv_err_data[m]); mask=np.isfinite(err_arr)&(err_arr>0)
        if mask.any(): ax.loglog(dt_arr_conv[mask],err_arr[mask],"o-",color=conv_colors_map[m],label=m.replace("Backward-Euler","BE").replace("Crank-Nicolson","CN").replace("Split-Step-FFT","SSF"),lw=2,ms=6)
    ref_dt_c=np.array([dt_list_conv[1],dt_list_conv[-1]])
    base_err=conv_err_data["Crank-Nicolson"][1] if len(conv_err_data["Crank-Nicolson"])>1 and np.isfinite(conv_err_data["Crank-Nicolson"][1]) else 1e-4
    for p,ls_p,col_p in [(1,"--","#aaa"),(2,"-","#888"),(4,":","#666")]:
        ax.loglog(ref_dt_c,ref_dt_c[0]**p/ref_dt_c**p*base_err,ls=ls_p,color=col_p,lw=1.2,alpha=0.6,label=f"O(dt^{p})")
    ax.set_xlabel("Time step  dt"); ax.set_ylabel("L2 Error at t = pi/2")
    ax.set_title("Convergence Study — L2 Error vs dt (HO Coherent)",fontsize=12,fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True,which="both",alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(cfg.outdir,"fig2n_conv.png"),bbox_inches="tight",dpi=600); plt.close(fig)
    print("  [Fig2n] Convergence study (loglog, O(dt^p) reference lines)")

    # D组：newtdse风格的GIF（3个独立GIF）
    if cfg.save_gif:
        print("  Generating newtdse-style GIFs (3 independent GIFs)...")
        ANIM_METHODS_HO=["Crank-Nicolson","Split-Step-FFT"]; ANIM_LABELS_HO={"Crank-Nicolson":"CN","Split-Step-FFT":"SSF","Exact":"Exact"}
        if "Crank-Nicolson" in results_hist_ho and results_hist_ho["Crank-Nicolson"].get("stable",False):
            t_grid_ho=np.array(results_hist_ho["Crank-Nicolson"]["t_hist"])
            nf_anim_ho=min(len(results_hist_ho["Crank-Nicolson"]["hist"]),len([ho_exact_fn(tt) for tt in t_grid_ho]))
        else: t_grid_ho=np.arange(0,T_final_ho,T_final_ho/100); nf_anim_ho=100
        exact_frames_ho=[ho_exact_fn(tt) for tt in t_grid_ho]

        def make_ho_gif(qty_key,ylabel,title_str,gif_filename):
            fig,ax=plt.subplots(figsize=(9,4.5)); cap_width_ho=2.0
            ax.set_xlim(-L_ho+cap_width_ho-0.3,L_ho-cap_width_ho+0.3); ax.set_xlabel("x"); ax.set_ylabel(ylabel)
            if qty_key=="real": fn=np.real; yl=(-1.1,1.1)
            elif qty_key=="imag": fn=np.imag; yl=(-1.1,1.1)
            else: fn=lambda p: np.abs(p)**2; yl=(-0.05,0.75)
            ax.set_ylim(*yl); Vn_gif=v_ho/(np.max(v_ho)+1e-30)*yl[1]*0.85
            ax.fill_between(x_ho,yl[0],Vn_gif,alpha=0.07,color="gray"); ax.plot(x_ho,Vn_gif,color="gray",lw=0.8,ls="--",alpha=0.5)
            lines_ho={}
            for m in ANIM_METHODS_HO:
                c=COLORS_UNION.get(m,"#333"); ls=STYLES_UNION.get(m,"-"); l,=ax.plot([],[],color=c,ls=ls,lw=2.0,label=ANIM_LABELS_HO.get(m,m),zorder=3); lines_ho[m]=l
            lex_ho,=ax.plot([],[],color=COLORS_UNION["Exact"],lw=1.6,ls="-",label="Exact",zorder=5); lines_ho["Exact"]=lex_ho
            ax.legend(loc="upper right",fontsize=9,framealpha=0.85); time_txt_ho=ax.text(0.02,0.95,"",transform=ax.transAxes,fontsize=10,va="top")
            ax.set_title(title_str); fig.tight_layout()
            def init():
                for l in lines_ho.values(): l.set_data([],[]); time_txt_ho.set_text(""); return list(lines_ho.values())+[time_txt_ho]
            def update(i):
                tt=t_grid_ho[i]
                for m in ANIM_METHODS_HO:
                    if m in results_hist_ho and results_hist_ho[m].get("stable",False):
                        hm=results_hist_ho[m]["hist"]; yv=fn(hm[i]) if i<len(hm) else None
                        if yv is not None and np.nanmax(np.abs(yv))<1e6: lines_ho[m].set_data(x_ho,yv)
                lines_ho["Exact"].set_data(x_ho,fn(exact_frames_ho[i])); time_txt_ho.set_text(f"t = {tt:.3f}   (t/T = {tt/T_final_ho:.3f})")
                return list(lines_ho.values())+[time_txt_ho]
            ani=FuncAnimation(fig,update,frames=nf_anim_ho,init_func=init,blit=True,interval=60,repeat=False)
            ani.save(os.path.join(cfg.outdir,gif_filename),writer=PillowWriter(fps=18),dpi=300); plt.close(fig)
            print(f"    Saved {gif_filename}")
        make_ho_gif("real","Re(psi)","Real part Re(psi) — HO Coherent","gif2_real.gif")
        make_ho_gif("imag","Im(psi)","Imaginary part Im(psi) — HO Coherent","gif2_imag.gif")
        make_ho_gif("dens","|psi|^2","Probability density |psi|^2 — HO Coherent","gif2_density.gif")

    print(f"\n  Physical Discussion:")
    print(f"  Coherent state in harmonic oscillator: shape preserved, center follows classical ellipse.")
    print(f"  Period T = 2*pi (full revival at t=2pi)")
    print(f"  ════════════════════════════════════════════")
    print(f"  Grid: N={N_ho}, dt={dt_ho}, steps={Nt_ho}, Domain=[{-L_ho},{L_ho}]")


# ──────────────────────────────────────────────────────────────
# 二维谐振子相干态精确解析解
# ──────────────────────────────────────────────────────────────

def _exact_2d_ho_coherent(X,Y,t,x0,y0,px0,py0):
    """二维各向同性谐振子相干态的精确解析解。V(X,Y)=0.5*(X^2+Y^2)"""
    xc=x0*np.cos(t)+px0*np.sin(t); yc=y0*np.cos(t)+py0*np.sin(t)
    pcx=-x0*np.sin(t)+px0*np.cos(t); pcy=-y0*np.sin(t)+py0*np.cos(t)
    phi0=(1/np.pi)**0.5*np.exp(-((X-xc)**2+(Y-yc)**2)/2.0)
    phase=np.exp(1j*(pcx*(X-xc)+pcy*(Y-yc)))
    return phi0*phase

def _classical_traj_2d_ho(t,x0,y0,px0,py0):
    """返回二维谐振子经典轨迹."""
    return (x0*np.cos(t)+px0*np.sin(t), y0*np.cos(t)+py0*np.sin(t),
            -x0*np.sin(t)+px0*np.cos(t), -y0*np.sin(t)+py0*np.cos(t))

def _expectations_2d(psi,X,Y,KX,KY,dx,dy):
    """计算二维期望值。"""
    from numpy.fft import fft2, ifft2
    prob=np.sum(np.abs(psi)**2)*dx*dy
    x_avg=float(np.real(np.sum(np.conj(psi)*X*psi)*dx*dy))
    y_avg=float(np.real(np.sum(np.conj(psi)*Y*psi)*dx*dy))
    pk=fft2(psi); ddx=ifft2(1j*KX*pk); ddy=ifft2(1j*KY*pk)
    return dict(prob=prob,x_avg=x_avg,y_avg=y_avg,
               px_avg=float(np.real(np.sum(np.conj(psi)*(-1j)*ddx)*dx*dy)),
               py_avg=float(np.real(np.sum(np.conj(psi)*(-1j)*ddy)*dx*dy)))


# ──────────────────────────────────────────────────────────────
# 实验③：二维各向同性谐振子相干态（来自case7.py）
# ──────────────────────────────────────────────────────────────

def experiment_2d_ho_coherent(cfg):
    """二维各向同性谐振子相干态 — ADI+SSF双方法 + 解析解验证"""
    print("="*60)
    print("Exp-03: 2D Isotropic HO Coherent State (from case7.py)")
    print("="*60)

    nx_2d,ny_2d=256,256; Lx_2d,Ly_2d=12.0,12.0
    X_2d,Y_2d,x_2d,y_2d,dx_2d,dy_2d,KX_2d,KY_2d=make_2d_grid(nx_2d,ny_2d,-Lx_2d/2,Lx_2d/2,-Ly_2d/2,Ly_2d/2)
    T_2d=6.0; dt_2d=0.005; Nt_2d=int(T_2d/dt_2d); t_2d=np.arange(0.0,T_2d+0.5*dt_2d,dt_2d)
    x0_2d,y0_2d,px0_2d,py0_2d=2.0,0.0,0.0,1.0
    print(f"  Grid: {nx_2d}x{ny_2d}, domain=[{-Lx_2d/2},{Lx_2d/2}]x[{-Ly_2d/2},{Ly_2d/2}]")
    print(f"  Time: T={T_2d}, dt={dt_2d}, steps={Nt_2d}")
    print(f"  IC: center=({x0_2d},{y0_2d}), momentum=({px0_2d},{py0_2d})")

    V_2d=0.5*(X_2d**2+Y_2d**2)
    # 直接使用解析解构造初态，确保与后续误差比较基准完全一致
    psi0_2d=_exact_2d_ho_coherent(X_2d,Y_2d,0.0,x0_2d,y0_2d,px0_2d,py0_2d)
    print(f"  Initial check: max|psi0 - exact(0)| = {np.max(np.abs(psi0_2d-_exact_2d_ho_coherent(X_2d,Y_2d,0.0,x0_2d,y0_2d,px0_2d,py0_2d))):.2e}")

    print("  Running ADI..."); _,hist_adi_2d=solve_2d(psi0_2d,V_2d,KX_2d,KY_2d,t_2d,dt_2d,dx_2d,dy_2d,store_every=len(t_2d)-1,method="adi"); psi_adi_2d=hist_adi_2d[-1]
    print("  Running SSF..."); _,hist_ssf_2d=solve_2d(psi0_2d,V_2d,KX_2d,KY_2d,t_2d,dt_2d,dx_2d,dy_2d,store_every=len(t_2d)-1,method="split-step-fft"); psi_ssf_2d=hist_ssf_2d[-1]
    psi_exact_2d=_exact_2d_ho_coherent(X_2d,Y_2d,T_2d,x0_2d,y0_2d,px0_2d,py0_2d)
    dpi_2d=get_adaptive_dpi(3); extent_2d=[x_2d.min(),x_2d.max(),y_2d.min(),y_2d.max()]

    # A组：ADI/SSF三值热图
    fig,axes=plt.subplots(1,3,figsize=(20,6),constrained_layout=True)
    im0=axes[0].imshow(np.abs(psi_adi_2d)**2,origin='lower',extent=extent_2d,cmap='viridis',aspect='equal')
    axes[0].set_title(r"ADI: $|\psi|^2$",fontsize=13); plt.colorbar(im0,ax=axes[0],shrink=0.85)
    im1=axes[1].imshow(np.real(psi_adi_2d),origin='lower',extent=extent_2d,cmap='RdBu_r',aspect='equal',vmin=-np.max(np.abs(np.real(psi_adi_2d))),vmax=np.max(np.abs(np.real(psi_adi_2d))))
    axes[1].set_title(r"ADI: $\mathrm{Re}[\psi]$",fontsize=13); plt.colorbar(im1,ax=axes[1],shrink=0.85)
    im2=axes[2].imshow(np.imag(psi_adi_2d),origin='lower',extent=extent_2d,cmap='RdBu_r',aspect='equal',vmin=-np.max(np.abs(np.imag(psi_adi_2d))),vmax=np.max(np.abs(np.imag(psi_adi_2d))))
    axes[2].set_title(r"ADI: $\mathrm{Im}[\psi]$",fontsize=13); plt.colorbar(im2,ax=axes[2],shrink=0.85)
    for ax in axes: ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.savefig(os.path.join(cfg.outdir,"fig3a_2d_adi.png"),dpi=dpi_2d,bbox_inches='tight'); plt.close(fig)
    print("  [Fig3a] ADI three-value heatmap")

    fig,axes=plt.subplots(1,3,figsize=(20,6),constrained_layout=True)
    im0=axes[0].imshow(np.abs(psi_ssf_2d)**2,origin='lower',extent=extent_2d,cmap='viridis',aspect='equal')
    axes[0].set_title(r"SSF: $|\psi|^2$",fontsize=13); plt.colorbar(im0,ax=axes[0],shrink=0.85)
    im1=axes[1].imshow(np.real(psi_ssf_2d),origin='lower',extent=extent_2d,cmap='RdBu_r',aspect='equal',vmin=-np.max(np.abs(np.real(psi_ssf_2d))),vmax=np.max(np.abs(np.real(psi_ssf_2d))))
    axes[1].set_title(r"SSF: $\mathrm{Re}[\psi]$",fontsize=13); plt.colorbar(im1,ax=axes[1],shrink=0.85)
    im2=axes[2].imshow(np.imag(psi_ssf_2d),origin='lower',extent=extent_2d,cmap='RdBu_r',aspect='equal',vmin=-np.max(np.abs(np.imag(psi_ssf_2d))),vmax=np.max(np.abs(np.imag(psi_ssf_2d))))
    axes[2].set_title(r"SSF: $\mathrm{Im}[\psi]$",fontsize=13); plt.colorbar(im2,ax=axes[2],shrink=0.85)
    for ax in axes: ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.savefig(os.path.join(cfg.outdir,"fig3b_2d_ssf.png"),dpi=dpi_2d,bbox_inches='tight'); plt.close(fig)
    print("  [Fig3b] SSF three-value heatmap")

    # Fig3c: 误差分析
    diff_adi_2d=psi_adi_2d-psi_exact_2d; diff_ssf_2d=psi_ssf_2d-psi_exact_2d
    l2_adi_2d=float(np.sqrt(np.sum(np.abs(diff_adi_2d)**2)*dx_2d*dy_2d)); l2_ssf_2d=float(np.sqrt(np.sum(np.abs(diff_ssf_2d)**2)*dx_2d*dy_2d))
    ref_norm_2d=float(np.sqrt(np.sum(np.abs(psi_exact_2d)**2)*dx_2d*dy_2d))
    rel_adi_2d=l2_adi_2d/ref_norm_2d; rel_ssf_2d=l2_ssf_2d/ref_norm_2d
    linf_adi_2d=float(np.max(np.abs(diff_adi_2d))); linf_ssf_2d=float(np.max(np.abs(diff_ssf_2d)))
    mass_adi_2d=probability_mass(psi_adi_2d.flatten(),dx_2d); mass_ssf_2d=probability_mass(psi_ssf_2d.flatten(),dx_2d); mass_ex_2d=probability_mass(psi_exact_2d.flatten(),dx_2d)
    fig,axes=plt.subplots(2,2,figsize=(15,12),constrained_layout=True)
    im00=axes[0,0].imshow(np.abs(diff_adi_2d),origin='lower',extent=extent_2d,cmap='hot',aspect='equal')
    axes[0,0].set_title(r"$|\psi_{ADI}-\psi_{exact}|$",fontsize=12); plt.colorbar(im00,ax=axes[0,0],shrink=0.85)
    im01=axes[0,1].imshow(np.abs(diff_ssf_2d),origin='lower',extent=extent_2d,cmap='hot',aspect='equal')
    axes[0,1].set_title(r"$|\psi_{SSF}-\psi_{exact}|$",fontsize=12); plt.colorbar(im01,ax=axes[0,1],shrink=0.85)
    metrics_names_2d=['L2 Error','Rel. L2','Linf Error','Mass']; adi_vals_2d=[l2_adi_2d,rel_adi_2d,linf_adi_2d,mass_adi_2d]; ssf_vals_2d=[l2_ssf_2d,rel_ssf_2d,linf_ssf_2d,mass_ssf_2d]; exact_vals_2d=[0,0,0,mass_ex_2d]
    x_pos_2d=np.arange(len(metrics_names_2d)); w_2d=0.25
    axes[1,0].bar(x_pos_2d-w_2d,adi_vals_2d,w_2d,label='ADI',color='#2176ae')
    axes[1,0].bar(x_pos_2d,ssf_vals_2d,w_2d,label='SSF',color='#9b2dca')
    axes[1,0].bar(x_pos_2d+w_2d,exact_vals_2d,w_2d,label='Exact',color='#06a77d',alpha=0.5)
    axes[1,0].set_xticks(x_pos_2d); axes[1,0].set_xticklabels(metrics_names_2d,fontsize=9)
    axes[1,0].set_title("Quantitative Comparison: ADI vs SSF vs Exact"); axes[1,0].legend(); axes[1,0].grid(True,alpha=0.3)
    for i,(av,sv) in enumerate(zip(adi_vals_2d,ssf_vals_2d)): axes[1,0].text(x_pos_2d[i]-w_2d,av*1.05,f'{av:.2e}',ha='center',fontsize=6.5); axes[1,0].text(x_pos_2d[i],sv*1.05,f'{sv:.2e}',ha='center',fontsize=6.5)
    mid_j_2d=ny_2d//2
    axes[1,1].plot(x_2d,np.abs(psi_exact_2d[mid_j_2d,:])**2,'k-',lw=1.8,label='Exact')
    axes[1,1].plot(x_2d,np.abs(psi_adi_2d[mid_j_2d,:])**2,'#2176ae',lw=1.5,ls='-',label='ADI')
    axes[1,1].plot(x_2d,np.abs(psi_ssf_2d[mid_j_2d,:])**2,'#9b2dca',lw=1.5,ls='--',label='SSF')
    axes[1,1].set_xlabel("x"); axes[1,1].set_ylabel(r"$|\psi(y=0)|^2$"); axes[1,1].set_title("Cross-section at y = 0"); axes[1,1].legend(fontsize=9); axes[1,1].grid(True,alpha=0.3)
    fig.savefig(os.path.join(cfg.outdir,"fig3c_2d_error_analysis.png"),dpi=get_adaptive_dpi(4),bbox_inches='tight'); plt.close(fig)
    print(f"  [Fig3c] Error analysis: ADI L2={l2_adi_2d:.2e}, SSF L2={l2_ssf_2d:.2e}")

    # B组：case7风格新增图
    print("  Running solver with history for trajectory analysis...")
    store_every_2d=10; _,hist_traj_2d=solve_2d(psi0_2d,V_2d,KX_2d,KY_2d,t_2d,dt_2d,dx_2d,dy_2d,store_every=store_every_2d,method="split-step-fft")
    times_traj_2d=np.linspace(0,T_2d,len(hist_traj_2d)); traj_num_2d=[]; traj_ana_2d=[]; metrics_list_2d=[]
    for i,psi_t in enumerate(hist_traj_2d):
        t_val=times_traj_2d[i]; psi_ex_t=_exact_2d_ho_coherent(X_2d,Y_2d,t_val,x0_2d,y0_2d,px0_2d,py0_2d)
        L2_t=float(np.sqrt(np.sum(np.abs(psi_t-psi_ex_t)**2)*dx_2d*dy_2d)); Linf_t=float(np.max(np.abs(psi_t-psi_ex_t)))
        exp_num_t=_expectations_2d(psi_t,X_2d,Y_2d,KX_2d,KY_2d,dx_2d,dy_2d)
        xc_t,yc_t,pcx_t,pcy_t=_classical_traj_2d_ho(t_val,x0_2d,y0_2d,px0_2d,py0_2d)
        metrics_list_2d.append((t_val,L2_t,Linf_t,exp_num_t,dict(x_avg=xc_t,y_avg=yc_t,px_avg=pcx_t,py_avg=pcy_t)))
        traj_num_2d.append((exp_num_t['x_avg'],exp_num_t['y_avg'])); traj_ana_2d.append((xc_t,yc_t))
    traj_num_2d=np.array(traj_num_2d); traj_ana_2d=np.array(traj_ana_2d)
    fig,ax=plt.subplots(figsize=(8,7),constrained_layout=True)
    ax.plot(traj_ana_2d[:,0],traj_ana_2d[:,1],'k-',lw=2.0,label='Analytic trajectory')
    ax.plot(traj_num_2d[:,0],traj_num_2d[:,1],'--',color='#2176ae',lw=1.8,label='Numerical (SSF)')
    ax.scatter([x0_2d],[y0_2d],c='green',s=80,zorder=5,marker='o',edgecolors='black',label=f'Start ({x0_2d},{y0_2d})')
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("2D HO Coherent State Center Trajectory (Numerical vs Analytic)")
    ax.legend(fontsize=9); ax.grid(True,alpha=0.3); ax.set_aspect('equal',adjustable='datalim')
    fig.savefig(os.path.join(cfg.outdir,"fig3d_trajectory.png"),dpi=150,bbox_inches='tight'); plt.close(fig)
    print("  [Fig3d] Coherent state center trajectory")
    metrics_path=os.path.join(cfg.outdir,"metrics_2d_ho.txt")
    with open(metrics_path,'w') as mf:
        mf.write("# t  L2_error  Linf_error  prob  <x>  <y>  <px>  <py>  x_exact  y_exact\n")
        for t,L2,Linf,exp_num,exp_ex in metrics_list_2d:
            mf.write(f"{t:.6f} {L2:.6e} {Linf:.6e} {exp_num['prob']:.12f} {exp_num['x_avg']:.6f} {exp_num['y_avg']:.6f} {exp_num['px_avg']:.6f} {exp_num['py_avg']:.6f} {exp_ex['x_avg']:.6f} {exp_ex['y_avg']:.6f}\n")
    print("  [Metrics] Saved metrics_2d_ho.txt")

    # C组：case7风格的GIF（逐帧PNG -> PIL归一化 -> imageio.mimsave）
    if cfg.save_gif:
        print("  Generating case7-style GIF (frame-by-frame PNG -> normalized GIF)...")
        tmp_dir=tempfile.mkdtemp(prefix="gif2d_frames_"); frames_gif=[]
        gif_store_every=max(1,Nt_2d//60)
        _,hist_gif_2d=solve_2d(psi0_2d,V_2d,KX_2d,KY_2d,t_2d,dt_2d,dx_2d,dy_2d,store_every=gif_store_every,method="split-step-fft")
        times_gif_2d=np.linspace(0,T_2d,len(hist_gif_2d))
        for i,psi_t in enumerate(hist_gif_2d):
            t_val=times_gif_2d[i]; psi_ex_t=_exact_2d_ho_coherent(X_2d,Y_2d,t_val,x0_2d,y0_2d,px0_2d,py0_2d)
            L2_frame=float(np.sqrt(np.sum(np.abs(psi_t-psi_ex_t)**2)*dx_2d*dy_2d)); Linf_frame=float(np.max(np.abs(psi_t-psi_ex_t)))
            fig,axs=plt.subplots(1,3,figsize=(15,4))
            im0=axs[0].imshow(np.abs(psi_t)**2,origin='lower',extent=extent_2d,cmap='viridis',aspect='equal'); axs[0].set_title('Numerical density'); plt.colorbar(im0,ax=axs[0])
            im1=axs[1].imshow(np.abs(psi_ex_t)**2,origin='lower',extent=extent_2d,cmap='viridis',aspect='equal'); axs[1].set_title('Analytic density'); plt.colorbar(im1,ax=axs[1])
            cs=axs[2].contourf(X_2d,Y_2d,np.abs(psi_t)**2-np.abs(psi_ex_t)**2,levels=20,cmap='RdBu_r',extent=extent_2d); axs[2].set_title('Difference (num - ana)'); plt.colorbar(cs,ax=axs[2])
            fig.suptitle(f'2D HO Coherent t={t_val:.3f} L2={L2_frame:.2e} Linf={Linf_frame:.2e}')
            fname=os.path.join(tmp_dir,f'frame_{i:04d}.png'); fig.savefig(fname,dpi=150,bbox_inches='tight'); plt.close(fig)
            frames_gif.append(PILImage.open(fname).convert('RGBA'))
        widths,heights=zip(*(im.size for im in frames_gif)); max_w,max_h=max(widths),max(heights)
        norm_frames=[]
        for im in frames_gif:
            bg=PILImage.new('RGBA',(max_w,max_h),(255,255,255,255)); x_off=(max_w-im.size[0])//2; y_off=(max_h-im.size[1])//2
            bg.paste(im,(x_off,y_off),im); norm_frames.append(np.array(bg.convert('RGB')))
        imageio.mimsave(os.path.join(cfg.outdir,"gif_2d_ho_coherent.gif"),norm_frames,duration=0.08)
        print(f"  [GIF3] Saved gif_2d_ho_coherent.gif ({len(norm_frames)} frames)")
        shutil.rmtree(tmp_dir); print("  Cleaned up temporary frame PNGs")

    print(f"  Grid: {nx_2d}x{ny_2d}, dt={dt_2d}, steps={Nt_2d}, Domain=[{-Lx_2d/2},{Lx_2d/2}]x[{-Ly_2d/2},{Ly_2d/2}]")


# ──────────────────────────────────────────────────────────────
# 实验④：Von Neumann 稳定性扫描（详细版）【不变】
# ──────────────────────────────────────────────────────────────

def experiment_stability_detailed(cfg):
    """Von Neumann 稳定性分析 — 详细扫描"""
    print("="*60); print("Exp-04: Von Neumann Stability Analysis (Detailed)"); print("="*60)
    ns=[128,256,512,1024]; dts=[0.0005,0.001,0.005,0.01,0.02,0.05,0.1]; t_end=1.0
    methods=["FTCS","Backward-Euler","Crank-Nicolson","RK4","Split-Step-FFT"]; rows=[]
    for n in tqdm(ns,desc="Grid sizes"):
        x,dx=grid(-20.0,20.0,n); psi0=gaussian_wavepacket(x,-5.0,1.0,2.0,dx); v=potential_free(x)
        for dt_val in dts:
            t_arr=np.arange(0.0,t_end+0.5*dt_val,dt_val)
            for method in methods:
                try:
                    _,hist=solve(method,psi0,v,x,t_arr,dx,dt_val,store_every=len(t_arr)-1)
                    mass=probability_mass(hist[-1],dx); peak=float(np.max(np.abs(hist[-1])))
                    stable=abs(mass-1.0)<0.5 and (peak<100 if np.isfinite(peak) else False)
                except Exception: stable=False; mass=np.nan; peak=np.nan
                rows.append({"method":method,"n":n,"dx":dx,"dt":dt_val,"stable":stable,"mass_final":mass,"peak_amp":peak,"mu":dt_val/dx**2 if dx>0 else np.nan})
    df=pd.DataFrame(rows); df.to_csv(os.path.join(cfg.outdir,"data_stability.csv"),index=False)
    dpi=get_adaptive_dpi(1); fig,ax=plt.subplots(figsize=(13,7),constrained_layout=True)
    method_order=methods; colors=["#9B9B9B","#2E86AB","#1B998B","#F18F01","#C73E1D"]
    for i,m in enumerate(method_order):
        sub=df[df["method"]==m]; stable_mask=sub["stable"]==True
        ax.scatter(sub.loc[stable_mask,"dt"],[i]*int(stable_mask.sum()),color=colors[i],marker='o',s=55,zorder=3,label=m if i==0 else None)
        unstable_mask=~stable_mask
        if unstable_mask.sum()>0:
            ax.scatter(sub.loc[unstable_mask,"dt"],[i]*int(unstable_mask.sum()),color=colors[i],marker='x',s=55,linewidths=2,zorder=3)
    ax.set_yticks(range(len(method_order))); ax.set_yticklabels(method_order,fontsize=10)
    ax.set_xlabel(r"$\Delta t$",fontsize=12); ax.set_title("Von Neumann Stability Map (o=stable, x=unstable)",fontsize=13)
    ax.grid(True,alpha=0.3)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],marker='o',color='gray',label='Stable',markersize=8,ls='None'),Line2D([0],[0],marker='x',color='gray',label='Unstable',markersize=8,mew=2,ls='None')],loc='lower right',fontsize=10)
    fig.savefig(os.path.join(cfg.outdir,"fig4a_stability_map.png"),dpi=dpi,bbox_inches='tight'); plt.close(fig)
    print("  [Fig4a] Stability map (stability vs dt)")
    fig,ax=plt.subplots(figsize=(13,7),constrained_layout=True)
    for i,m in enumerate(method_order):
        sub=df[df["method"]==m]; stable_mask=sub["stable"]==True
        mu_stable=sub.loc[stable_mask,"mu"].values if stable_mask.sum()>0 else []
        mu_unstable=sub.loc[~stable_mask,"mu"].values if (~stable_mask).sum()>0 else []
        if len(mu_stable)>0: ax.scatter(mu_stable,[i]*len(mu_stable),color=colors[i],marker='o',s=50,zorder=3)
        if len(mu_unstable)>0: ax.scatter(mu_unstable,[i]*len(mu_unstable),color=colors[i],marker='x',s=50,linewidths=2,zorder=3)
    ax.axvline(0.5,color='#9B9B9B',ls=':',lw=2,alpha=0.7,label=r'Theoretical FTCS limit: $\mu_c=0.5$')
    ax.set_yticks(range(len(method_order))); ax.set_yticklabels(method_order,fontsize=10)
    ax.set_xlabel(r"Stability Parameter $\mu = \Delta t / \Delta x^2$",fontsize=12); ax.set_title("Stability Boundary in Stability Parameter Space",fontsize=13)
    ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
    fig.savefig(os.path.join(cfg.outdir,"fig4b_stability_cfl.png"),dpi=dpi,bbox_inches='tight'); plt.close(fig)
    print("  [Fig4b] Stability Parameter boundary")


# ──────────────────────────────────────────────────────────────
# 实验⑤：Crank-Nicolson 收敛性验证（详细版）【不变】
# ──────────────────────────────────────────────────────────────

def experiment_convergence_detailed(cfg):
    """CN格式收敛性 — 多网格 + 多时间步长"""
    print("="*60); print("Exp-05: Crank-Nicolson Convergence Verification (Detailed)"); print("="*60)
    ns=[64,128,256,512,1024,2048]; dt_fixed=0.0002; t_end=1.0; rows_space=[]
    for n in tqdm(ns,desc="Spatial convergence"):
        x,dx=grid(-20.0,20.0,n); t=np.arange(0.0,t_end+0.5*dt_fixed,dt_fixed)
        x0,sigma,k0=-5.0,1.0,2.5; psi0=gaussian_wavepacket(x,x0,sigma,k0,dx); v=potential_free(x)
        psi_exact=exact_free_gaussian(x,t_end,x0,sigma,k0,dx); _,hist=solve("Crank-Nicolson",psi0,v,x,t,dx,dt_fixed,store_every=len(t)-1)
        l1,l2,linf=l1_l2_linf_error(hist[-1],psi_exact,dx); mass=probability_mass(hist[-1],dx)
        rows_space.append({"n":n,"dx":dx,"L1":l1,"L2":l2,"Linf":linf,"mass":mass,"mass_err":abs(mass-1.0)})
    df_space=pd.DataFrame(rows_space)
    n_fixed=1024; dts=[0.002,0.001,0.0005,0.0002,0.0001]; rows_time=[]
    x,dx=grid(-20.0,20.0,n_fixed); x0,sigma,k0=-5.0,1.0,2.5
    psi0=gaussian_wavepacket(x,x0,sigma,k0,dx); v=potential_free(x); psi_exact=exact_free_gaussian(x,t_end,x0,sigma,k0,dx)
    for dt_val in tqdm(dts,desc="Temporal convergence"):
        t=np.arange(0.0,t_end+0.5*dt_val,dt_val); _,hist=solve("Crank-Nicolson",psi0,v,x,t,dx,dt_val,store_every=len(t)-1)
        l1,l2,linf=l1_l2_linf_error(hist[-1],psi_exact,dx); mass=probability_mass(hist[-1],dx)
        rows_time.append({"dt":dt_val,"L1":l1,"L2":l2,"Linf":linf,"mass":mass,"mass_err":abs(mass-1.0)})
    df_time=pd.DataFrame(rows_time)
    dpi=get_adaptive_dpi(2)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5.5),constrained_layout=True)
    ax1.loglog(df_space["dx"],df_space["L2"],'-o',color="#1B998B",lw=2,markersize=7)
    ref_x=df_space["dx"].values; ax1.loglog(ref_x,0.5*ref_x**2,'--',color="#F18F01",lw=1.5,label=r'$\propto \Delta x^2$')
    ax1.set_xlabel(r"$\Delta x$"); ax1.set_ylabel("L2 Error"); ax1.set_title("Spatial Convergence: CN (Fixed dt)"); ax1.legend(); ax1.grid(True,which="both",alpha=0.3)
    ax2.semilogy(df_space["n"],df_space["mass_err"],'-o',color="#2E86AB",lw=2,markersize=7)
    ax2.set_xlabel("Grid Size n"); ax2.set_ylabel("|M - 1|"); ax2.set_title("Mass Conservation Error (Spatial)"); ax2.grid(True,alpha=0.3)
    fig.savefig(os.path.join(cfg.outdir,"fig5a_convergence_space.png"),dpi=dpi,bbox_inches='tight'); plt.close(fig)
    print("  [Fig5a] Spatial convergence order")
    if len(df_space)>=3:
        dx_vals=df_space["dx"].values; l2_vals=df_space["L2"].values
        orders=[]
        for i in range(len(dx_vals)-1):
            if l2_vals[i+1]>0 and dx_vals[i+1]>0:
                p=np.log(l2_vals[i]/l2_vals[i+1])/np.log(dx_vals[i]/dx_vals[i+1]); orders.append(p)
        if orders: print(f"  Measured spatial convergence order: ~{np.mean(orders[-3:]):.2f} (expected 2.00)")
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5.5),constrained_layout=True)
    ax1.loglog(df_time["dt"],df_time["L2"],'-s',color="#C73E1D",lw=2,markersize=7)
    ref_dt=df_time["dt"].values; ax1.loglog(ref_dt,0.5*ref_dt**2,'--',color="#F18F01",lw=1.5,label=r'$\propto \Delta t^2$')
    ax1.set_xlabel(r"$\Delta t$"); ax1.set_ylabel("L2 Error"); ax1.set_title("Temporal Convergence: CN (Fixed n)"); ax1.legend(); ax1.grid(True,which="both",alpha=0.3)
    ax2.semilogy(df_time["dt"],df_time["mass_err"],'-s',color="#2E86AB",lw=2,markersize=7)
    ax2.set_xlabel(r"$\Delta t$"); ax2.set_ylabel("|M - 1|"); ax2.set_title("Mass Conservation Error (Temporal)"); ax2.grid(True,alpha=0.3)
    fig.savefig(os.path.join(cfg.outdir,"fig5b_convergence_time.png"),dpi=dpi,bbox_inches='tight'); plt.close(fig)
    print("  [Fig5b] Temporal convergence order")
    print("\n  Spatial:"); print(df_space[["n","dx","L2","mass_err"]].round(8).to_string(index=False))
    print("\n  Temporal:"); print(df_time[["dt","L2","mass_err"]].round(8).to_string(index=False))


# ──────────────────────────────────────────────────────────────
# 轻量GIF生成函数
# ──────────────────────────────────────────────────────────────

def _save_lightweight_1d_gif_with_exact(outdir,x,psi_hist,saved_t,exact_fn,filename,title="",show_barrier=None,show_v=None,fps=12):
    """Save 1D GIF showing BOTH numerical and exact solution overlaid."""
    from matplotlib.animation import FuncAnimation,PillowWriter
    max_frames=50
    if len(psi_hist)>max_frames:
        step=max(1,len(psi_hist)//max_frames); indices=np.arange(0,len(psi_hist),step)[:max_frames]
        psi_hist=np.array(psi_hist)[indices]; saved_t=np.array(saved_t)[indices]
    all_abs=np.array([np.max(np.abs(p)**2) for p in psi_hist]); global_max=float(np.max(all_abs))
    all_abs_ex=[]
    for ti in saved_t:
        try: ex=exact_fn(ti); all_abs_ex.append(float(np.max(np.abs(ex)**2)))
        except: all_abs_ex.append(global_max)
    global_max=max(global_max,max(all_abs_ex) if all_abs_ex else global_max)
    x_min_plot,x_max_plot=x[0],x[-1]
    for frame in psi_hist:
        mask=np.abs(frame)**2>global_max*1e-3
        if np.any(mask):
            idx_nonzero=np.where(mask)[0]; x_min_plot=min(x_min_plot,x[max(0,idx_nonzero[0])-10]); x_max_plot=max(x_max_plot,x[min(len(x)-1,idx_nonzero[-1]+10)])
    fig,ax0=plt.subplots(1,1,figsize=(9,4.5),constrained_layout=True)
    line_num,=ax0.plot([],[],lw=1.8,color="#2E86AB",label='Numerical')
    line_ex,=ax0.plot([],[],'--',lw=1.5,color="#C73E1D",label='Exact',alpha=0.8)
    ax0.set_xlim(x_min_plot,x_max_plot); ax0.set_ylim(0,global_max*1.15); ax0.set_ylabel(r"$|\psi|^2$",fontsize=11); ax0.grid(True,alpha=0.25)
    if show_v is not None:
        ax0_twin=ax0.twinx(); ax0_twin.plot(x,show_v/(np.max(np.abs(show_v))+1e-30)*global_max*0.3,'gray',':',lw=0.8,alpha=0.5); ax0_twin.set_yticks([])
    if show_barrier: bl,br=show_barrier; ax0.axvline(bl,color='red',ls='--',lw=1.2,alpha=0.7); ax0.axvline(br,color='red',ls='--',lw=1.2,alpha=0.7)
    ax0.legend(fontsize=9,loc='upper right'); title_text=fig.suptitle(f"{title}, t = {saved_t[0]:.2f}",fontsize=12,y=0.98)
    def init(): line_num.set_data([x_min_plot],[0]); line_ex.set_data([x_min_plot],[0]); return line_num,line_ex,[title_text]
    def update(frame):
        psi_f=psi_hist[frame]; line_num.set_data(x,np.abs(psi_f)**2)
        try: ex_f=exact_fn(saved_t[frame]); line_ex.set_data(x,np.abs(ex_f)**2)
        except Exception: pass
        title_text.set_text(f"{title}, t = {saved_t[frame]:.2f}"); return line_num,line_ex,[title_text]
    ani=FuncAnimation(fig,update,frames=len(saved_t),init_func=init,blit=False,interval=max(30,1000//fps),repeat=False)
    ani.save(os.path.join(outdir,filename),writer=PillowWriter(fps=fps),dpi=300); plt.close(fig)

def _save_lightweight_1d_gif_three_line(outdir,x,psi_hist,saved_t,filename,title="",show_barrier=None,fps=12):
    """Save lightweight 1D GIF with THREE lines: |psi|^2 + Re + Im."""
    from matplotlib.animation import FuncAnimation,PillowWriter
    max_frames=50
    if len(psi_hist)>max_frames:
        step=max(1,len(psi_hist)//max_frames); indices=np.arange(0,len(psi_hist),step)[:max_frames]
        psi_hist=np.array(psi_hist)[indices]; saved_t=np.array(saved_t)[indices]
    all_abs=np.array([np.max(np.abs(p)**2) for p in psi_hist])
    all_re=np.array([np.max(np.real(p)) for p in psi_hist]); all_im=np.array([np.min(np.real(p)) for p in psi_hist])
    x_min_plot,x_max_plot=x[0],x[-1]
    global_max_abs=float(np.max(all_abs))
    for frame in psi_hist:
        mask=np.abs(frame)**2>global_max_abs*1e-3
        if np.any(mask):
            idx_nonzero=np.where(mask)[0]; x_min_plot=min(x_min_plot,x[max(0,idx_nonzero[0])-10]); x_max_plot=max(x_max_plot,x[min(len(x)-1,idx_nonzero[-1]+10)])
    fig,(ax0,ax1,ax2)=plt.subplots(3,1,figsize=(7.5,4.5),constrained_layout=True,sharex=True,gridspec_kw={'height_ratios':[1.2,1,1]})
    line_abs,=ax0.plot([],[],lw=1.8,color="#1B998B"); ax0.set_xlim(x_min_plot,x_max_plot); ax0.set_ylim(0,float(np.max(all_abs))*1.15)
    ax0.set_ylabel(r"$|\psi|^2$",fontsize=11); ax0.grid(True,alpha=0.25)
    line_re,=ax1.plot([],[],lw=1.5,color="#2E86AB"); yrange_re=max(abs(float(np.max(all_re))),abs(float(np.min(all_im))))*1.15
    ax1.set_ylim(-yrange_re,yrange_re); ax1.set_ylabel(r"Re[$\psi$]",fontsize=11); ax1.grid(True,alpha=0.25)
    line_im,=ax2.plot([],[],lw=1.5,color="#C73E1D"); yrange_im=max(abs(float(np.max(all_im))),abs(float(np.min(all_im))))*1.15
    ax2.set_ylim(-yrange_im,yrange_im); ax2.set_xlabel("x",fontsize=11); ax2.set_ylabel(r"Im[$\psi$]",fontsize=11); ax2.grid(True,alpha=0.25)
    if show_barrier:
        bl,br=show_barrier
        for ax in [ax0,ax1,ax2]: ax.axvline(bl,color='red',ls='--',lw=1.2,alpha=0.7); ax.axvline(br,color='red',ls='--',lw=1.2,alpha=0.7)
        ax0.axvspan(bl,br,color='gray',alpha=0.12)
    title_text=fig.suptitle(f"{title}, t = {saved_t[0]:.2f}",fontsize=12,y=0.99)
    def init(): line_abs.set_data([x_min_plot],[0]); line_re.set_data([x_min_plot],[0]); line_im.set_data([x_min_plot],[0]); return line_abs,line_re,line_im,[title_text]
    def update(frame):
        psi_f=psi_hist[frame]; line_abs.set_data(x,np.abs(psi_f)**2); line_re.set_data(x,np.real(psi_f)); line_im.set_data(x,np.imag(psi_f))
        title_text.set_text(f"{title}, t = {saved_t[frame]:.2f}"); return line_abs,line_re,line_im,[title_text]
    ani=FuncAnimation(fig,update,frames=len(saved_t),init_func=init,blit=False,interval=max(30,1000//fps),repeat=False)
    ani.save(os.path.join(outdir,filename),writer=PillowWriter(fps=fps),dpi=300); plt.close(fig)

def _save_lightweight_2d_gif(outdir,X,Y,saved_t,hist,filename,title="",fps=10,extent_override=None):
    """Save lightweight 2D GIF."""
    from matplotlib.animation import FuncAnimation,PillowWriter
    max_frames=40
    if len(hist)>max_frames:
        step=max(1,len(hist)//max_frames); hst=np.array(hist)[::step][:max_frames]; saved_t=np.array(saved_t)[::step][:max_frames].tolist()
    fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True)
    if extent_override is not None: extent=list(extent_override)
    else: xv=X[:,0]; yv=Y[0,:]; extent=[xv[0],xv[-1],yv[0],yv[-1]]
    global_vmax=float(np.max(np.abs(hst)**2)); vmin_val=max(global_vmax*0.005,1e-8)
    im=ax.imshow(np.abs(hst[0]).T**2,origin="lower",aspect="equal",extent=extent,cmap="inferno",vmin=vmin_val,vmax=global_vmax)
    ax.set_xlabel("x",fontsize=11); ax.set_ylabel("y",fontsize=11); cb=fig.colorbar(im,ax=ax,shrink=0.85)
    def update(frame): im.set_array(np.abs(hst[frame]).T**2); ax.set_title(f"{title}, t = {saved_t[frame]:.2f}",fontsize=11); return [im]
    ani=FuncAnimation(fig,update,frames=len(saved_t),blit=True,interval=max(40,1000//fps))
    ani.save(os.path.join(outdir,filename),writer=PillowWriter(fps=fps),dpi=300); plt.close(fig)

def _save_lightweight_2d_exact_gif(outdir,X,Y,x,y,t_list,psi0,sigma,kx,ky,dx,dy,filename,title="",fps=10,extent_override=None):
    """Save lightweight 2D GIF using exact analytic solution (computed per frame)."""
    from matplotlib.animation import FuncAnimation,PillowWriter
    max_frames=40
    if len(t_list)>max_frames: step=max(1,len(t_list)//max_frames); t_list=np.array(t_list)[::step][:max_frames].tolist()
    fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True)
    if extent_override is not None: extent=list(extent_override)
    else: xv=X[:,0]; yv=Y[0,:]; extent=[xv[0],xv[-1],yv[0],yv[-1]]
    psi_first=exact_free_gaussian_2d(X,Y,t_list[0],-6.0,0.0,sigma,kx,ky,dx,dy)
    global_vmax=float(np.max(np.abs(psi_first)**2)); vmin_val=max(global_vmax*0.005,1e-8)
    im=ax.imshow(np.abs(psi_first).T**2,origin="lower",aspect="equal",extent=extent,cmap="inferno",vmin=vmin_val,vmax=global_vmax)
    ax.set_xlabel("x",fontsize=11); ax.set_ylabel("y",fontsize=11); cb=fig.colorbar(im,ax=ax,shrink=0.85)
    def update(frame): t_val=t_list[frame]; psi_frame=exact_free_gaussian_2d(X,Y,t_val,-6.0,0.0,sigma,kx,ky,dx,dy); im.set_array(np.abs(psi_frame).T**2); ax.set_title(f"{title}, t = {t_val:.2f}",fontsize=11); return [im]
    ani=FuncAnimation(fig,update,frames=len(t_list),blit=True,interval=max(40,1000//fps))
    ani.save(os.path.join(outdir,filename),writer=PillowWriter(fps=fps),dpi=450); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────

def main(experiment:str="all"):
    """运行实验。experiment='all' 运行全部，'well' 仅运行无限深势阱。"""
    outdir="tdse_experiments_v6"; os.makedirs(outdir,exist_ok=True)
    cfg=RunConfig(outdir=outdir,quick=False,save_gif=True,dpi=600,grid_size=1024)
    print(f"\n{'═'*62}\n  TDSE Numerical Methods — ODE Course\n  Output: {os.path.abspath(outdir)}\n{'═'*62}\n")
    start=time.perf_counter()
    if experiment=="well":
        experiment_1d_infinite_well(cfg)
    else:
        experiment_1d_infinite_well(cfg)       # Exp-01
        experiment_1d_ho_coherent(cfg)         # Exp-02 (新!)
        experiment_2d_ho_coherent(cfg)         # Exp-03 (新!)
        experiment_stability_detailed(cfg)     # Exp-04 [不变]
        experiment_convergence_detailed(cfg)   # Exp-05 [不变]
    elapsed=time.perf_counter()-start
    print(f"\n{'═'*62}\n  All done! Total: {elapsed:.1f}s\n  Output: {outdir}/\n{'═'*62}\n")

if __name__ == "__main__":
    main()
