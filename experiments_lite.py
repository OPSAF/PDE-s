#!/usr/bin/env python3
"""
TDSE 数值方法对比实验 — 微分方程数值解课程专用

实验结构（按重要性排序）：

  实验①  一维无限深势阱（主实验，有解析解）
          → 五方法对比 | 解析解验证 | 完整误差分析(L1/L2/L∞/相对/Re-Im分解) | 三值图 | 动图

  实验②  一维矩形势垒隧穿（扩展实验，无解析解）
          → 五方法对比 | 物理现象展示 | 三值图 | 动图（无误差分析）

  实验③  二维自由传播 V=0（有解析解）
          → ADI为主图 | SSF为基准 | 误差对比分析 | 三值图 | 动图(ADI)

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
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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


def plot_1d_three_panel(x, results, methods, colors, labels, title_prefix,
                        outdir, filename, dpi=600,
                        show_barrier=None, show_v=None,
                        psi_exact=None):
    """标准三面板：|psi|^2, Re[psi], Im[psi]（可选含解析解）"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    axes[0].set_title(r"$|\psi|^2$", fontsize=13)
    axes[1].set_title(r"$\mathrm{Re}[\psi]$", fontsize=13)
    axes[2].set_title(r"$\mathrm{Im}[\psi]$", fontsize=13)

    # 先画解析解（如果有）
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
            # 标记失败的方法
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


# ──────────────────────────────────────────────────────────────
# 无限深势阱解析解
# ──────────────────────────────────────────────────────────────
#
# 【公式来源验证】—— 与标准量子力学教材一致
#
# 参考来源：
#   [1] ComPADRE Quantum Physics, Section 10.7 "Wave Packet Dynamics"
#       https://www.compadre.org/PQP/quantum-theory/supp10_1_4.cfm
#   [2] Robinett, R.W., "Quantum Wave Packet Revivals," Phys. Rep. 392, 1-119 (2004)
#   [3] Griffiths, Introduction to Quantum Mechanics, Ch. 2.2
#
# 标准公式（原子单位 ħ=1, m=1）：
#   势阱范围 [a, b]，宽度 L = b - a
#
#   本征函数：φ_n(x) = √(2/L) · sin(n·π·(x-a)/L)    （n=1,2,3,...）
#   本征值：  E_n = (n·π)² / (2·L²)                 （能级）
#
#   时间演化（本征函数展开法）：
#     Ψ(x,t) = Σ_{n=1}^{∞} c_n · φ_n(x) · exp(-i·E_n·t)
#
#   展开系数：
#     c_n = ∫_a^b φ_n*(x) · Ψ(x,0) dx            （内积投影）
#
# 关键物理时间尺度：
#   经典周期 T_cl = 2L/v_{n0}，其中 v_{n0} ≈ p_{n0}/m = k₀（群速度）
#   复苏周期 T_rev = 4mL²/(π·ħ) = 4L²/π          （波包完全复原）
#
# ⚠️ 注意：我们的初值是高斯波包（不是本征态！），因此：
#   - 需要大量本征态叠加来逼近高斯形状（N=200项）
#   - 高斯在边界处不满足ψ(a)=ψ(b)=0，存在Gibbs现象
#   - 如果初值确实是本征态φ_n，则演化退化为纯相位旋转：Ψ(t)=φ_n·exp(-iE_nt)
#

def _exact_infinite_well_gaussian(x, t, x0, sigma, k0, a, b, n_eigen=200):
    """
    无限深势阱中高斯波包的精确解（本征函数展开法）。

    势阱范围 [a, b]，宽度 L = b - a。
    本征函数: φ_n(x) = sqrt(2/L) * sin(n*pi*(x-a)/L)
    本征值: E_n = (n*pi)^2 / (2*L^2)
    展开系数: c_n = <φ_n|ψ_0>
    精确解:   ψ(x,t) = Σ c_n * φ_n(x) * exp(-i*E_n*t)
    """
    L = b - a
    psi = np.zeros_like(x, dtype=complex)
    psi0 = gaussian_wavepacket(x, x0, sigma, k0, (x[1]-x[0]))
    # 归一化初始波包
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
#
# 【初值设置】—— 本征态（非高斯波包）
#
# 本征态定义：
#   φ_n(x) = √(2/L) · sin(n·π·(x-a)/L),   x ∈ [a, b]
#   E_n = (n·π)² / (2·L²)
#
# 时间演化（纯相位旋转）：
#   Ψ(x,t) = φ_n(x) · exp(-i·E_n·t)
#
# 关键性质：
#   |Ψ(x,t)|² = |φ_n(x)|²  —— 概率密度不随时间变化！
#   这是比高斯波包更严格的酉性检验：数值方法必须精确保持 |ψ|² 不变。
#

def experiment_1d_infinite_well(cfg):
    """一维无限深势阱 — 五种方法 + 解析解完整对比（初值=本征态）"""
    print("=" * 60)
    print("Exp-01: 1D Infinite Well (MAIN — eigenstate initial condition)")
    print("=" * 60)

    # ── 求解区间 = 势阱区间（正确边界条件：ψ(-10)=ψ(10)=0）──
    n = 2048
    well_left, well_right = -10.0, 10.0
    L = well_right - well_left
    x, dx = grid(well_left, well_right, n)  # 网格精确覆盖势阱区间 [-10, 10]
    dt = 0.001            # 时间步长（0.0005导致RK4也发散，回调）
    t_end = 8.0
    t = np.arange(0.0, t_end + 0.5 * dt, dt)

    # ── 初值：本征态 φ_n（Dirichlet BC: φ_n(±10)=0）──
    eigen_n = 3
    E_n = (eigen_n * np.pi)**2 / (2.0 * L**2)

    # 网格已精确覆盖势阱，直接定义本征函数
    phi_n = np.sqrt(2.0 / L) * np.sin(eigen_n * np.pi * (x - well_left) / L)
    phi_n = phi_n.astype(complex)
    phi_n = normalize(phi_n, dx)
    psi0 = phi_n.copy()

    print(f"  Eigenstate n={eigen_n}, E_n={E_n:.4f}, n={n}, dt={dt}")
    print(f"  Domain: [{well_left}, {well_right}], Dirichlet BC enforced")
    print(f"  |psi_0|^2 should be time-independent")

    # 网格完全在势阱内，V=0（无限深势阱内部无势能）
    v = np.zeros_like(x, dtype=float)

    methods_all = ["FTCS", "Backward-Euler", "Crank-Nicolson", "RK4", "Split-Step-FFT"]
    colors_all = ["#9B9B9B", "#2E86AB", "#1B998B", "#F18F01", "#C73E1D"]
    labels_all = ["FTCS", "BE", "CN", "RK4", "SSF"]

    results = run_1d_methods(methods_all, psi0, v, x, t, dx, dt)

    # ── 解析解：φ_n(x) · exp(-i·E_n·t)（纯相位演化）──
    psi_exact = phi_n * np.exp(-1j * E_n * t_end)

    dpi = get_adaptive_dpi(3)

    # ── 图1a: 全部5方法 + 解析解 三值对比（含FTCS发散标记）──
    plot_1d_three_panel(
        x, results, methods_all, colors_all, labels_all,
        "Infinite Well (all)", cfg.outdir, "fig1a_inf_well_all.png",
        dpi=dpi, show_barrier=(well_left, well_right), psi_exact=psi_exact)
    print("  [Fig1a] All 5 methods + Exact: three-value (FTCS marked if diverged)")

    # ── 图1b: 稳定方法 + 解析解三值对比（无FTCS）──
    m_stable = ["Backward-Euler", "Crank-Nicolson", "RK4", "Split-Step-FFT"]
    c_stable = ["#2E86AB", "#1B998B", "#F18F01", "#C73E1D"]
    l_stable = ["BE", "CN", "RK4", "SSF"]
    plot_1d_three_panel(
        x, results, m_stable, c_stable, l_stable,
        "Infinite Well (stable)", cfg.outdir, "fig1b_inf_well_stable.png",
        dpi=dpi, show_barrier=(well_left, well_right), psi_exact=psi_exact)
    print("  [Fig1b] Stable methods + Exact: three-value")

    # ── 图1c: 各方法与解析解单独对比（|psi|^2 + Re/Im分离）──
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
            ax.set_title(f"{method}: L1={l1:.2e}  L2={l2:.2e}  L∞={linf:.2e}", fontsize=10)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlabel("x")
    fig.savefig(os.path.join(cfg.outdir, "fig1c_inf_well_vs_exact.png"),
                dpi=get_adaptive_dpi(4), bbox_inches='tight')
    plt.close(fig)
    print("  [Fig1c] Stable methods vs Exact (separate)")

    # ── 图1d: 完整误差分析表（修复文字排版错位问题）──
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
                "Mass": r["mass"],
                "L1": l1, "L2": l2, "Linf": linf,
                "Rel_L2": rel_l2,
                "L2_Re": l2_re, "L2_Im": l2_im,
            })
        else:
            row.update({"Mass": np.nan, "L1": np.nan, "L2": np.nan,
                        "Linf": np.nan, "Rel_L2": np.nan,
                        "L2_Re": np.nan, "L2_Im": np.nan})
        err_rows.append(row)

    df_err = pd.DataFrame(err_rows)
    df_err.to_csv(os.path.join(cfg.outdir, "data_inf_well_errors.csv"), index=False)

    # 绘制误差柱状图组 — 修复文字溢出和排版错位
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    metrics = [
        ("L1 Error", "L1", True),
        ("L2 Error", "L2", True),
        ("Linf Error", "Linf", True),
        ("Relative L2", "Rel_L2", True),
        ("L2 Real Part", "L2_Re", True),
        ("L2 Imag Part", "L2_Im", True),
    ]
    # 只取有效方法（排除NaN）
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
                ax.set_yscale('log')
                # log坐标下设置合理的ylim避免collapsed
                v_min = min(vals_finite) * 0.5
                v_max = max(vals_finite) * 5.0
                ax.set_ylim(v_min, v_max)
        ax.set_xticks(range(len(valid_methods)))
        ax.set_xticklabels(valid_methods, fontsize=9, rotation=10)
        ax.set_title(title, fontsize=11); ax.grid(True, alpha=0.3)
        # 用enumerate而非bar.index，安全可靠
        for bi, (bar, val) in enumerate(zip(bars, vals)):
            if np.isfinite(val) and val > 0:
                bh = bar.get_height()
                bx = bar.get_x() + bar.get_width() / 2.
                # 统一放在柱顶上方，用bbox防止飞出
                ax.text(bx, bh * 1.12, f'{val:.2e}', ha='center', va='bottom',
                       fontsize=7, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.15',
                                facecolor=clrs[bi], alpha=0.8,
                                edgecolor='none'),
                       zorder=10)

    fig.suptitle("Infinite Well - Complete Error Analysis vs Analytic Solution",
                 fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(cfg.outdir, "fig1d_inf_well_error_analysis.png"),
                dpi=get_adaptive_dpi(6), bbox_inches='tight')
    plt.close(fig)
    print("  [Fig1d] Complete error analysis (6 metrics)")
    print(df_err.round(8).to_string(index=False))

    # ~~ 删除图1e（单独解析解图），已合并到图1a和图1b中 ~~

    # ── 动图（CN，三线）──
    if results["Crank-Nicolson"]["stable"]:
        print("  Generating animation (CN, eigenstate evolution)...")
        t_long = np.arange(0.0, 28.0 + 0.5 * dt, dt)
        _, hist_long = solve("Crank-Nicolson", psi0, v, x, t_long, dx, dt,
                             store_every=int(len(t_long)/80))
        t_snap = t_long[::int(len(t_long)//80)][:len(hist_long)]
        _save_lightweight_1d_gif_three_line(
            cfg.outdir, x, hist_long, t_snap,
            "gif_inf_well.gif",
            title=f"Infinite Well (Eigenstate n={eigen_n})",
            show_barrier=(well_left, well_right),
            fps=24
        )
        print("  [GIF1] Eigenstate animation (|psi|^2 should be constant!)")

    print(f"  Grid: n={n}, dt={dt}, steps={len(t)-1}, Well width={well_right-well_left}")


# ──────────────────────────────────────────────────────────────
# 实验②：一维矩形势垒隧穿（扩展实验，无解析解）
# ──────────────────────────────────────────────────────────────

def experiment_1d_tunneling(cfg):
    """一维量子隧穿 — 方法对比 + 物理现象展示（无解析解对比）"""
    print("=" * 60)
    print("Exp-02: 1D Tunneling — Rectangular Barrier (EXTENDED)")
    print("=" * 60)

    n = 1024
    x, dx = grid(-40.0, 40.0, n)
    dt = 0.001
    t_end = 7.0
    t = np.arange(0.0, t_end + 0.5 * dt, dt)

    x0, sigma, k0 = -8.0, 1.2, 3.5
    psi0 = gaussian_wavepacket(x, x0, sigma, k0, dx)
    a_b, b_b = -1.5, 1.5
    V0 = 2.0
    v = potential_rect_barrier(x, v0=V0, a=a_b, b=b_b)

    methods_all = ["FTCS", "Backward-Euler", "Crank-Nicolson", "RK4", "Split-Step-FFT"]
    colors_all = ["#9B9B9B", "#2E86AB", "#1B998B", "#F18F01", "#C73E1D"]
    labels_all = ["FTCS", "BE", "CN", "RK4", "SSF"]

    results = run_1d_methods(methods_all, psi0, v, x, t, dx, dt)

    dpi = get_adaptive_dpi(3)

    # ── 图2a: 全部5方法 ──
    plot_1d_three_panel(
        x, results, methods_all, colors_all, labels_all,
        "Tunneling (all)", cfg.outdir, "fig2a_tunneling_all.png",
        dpi=dpi, show_barrier=(a_b, b_b), show_v=v)
    print("  [Fig2a] All 5 methods: three-value")

    # ── 图2b: 稳定方法（无FTCS）──
    m_stable = ["Backward-Euler", "Crank-Nicolson", "RK4", "Split-Step-FFT"]
    c_stable = ["#2E86AB", "#1B998B", "#F18F01", "#C73E1D"]
    l_stable = ["BE", "CN", "RK4", "SSF"]
    plot_1d_three_panel(
        x, results, m_stable, c_stable, l_stable,
        "Tunneling (no FTCS)", cfg.outdir, "fig2b_tunneling_no_ftcs.png",
        dpi=dpi, show_barrier=(a_b, b_b), show_v=v)
    print("  [Fig2b] Stable methods only: three-value")

    # ── 图2c: CN结果详解（透射+反射）──
    if results["Crank-Nicolson"]["stable"]:
        psi_cn = results["Crank-Nicolson"]["psi"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

        ax = axes[0]
        ax.fill_between([a_b, b_b], [0], [np.max(np.abs(psi_cn)**2)*1.3],
                        color='gray', alpha=0.25, label=f'Barrier V₀={V0}')
        ax.plot(x, np.abs(psi_cn)**2, '#1B998B', lw=1.8, label=r'$|\psi|^2$')
        ax.plot(x, np.abs(psi0)**2, '#F18F01', lw=1.0, ls='--', alpha=0.5, label=r'$|\psi_0|^2$')
        ax.axvline(a_b, color='k', linestyle='--', lw=1.0, alpha=0.5); ax.axvline(b_b, color='k', linestyle='--', lw=1.0, alpha=0.5)
        ax.set_xlabel("x"); ax.set_ylabel(r"$|\psi|^2$")
        ax.set_title("Tunneling Result (CN): Transmission + Reflection")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(x, np.real(psi_cn), '#2E86AB', lw=1.5, label=r'Re[$\psi$]')
        ax.plot(x, np.imag(psi_cn), '#C73E1D', lw=1.5, label=r'Im[$\psi$]')
        ax.fill_between([a_b, b_b], [np.min(np.real(psi_cn))], [np.max(np.real(psi_cn))],
                        color='gray', alpha=0.12)
        ax.axvline(a_b, color='k', linestyle='--', lw=1.0, alpha=0.5); ax.axvline(b_b, color='k', linestyle='--', lw=1.0, alpha=0.5)
        ax.set_xlabel("x"); ax.set_ylabel("Amplitude")
        ax.set_title("Real & Imaginary Parts (CN)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        fig.savefig(os.path.join(cfg.outdir, "fig2c_tunneling_cn_detail.png"),
                    dpi=get_adaptive_dpi(2), bbox_inches='tight')
        plt.close(fig)
        print("  [Fig2c] CN tunneling detail (transmission/reflection)")

    # ── 图2d: 质量守恒对比（全部方法）──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    masses = []
    for m in methods_all:
        val = results[m]["mass"]
        masses.append(val if (np.isfinite(val) and not np.isnan(val)) else np.nan)

    axes[0].bar(range(len(methods_all)), masses, color=colors_all)
    axes[0].axhline(1.0, color='r', linestyle='--', lw=1.2, alpha=0.7)
    axes[0].set_xticks(range(len(methods_all))); axes[0].set_xticklabels(labels_all, rotation=15)
    axes[0].set_ylabel("Probability Mass"); axes[0].set_title("Mass Conservation (All)")
    axes[0].grid(True, alpha=0.3)

    amps = []
    for m in methods_all:
        r = results[m]
        if r["stable"] and r["psi"] is not None:
            val = np.max(np.abs(r["psi"]))
            amps.append(min(val, 100.0) if np.isfinite(val) else 100.0)
        else:
            amps.append(100.0)
    axes[1].bar(range(len(methods_all)), amps, color=colors_all)
    axes[1].axhline(10.0, color='r', linestyle='--', lw=1.2, alpha=0.7, label='Divergence threshold')
    axes[1].set_xticks(range(len(methods_all))); axes[1].set_xticklabels(labels_all, rotation=15)
    axes[1].set_ylabel(r"$\max|\psi|$ (capped)"); axes[1].set_title("Peak Amplitude")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.savefig(os.path.join(cfg.outdir, "fig2d_tunneling_mass_amp.png"),
                dpi=get_adaptive_dpi(2), bbox_inches='tight')
    plt.close(fig)
    print("  [Fig2d] Mass conservation + amplitude check (All)")

    # ── 图2e: 质量守恒（无FTCS，稳定方法细节）──
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    stable_masses = []
    stable_labels = []
    stable_colors = []
    for i, m in enumerate(m_stable):
        val = results[m]["mass"]
        if np.isfinite(val) and not np.isnan(val):
            stable_masses.append(val)
        else:
            stable_masses.append(np.nan)
        stable_labels.append(l_stable[i])
        stable_colors.append(c_stable[i])

    bars = ax.bar(range(len(m_stable)), stable_masses, color=stable_colors,
                  edgecolor='white', lw=0.5)
    ax.axhline(1.0, color='r', linestyle='--', lw=1.5, alpha=0.7, label='Ideal M=1')
    ax.set_xticks(range(len(m_stable))); ax.set_xticklabels(stable_labels, fontsize=11)
    ax.set_ylabel("Probability Mass", fontsize=12); ax.set_title("Mass Conservation (Stable Methods Only)")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    # 在柱子上方标注数值
    for bar, m_val in zip(bars, stable_masses):
        if np.isfinite(m_val):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()*1.02,
                   f'{m_val:.6f}', ha='center', va='bottom', fontsize=9)
    fig.savefig(os.path.join(cfg.outdir, "fig2e_tunneling_mass_no_ftcs.png"),
                dpi=get_adaptive_dpi(1), bbox_inches='tight')
    plt.close(fig)
    print("  [Fig2e] Mass conservation (stable only, no FTCS)")

    # ── 图2f: 质量+振幅（无FTCS，稳定方法完整对比）──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    stable_masses = []
    stable_amps = []
    for i, m in enumerate(m_stable):
        r = results[m]
        m_val = r["mass"]
        if np.isfinite(m_val) and not np.isnan(m_val):
            stable_masses.append(m_val)
        else:
            stable_masses.append(np.nan)
        if r["stable"] and r["psi"] is not None:
            amp = min(np.max(np.abs(r["psi"])), 100.0) if np.isfinite(np.max(np.abs(r["psi"]))) else 100.0
        else:
            amp = 100.0
        stable_amps.append(amp)

    axes[0].bar(range(len(m_stable)), stable_masses, color=c_stable,
                edgecolor='white', lw=0.5)
    axes[0].axhline(1.0, color='r', linestyle='--', lw=1.5, alpha=0.7, label='Ideal M=1')
    axes[0].set_xticks(range(len(m_stable))); axes[0].set_xticklabels(l_stable, fontsize=10)
    axes[0].set_ylabel("Probability Mass"); axes[0].set_title("Mass (Stable Only)")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    for j, mv in enumerate(stable_masses):
        if np.isfinite(mv):
            axes[0].text(j, mv * 1.005, f'{mv:.5f}', ha='center', va='bottom', fontsize=7)

    axes[1].bar(range(len(m_stable)), stable_amps, color=c_stable,
                edgecolor='white', lw=0.5)
    axes[1].axhline(10.0, color='r', linestyle='--', lw=1.2, alpha=0.7, label='Divergence threshold')
    axes[1].set_xticks(range(len(m_stable))); axes[1].set_xticklabels(l_stable, fontsize=10)
    axes[1].set_ylabel(r"$\max|\psi|$ (capped)"); axes[1].set_title("Amplitude (Stable Only)")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    for j, av in enumerate(stable_amps):
        if np.isfinite(av):
            axes[1].text(j, av * 1.05, f'{av:.3f}', ha='center', va='bottom', fontsize=7)

    fig.savefig(os.path.join(cfg.outdir, "fig2f_tunneling_mass_amp_no_ftcs.png"),
                dpi=get_adaptive_dpi(2), bbox_inches='tight')
    plt.close(fig)
    print("  [Fig2f] Mass + Amplitude (stable only, no FTCS)")

    # ── 隧穿动图（CN，三线）──
    if results["Crank-Nicolson"]["stable"]:
        print("  Generating tunneling animation...")
        t_hist, hist = solve("Crank-Nicolson", psi0, v, x, t, dx, dt, store_every=50)
        _save_lightweight_1d_gif_three_line(
            cfg.outdir, x, hist, t_hist,
            "gif_tunneling.gif",
            title="Quantum Tunneling",
            show_barrier=(a_b, b_b),
            fps=20
        )
        print("  [GIF2] Tunneling animation (three-line, 20fps)")

    print(f"  Grid: n={n}, dt={dt}, steps={len(t)-1}, Barrier=[{a_b},{b_b}], V0={V0}")


# ──────────────────────────────────────────────────────────────
# 实验③：二维自由传播 V=0（有解析解，ADI为主）
# ──────────────────────────────────────────────────────────────
#
# 【2D自由高斯解析解 — 公式验证】
#
# 参考来源：
#   [1] Libretexts, "Propagation of a Gaussian Wavepacket"
#       https://eng.libretexts.org/.../Appendix_1_-_Electron_Wavepacket_Propagation
#   [2] ar5iv:2305.00059, "Free expansion of a Gaussian wavepacket"
#   [3] 标准QM教材, 自由粒子Gaussian波包章节
#
# 标准公式（原子单位 ħ=1, m=1）：
#   一维：ψ(x,t) = (σ₀/√(σ₀²+it)) · exp(-(x-x₀-k₀t)²/(2(σ₀²+it))) · exp(i(k₀x - k₀²t/2))
#
#   二维（V=0可分离）：
#     ψ(x,y,t) = ψ_1D(x,t; x₀,σ,kx₀) × ψ_1D(y,t; y₀,σ,ky₀)
#
#   即：prefactor = (σ/√(σ²+it))², envelope = exp(-Σ(x_i-x_{0i}-k_i t)²/(2(σ²+it)))
#
# 物理特征：
#   - 波包中心以群速度 v_g = (kx₀, ky₀) 匀速运动
#   - 波包宽度随时间扩散：σ(t) = σ₀·√(1 + t²/σ₀⁴)
#   - 概率守恒：∫|ψ|²dxdy = 1 （归一化保持）
#
# ⚠️ potentials.py 中 exact_free_gaussian_2d() 的实现已验证与此一致 ✅
#

def experiment_2d_free(cfg):
    """二维自由传播 — ADI为主图 + SSF基准 + 解析解误差分析"""
    print("=" * 60)
    print("Exp-03: 2D Free Propagation V=0 (has analytic solution)")
    print("=" * 60)

    nx, ny = 192, 192
    X, Y, x, y, dx, dy, KX, KY = make_2d_grid(nx, ny, -16.0, 16.0, -10.0, 10.0)
    dt = 0.005           # 更小的时间步长，提高精度
    t_end = 4.0         # 更紧凑的时长
    t = np.arange(0.0, t_end + 0.5 * dt, dt)

    # ── 初值：二维高斯波包（与解析解参数一致）──
    x0, y0 = -6.0, 0.0   # 初始位置更靠近中心（减少空白）
    sigma = 1.2          # 稍窄一点（更紧凑）
    kx, ky = 3.0, 0.0   # 稍慢一点（在更小的区域内运动）

    psi0 = gaussian_wavepacket_2d(X, Y, x0, y0, sigma, kx, ky, dx, dy)
    V = potential_free(X)  # V = 0 everywhere

    # ── 解析解 ──
    print("  Computing 2D analytic solution...")
    psi_exact = exact_free_gaussian_2d(X, Y, t_end, x0, y0, sigma, kx, ky, dx, dy)

    # 验证初值一致性
    psi_exact_t0 = exact_free_gaussian_2d(X, Y, 0.0, x0, y0, sigma, kx, ky, dx, dy)
    init_diff = np.max(np.abs(psi0 - psi_exact_t0))
    print(f"  Initial condition check: max|psi_0 - psi_exact(t=0)| = {init_diff:.2e} (should be ~0)")
    if init_diff > 1e-10:
        print("  WARNING: Initial condition does NOT match analytic solution!")

    # 运行两种方法
    print("  Running ADI...")
    _, hist_adi = solve_2d(psi0, V, KX, KY, t, dt, dx, dy,
                           store_every=len(t)-1, method="adi")
    psi_adi = hist_adi[-1]

    print("  Running SSF (reference)...")
    _, hist_ssf = solve_2d(psi0, V, KX, KY, t, dt, dx, dy,
                           store_every=len(t)-1, method="split-step-fft")
    psi_ssf = hist_ssf[-1]

    dpi = get_adaptive_dpi(3)
    extent_compact = (-8.0, 10.0, -6.0, 6.0)  # 聚焦视图范围

    # ── 图3a: ADI 三值主图（紧凑坐标）──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    im0 = axes[0].pcolormesh(x, y, np.abs(psi_adi)**2, cmap='viridis', shading='gouraud')
    axes[0].set_title(r"ADI: $|\psi|^2$", fontsize=13); plt.colorbar(im0, ax=axes[0], shrink=0.85)
    axes[0].set_xlim(extent_compact[0], extent_compact[1])
    axes[0].set_ylim(extent_compact[2], extent_compact[3])
    axes[0].set_aspect('equal')

    im1 = axes[1].pcolormesh(x, y, np.real(psi_adi), cmap='RdBu_r', shading='gouraud',
                             vmin=-np.max(np.abs(np.real(psi_adi))),
                             vmax=np.max(np.abs(np.real(psi_adi))))
    axes[1].set_title(r"ADI: $\mathrm{Re}[\psi]$", fontsize=13); plt.colorbar(im1, ax=axes[1], shrink=0.85)
    axes[1].set_xlim(extent_compact[0], extent_compact[1]); axes[1].set_ylim(extent_compact[2], extent_compact[3])
    axes[1].set_aspect('equal')

    im2 = axes[2].pcolormesh(x, y, np.imag(psi_adi), cmap='RdBu_r', shading='gouraud',
                             vmin=-np.max(np.abs(np.imag(psi_adi))),
                             vmax=np.max(np.abs(np.imag(psi_adi))))
    axes[2].set_title(r"ADI: $\mathrm{Im}[\psi]$", fontsize=13); plt.colorbar(im2, ax=axes[2], shrink=0.85)
    axes[2].set_xlim(extent_compact[0], extent_compact[1]); axes[2].set_ylim(extent_compact[2], extent_compact[3])
    axes[2].set_aspect('equal')
    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.savefig(os.path.join(cfg.outdir, "fig3a_2d_adi_result.png"), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  [Fig3a] ADI main result: three-value (compact)")

    # ── 图3b: 解析解三值图（紧凑坐标）──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    im0 = axes[0].pcolormesh(x, y, np.abs(psi_exact)**2, cmap='viridis', shading='gouraud')
    axes[0].set_title(r"Exact: $|\psi|^2$", fontsize=13); plt.colorbar(im0, ax=axes[0], shrink=0.85)
    axes[0].set_xlim(extent_compact[0], extent_compact[1]); axes[0].set_ylim(extent_compact[2], extent_compact[3])
    axes[0].set_aspect('equal')
    im1 = axes[1].pcolormesh(x, y, np.real(psi_exact), cmap='RdBu_r', shading='gouraud',
                             vmin=-np.max(np.abs(np.real(psi_exact))),
                             vmax=np.max(np.abs(np.real(psi_exact))))
    axes[1].set_title(r"Exact: $\mathrm{Re}[\psi]$", fontsize=13); plt.colorbar(im1, ax=axes[1], shrink=0.85)
    axes[1].set_xlim(extent_compact[0], extent_compact[1]); axes[1].set_ylim(extent_compact[2], extent_compact[3])
    axes[1].set_aspect('equal')
    im2 = axes[2].pcolormesh(x, y, np.imag(psi_exact), cmap='RdBu_r', shading='gouraud',
                             vmin=-np.max(np.abs(np.imag(psi_exact))),
                             vmax=np.max(np.abs(np.imag(psi_exact))))
    axes[2].set_title(r"Exact: $\mathrm{Im}[\psi]$", fontsize=13); plt.colorbar(im2, ax=axes[2], shrink=0.85)
    axes[2].set_xlim(extent_compact[0], extent_compact[1]); axes[2].set_ylim(extent_compact[2], extent_compact[3])
    axes[2].set_aspect('equal')
    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.savefig(os.path.join(cfg.outdir, "fig3b_2d_exact.png"), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  [Fig3b] Exact solution: three-value (compact)")

    # ── 图3c: ADI vs Exact 误差分析（核心！）──
    diff_adi = psi_adi - psi_exact
    diff_ssf = psi_ssf - psi_exact

    l2_adi = float(np.sqrt(np.sum(np.abs(diff_adi)**2) * dx * dy))
    l2_ssf = float(np.sqrt(np.sum(np.abs(diff_ssf)**2) * dx * dy))
    ref_norm = float(np.sqrt(np.sum(np.abs(psi_exact)**2) * dx * dy))
    rel_adi = l2_adi / ref_norm; rel_ssf = l2_ssf / ref_norm
    linf_adi = float(np.max(np.abs(diff_adi))); linf_ssf = float(np.max(np.abs(diff_ssf)))

    mass_adi = probability_mass(psi_adi.flatten(), dx)
    mass_ssf = probability_mass(psi_ssf.flatten(), dx)
    mass_ex = probability_mass(psi_exact.flatten(), dx)

    dpi = get_adaptive_dpi(4)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)

    im00 = axes[0,0].pcolormesh(x, y, np.abs(diff_adi), cmap='hot', shading='gouraud')
    axes[0,0].set_title(r"$|\psi_{ADI} - \psi_{exact}|$", fontsize=12)
    plt.colorbar(im00, ax=axes[0,0], shrink=0.85); axes[0,0].set_aspect('equal')

    im01 = axes[0,1].pcolormesh(x, y, np.abs(diff_ssf), cmap='hot', shading='gouraud')
    axes[0,1].set_title(r"$|\psi_{SSF} - \psi_{exact}|$", fontsize=12)
    plt.colorbar(im01, ax=axes[0,1], shrink=0.85); axes[0,1].set_aspect('equal')

    # 定量指标
    metrics_names = ['L2 Error', 'Rel. L2', 'Linf Error', 'Mass']
    adi_vals = [l2_adi, rel_adi, linf_adi, mass_adi]
    ssf_vals = [l2_ssf, rel_ssf, linf_ssf, mass_ssf]
    exact_vals = [0, 0, 0, mass_ex]

    x_pos = np.arange(len(metrics_names))
    w = 0.25
    axes[1,0].bar(x_pos - w, adi_vals, w, label='ADI', color='#2E86AB')
    axes[1,0].bar(x_pos, ssf_vals, w, label='SSF', color='#C73E1D')
    axes[1,0].bar(x_pos + w, exact_vals, w, label='Exact', color='#1B998B', alpha=0.5)
    axes[1,0].set_xticks(x_pos); axes[1,0].set_xticklabels(metrics_names, fontsize=9)
    axes[1,0].set_title("Quantitative Comparison: ADI vs SSF vs Exact")
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
    for i, (av, sv) in enumerate(zip(adi_vals, ssf_vals)):
        axes[1,0].text(x_pos[i]-w, av*1.05, f'{av:.2e}', ha='center', fontsize=6.5)
        axes[1,0].text(x_pos[i], sv*1.05, f'{sv:.2e}', ha='center', fontsize=6.5)

    # y=0 截线
    mid_j = ny // 2
    axes[1,1].plot(x, np.abs(psi_exact[mid_j,:])**2, 'k-', lw=1.8, label='Exact')
    axes[1,1].plot(x, np.abs(psi_adi[mid_j,:])**2, '#2E86AB', lw=1.5, ls='-', label='ADI')
    axes[1,1].plot(x, np.abs(psi_ssf[mid_j,:])**2, '#C73E1D', lw=1.5, ls='--', label='SSF')
    axes[1,1].set_xlabel("x"); axes[1,1].set_ylabel(r"$|\psi(y=0)|^2$")
    axes[1,1].set_title("Cross-section at y = 0")
    axes[1,1].legend(fontsize=9); axes[1,1].grid(True, alpha=0.3)

    fig.savefig(os.path.join(cfg.outdir, "fig3c_2d_error_analysis.png"),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Fig3c] ADI/SSF vs Exact error analysis")
    print(f"       ADI: L2={l2_adi:.2e}, Rel={rel_adi:.4f}, Linf={linf_adi:.2e}, M={mass_adi:.6f}")
    print(f"       SSF: L2={l2_ssf:.2e}, Rel={rel_ssf:.4f}, Linf={linf_ssf:.2e}, M={mass_ssf:.6f}")

    # ── 二维自由传播动图（两个：解析解 + ADI，DPI=450）──
    print("  Generating 2D animations (Exact + ADI)...")

    # 动图1: ADI数值解
    t_anim, hist_anim = solve_2d(psi0, V, KX, KY, t, dt, dx, dy,
                                  store_every=30, method="adi")
    _save_lightweight_2d_gif(
        cfg.outdir, X, Y, t_anim, hist_anim,
        "gif_2d_free_adi.gif",
        title="2D Free Propagation (ADI)",
        fps=15,
        extent_override=extent_compact
    )
    print("  [GIF3a] 2D free propagation animation (ADI, 15fps)")

    # 动图2: 解析解（逐帧计算）
    _save_lightweight_2d_exact_gif(
        cfg.outdir, X, Y, x, y, t_anim,
        psi0, sigma, kx, ky, dx, dy,
        "gif_2d_free_exact.gif",
        title="2D Free Propagation (Exact)",
        fps=15,
        extent_override=extent_compact
    )
    print("  [GIF3b] 2D free propagation animation (Exact, 15fps)")

    print(f"  Grid: {nx}x{ny}, dt={dt}, steps={len(t)-1}")


# ──────────────────────────────────────────────────────────────
# 实验④：Von Neumann 稳定性扫描（详细版）
# ──────────────────────────────────────────────────────────────

def experiment_stability_detailed(cfg):
    """Von Neumann 稳定性分析 — 详细扫描"""
    print("=" * 60)
    print("Exp-04: Von Neumann Stability Analysis (Detailed)")
    print("=" * 60)

    ns = [128, 256, 512, 1024]
    dts = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    t_end = 1.0
    methods = ["FTCS", "Backward-Euler", "Crank-Nicolson", "RK4", "Split-Step-FFT"]
    rows = []

    for n in tqdm(ns, desc="Grid sizes"):
        x, dx = grid(-20.0, 20.0, n)
        psi0 = gaussian_wavepacket(x, -5.0, 1.0, 2.0, dx)
        v = potential_free(x)
        for dt_val in dts:
            t_arr = np.arange(0.0, t_end + 0.5 * dt_val, dt_val)
            for method in methods:
                try:
                    _, hist = solve(method, psi0, v, x, t_arr, dx, dt_val, store_every=len(t_arr)-1)
                    mass = probability_mass(hist[-1], dx)
                    peak = float(np.max(np.abs(hist[-1])))
                    stable = abs(mass - 1.0) < 0.5 and (peak < 100 if np.isfinite(peak) else False)
                except Exception:
                    stable = False; mass = np.nan; peak = np.nan
                rows.append({
                    "method": method, "n": n, "dx": dx, "dt": dt_val,
                    "stable": stable, "mass_final": mass, "peak_amp": peak,
                    "mu": dt_val / dx**2 if dx > 0 else np.nan,   # CFL数
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.outdir, "data_stability.csv"), index=False)

    # ── 图4a: 稳定性区域散点图 ──
    dpi = get_adaptive_dpi(1)
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    method_order = methods
    colors = ["#9B9B9B", "#2E86AB", "#1B998B", "#F18F01", "#C73E1D"]
    for i, m in enumerate(method_order):
        sub = df[df["method"]==m]
        stable_mask = sub["stable"] == True
        ax.scatter(sub.loc[stable_mask, "dt"], [i]*int(stable_mask.sum()),
                   color=colors[i], marker='o', s=55, zorder=3, label=m if i==0 else None)
        unstable_mask = ~stable_mask
        if unstable_mask.sum() > 0:
            ax.scatter(sub.loc[unstable_mask, "dt"], [i]*int(unstable_mask.sum()),
                       color=colors[i], marker='x', s=55, linewidths=2, zorder=3)
    ax.set_yticks(range(len(method_order))); ax.set_yticklabels(method_order, fontsize=10)
    ax.set_xlabel(r"$\Delta t$", fontsize=12)
    ax.set_title("Von Neumann Stability Map (o=stable, x=unstable)", fontsize=13)
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker='o', color='gray', label='Stable', markersize=8, ls='None'),
        Line2D([0],[0], marker='x', color='gray', label='Unstable', markersize=8, mew=2, ls='None'),
    ], loc='lower right', fontsize=10)
    fig.savefig(os.path.join(cfg.outdir, "fig4a_stability_map.png"), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  [Fig4a] Stability map (stability vs dt)")

    # ── 图4b: CFL数(μ=dt/dx^2)稳定性边界 ──
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    for i, m in enumerate(method_order):
        sub = df[df["method"]==m]
        stable_mask = sub["stable"] == True
        mu_stable = sub.loc[stable_mask, "mu"].values if stable_mask.sum() > 0 else []
        mu_unstable = sub.loc[~stable_mask, "mu"].values if (~stable_mask).sum() > 0 else []
        if len(mu_stable) > 0:
            ax.scatter(mu_stable, [i]*len(mu_stable), color=colors[i], marker='o', s=50, zorder=3)
        if len(mu_unstable) > 0:
            ax.scatter(mu_unstable, [i]*len(mu_unstable), color=colors[i], marker='x', s=50,
                      linewidths=2, zorder=3)
    # FTCS理论临界线 μ_c = 0.5
    ax.axvline(0.5, color='#9B9B9B', ls=':', lw=2, alpha=0.7, label=r'Theoretical FTCS limit: $\mu_c=0.5$')
    ax.set_yticks(range(len(method_order))); ax.set_yticklabels(method_order, fontsize=10)
    ax.set_xlabel(r"CFL number $\mu = \Delta t / \Delta x^2$", fontsize=12)
    ax.set_title("Stability Boundary in CFL Space", fontsize=13)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
    fig.savefig(os.path.join(cfg.outdir, "fig4b_stability_cfl.png"), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  [Fig4b] CFL stability boundary")


# ──────────────────────────────────────────────────────────────
# 实验⑤：Crank-Nicolson 收敛性验证（详细版）
# ──────────────────────────────────────────────────────────────

def experiment_convergence_detailed(cfg):
    """CN格式收敛性 — 多网格 + 多时间步长"""
    print("=" * 60)
    print("Exp-05: Crank-Nicolson Convergence Verification (Detailed)")
    print("=" * 60)

    # --- 空间收敛性 ---
    ns = [64, 128, 256, 512, 1024, 2048]
    dt_fixed = 0.0002     # 很小的固定dt，确保时间误差可忽略
    t_end = 1.0
    rows_space = []

    for n in tqdm(ns, desc="Spatial convergence"):
        x, dx = grid(-20.0, 20.0, n)
        t = np.arange(0.0, t_end + 0.5 * dt_fixed, dt_fixed)
        x0, sigma, k0 = -5.0, 1.0, 2.5
        psi0 = gaussian_wavepacket(x, x0, sigma, k0, dx)
        v = potential_free(x)
        psi_exact = exact_free_gaussian(x, t_end, x0, sigma, k0, dx)
        _, hist = solve("Crank-Nicolson", psi0, v, x, t, dx, dt_fixed, store_every=len(t)-1)
        l1, l2, linf = l1_l2_linf_error(hist[-1], psi_exact, dx)
        mass = probability_mass(hist[-1], dx)
        rows_space.append({"n": n, "dx": dx, "L1": l1, "L2": l2, "Linf": linf,
                           "mass": mass, "mass_err": abs(mass-1.0)})

    df_space = pd.DataFrame(rows_space)

    # --- 时间收敛性 ---
    n_fixed = 1024
    dts = [0.002, 0.001, 0.0005, 0.0002, 0.0001]
    rows_time = []

    x, dx = grid(-20.0, 20.0, n_fixed)
    x0, sigma, k0 = -5.0, 1.0, 2.5
    psi0 = gaussian_wavepacket(x, x0, sigma, k0, dx)
    v = potential_free(x)
    psi_exact = exact_free_gaussian(x, t_end, x0, sigma, k0, dx)

    for dt_val in tqdm(dts, desc="Temporal convergence"):
        t = np.arange(0.0, t_end + 0.5 * dt_val, dt_val)
        _, hist = solve("Crank-Nicolson", psi0, v, x, t, dx, dt_val, store_every=len(t)-1)
        l1, l2, linf = l1_l2_linf_error(hist[-1], psi_exact, dx)
        mass = probability_mass(hist[-1], dx)
        rows_time.append({"dt": dt_val, "L1": l1, "L2": l2, "Linf": linf,
                          "mass": mass, "mass_err": abs(mass-1.0)})

    df_time = pd.DataFrame(rows_time)

    dpi = get_adaptive_dpi(2)

    # ── 图5a: 空间收敛（log-log）──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    ax1.loglog(df_space["dx"], df_space["L2"], '-o', color="#1B998B", lw=2, markersize=7)
    ref_x = df_space["dx"].values
    ax1.loglog(ref_x, 0.5 * ref_x**2, '--', color="#F18F01", lw=1.5, label=r'$\propto \Delta x^2$')
    ax1.set_xlabel(r"$\Delta x$"); ax1.set_ylabel("L2 Error")
    ax1.set_title("Spatial Convergence: CN (Fixed dt)")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

    ax2.semilogy(df_space["n"], df_space["mass_err"], '-o', color="#2E86AB", lw=2, markersize=7)
    ax2.set_xlabel("Grid Size n"); ax2.set_ylabel("|M - 1|")
    ax2.set_title("Mass Conservation Error (Spatial)")
    ax2.grid(True, alpha=0.3)

    fig.savefig(os.path.join(cfg.outdir, "fig5a_convergence_space.png"),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  [Fig5a] Spatial convergence order")

    # 计算实际收敛阶
    if len(df_space) >= 3:
        dx_vals = df_space["dx"].values
        l2_vals = df_space["L2"].values
        # p ≈ log(L2_i / L2_{i+1}) / log(dx_i / dx_{i+1})
        orders = []
        for i in range(len(dx_vals)-1):
            if l2_vals[i+1] > 0 and dx_vals[i+1] > 0:
                p = np.log(l2_vals[i]/l2_vals[i+1]) / np.log(dx_vals[i]/dx_vals[i+1])
                orders.append(p)
        if orders:
            avg_p = np.mean(orders[-3:])  # 用最后几对
            print(f"  Measured spatial convergence order: ~{avg_p:.2f} (expected 2.00)")

    # ── 图5b: 时间收敛 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    ax1.loglog(df_time["dt"], df_time["L2"], '-s', color="#C73E1D", lw=2, markersize=7)
    ref_dt = df_time["dt"].values
    ax1.loglog(ref_dt, 0.5 * ref_dt**2, '--', color="#F18F01", lw=1.5, label=r'$\propto \Delta t^2$')
    ax1.set_xlabel(r"$\Delta t$"); ax1.set_ylabel("L2 Error")
    ax1.set_title("Temporal Convergence: CN (Fixed n)")
    ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

    ax2.semilogy(df_time["dt"], df_time["mass_err"], '-s', color="#2E86AB", lw=2, markersize=7)
    ax2.set_xlabel(r"$\Delta t$"); ax2.set_ylabel("|M - 1|")
    ax2.set_title("Mass Conservation Error (Temporal)")
    ax2.grid(True, alpha=0.3)

    fig.savefig(os.path.join(cfg.outdir, "fig5b_convergence_time.png"),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print("  [Fig5b] Temporal convergence order")

    print("\n  Spatial:")
    print(df_space[["n","dx","L2","mass_err"]].round(8).to_string(index=False))
    print("\n  Temporal:")
    print(df_time[["dt","L2","mass_err"]].round(8).to_string(index=False))


# ──────────────────────────────────────────────────────────────
# 轻量GIF生成函数
# ──────────────────────────────────────────────────────────────

def _save_lightweight_1d_gif_three_line(
    outdir: str, x: Array, psi_hist: Array, saved_t: Array,
    filename: str, title: str = "",
    show_barrier: Optional[Tuple[float, float]] = None,
    fps: int = 12,
) -> None:
    """Save lightweight 1D GIF with THREE lines: |psi|^2 + Re + Im."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    max_frames = 50
    if len(psi_hist) > max_frames:
        step = max(1, len(psi_hist) // max_frames)
        indices = np.arange(0, len(psi_hist), step)[:max_frames]
        psi_hist = np.array(psi_hist)[indices]
        saved_t = np.array(saved_t)[indices]

    all_abs = np.array([np.max(np.abs(p)**2) for p in psi_hist])
    all_re = np.array([np.max(np.real(p)) for p in psi_hist])
    all_im = np.array([np.max(np.imag(p)) for p in psi_hist])
    all_re_min = np.array([np.min(np.real(p)) for p in psi_hist])
    all_im_min = np.array([np.min(np.imag(p)) for p in psi_hist])

    # 缩短x轴范围：只取有效区域（去掉两端空白）
    x_min_plot, x_max_plot = x[0], x[-1]
    # 自动检测有效区域（概率密度>最大值的1e-3的位置，更紧凑）
    global_max_abs = float(np.max(all_abs))
    for frame in psi_hist:
        mask = np.abs(frame)**2 > global_max_abs * 1e-3
        if np.any(mask):
            idx_nonzero = np.where(mask)[0]
            x_min_plot = min(x_min_plot, x[max(0, idx_nonzero[0]-10)])
            x_max_plot = max(x_max_plot, x[min(len(x)-1, idx_nonzero[-1]+10)])

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(7.5, 4.5),
                                          constrained_layout=True,
                                          sharex=True,
                                          gridspec_kw={'height_ratios': [1.2, 1, 1]})

    line_abs, = ax0.plot([], [], lw=1.8, color="#1B998B")
    ax0.set_xlim(x_min_plot, x_max_plot); ax0.set_ylim(0, float(np.max(all_abs))*1.15)
    ax0.set_ylabel(r"$|\psi|^2$", fontsize=11); ax0.grid(True, alpha=0.25)

    line_re, = ax1.plot([], [], lw=1.5, color="#2E86AB")
    yrange_re = max(abs(float(np.max(all_re))), abs(float(np.min(all_re_min)))) * 1.15
    ax1.set_ylim(-yrange_re, yrange_re)
    ax1.set_ylabel(r"Re[$\psi$]", fontsize=11); ax1.grid(True, alpha=0.25)

    line_im, = ax2.plot([], [], lw=1.5, color="#C73E1D")
    yrange_im = max(abs(float(np.max(all_im))), abs(float(np.min(all_im_min)))) * 1.15
    ax2.set_ylim(-yrange_im, yrange_im)
    ax2.set_xlabel("x", fontsize=11); ax2.set_ylabel(r"Im[$\psi$]", fontsize=11)
    ax2.grid(True, alpha=0.25)

    if show_barrier:
        bl, br = show_barrier
        for ax in [ax0, ax1, ax2]:
            ax.axvline(bl, color='red', ls='--', lw=1.2, alpha=0.7)
            ax.axvline(br, color='red', ls='--', lw=1.2, alpha=0.7)
        ax0.axvspan(bl, br, color='gray', alpha=0.12)

    title_text = fig.suptitle(f"{title}, t = {saved_t[0]:.2f}", fontsize=12, y=0.99)

    def init():
        line_abs.set_data([x_min_plot], [0]); line_re.set_data([x_min_plot], [0])
        line_im.set_data([x_min_plot], [0])
        return line_abs, line_re, line_im

    def update(frame):
        psi_f = psi_hist[frame]
        line_abs.set_data(x, np.abs(psi_f)**2)
        line_re.set_data(x, np.real(psi_f)); line_im.set_data(x, np.imag(psi_f))
        title_text.set_text(f"{title}, t = {saved_t[frame]:.2f}")
        return line_abs, line_re, line_im, title_text

    ani = FuncAnimation(fig, update, frames=len(saved_t), init_func=init,
                        blit=False, interval=max(30, 1000//fps), repeat=False)
    ani.save(os.path.join(outdir, filename), writer=PillowWriter(fps=fps), dpi=300)
    plt.close(fig)


def _save_lightweight_2d_gif(
    outdir: str, X: Array, Y: Array, saved_t: Array, hist: Array,
    filename: str, title: str = "",
    obstacle_plot: Optional[Tuple[float, float, float]] = None,
    fps: int = 10,
    extent_override: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """Save lightweight 2D GIF."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    max_frames = 40
    if len(hist) > max_frames:
        step = max(1, len(hist) // max_frames)
        hist = np.array(hist); saved_t = np.array(saved_t)
        indices = np.arange(0, len(saved_t), step)[:max_frames]
        saved_t = saved_t[indices].tolist(); hist = hist[indices]

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    if extent_override is not None:
        extent = list(extent_override)
    else:
        xv = X[:, 0]; yv = Y[0, :]
        extent = [xv[0], xv[-1], yv[0], yv[-1]]

    global_vmax = float(np.max(np.abs(hist) ** 2))
    vmin_val = max(global_vmax * 0.005, 1e-8)

    im = ax.imshow(np.abs(hist[0]).T ** 2, origin="lower", aspect="equal",
                    extent=extent, cmap="inferno", vmin=vmin_val, vmax=global_vmax)
    ax.set_xlabel("x", fontsize=11); ax.set_ylabel("y", fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    if obstacle_plot:
        cx, cy, cr = obstacle_plot
        theta = np.linspace(0, 2*np.pi, 80)
        ax.plot(cx+cr*np.cos(theta), cy+cr*np.sin(theta), "w--", lw=1.5)

    def update(frame):
        im.set_array(np.abs(hist[frame]).T ** 2)
        ax.set_title(f"{title}, t = {saved_t[frame]:.2f}", fontsize=11)
        return [im]

    ani = FuncAnimation(fig, update, frames=len(saved_t), blit=True,
                        interval=max(40, 1000//fps))
    ani.save(os.path.join(outdir, filename), writer=PillowWriter(fps=fps), dpi=300)
    plt.close(fig)


def _save_lightweight_2d_exact_gif(
    outdir: str, X: Array, Y: Array, x: Array, y: Array,
    t_list: Array,
    psi0: Array, sigma: float, kx: float, ky: float,
    dx: float, dy: float,
    filename: str, title: str = "",
    fps: int = 10,
    extent_override: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """Save lightweight 2D GIF using exact analytic solution (computed per frame)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    max_frames = 40
    if len(t_list) > max_frames:
        step = max(1, len(t_list) // max_frames)
        t_list = np.array(t_list)[::step][:max_frames].tolist()

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    if extent_override is not None:
        extent = list(extent_override)
    else:
        xv = X[:, 0]; yv = Y[0, :]
        extent = [xv[0], xv[-1], yv[0], yv[-1]]

    # 第一帧计算初始范围
    psi_first = exact_free_gaussian_2d(X, Y, t_list[0], -6.0, 0.0, sigma, kx, ky, dx, dy)
    global_vmax = float(np.max(np.abs(psi_first) ** 2))
    vmin_val = max(global_vmax * 0.005, 1e-8)

    im = ax.imshow(np.abs(psi_first).T ** 2, origin="lower", aspect="equal",
                    extent=extent, cmap="inferno", vmin=vmin_val, vmax=global_vmax)
    ax.set_xlabel("x", fontsize=11); ax.set_ylabel("y", fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)

    def update(frame):
        t_val = t_list[frame]
        psi_frame = exact_free_gaussian_2d(X, Y, t_val, -6.0, 0.0, sigma, kx, ky, dx, dy)
        im.set_array(np.abs(psi_frame).T ** 2)
        ax.set_title(f"{title}, t = {t_val:.2f}", fontsize=11)
        return [im]

    ani = FuncAnimation(fig, update, frames=len(t_list), blit=True,
                        interval=max(40, 1000//fps))
    ani.save(os.path.join(outdir, filename), writer=PillowWriter(fps=fps), dpi=450)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────

def main(experiment: str = "all"):
    """运行实验。experiment='all' 运行全部，'well' 仅运行无限深势阱。"""
    outdir = "tdse_experiments_v2"
    os.makedirs(outdir, exist_ok=True)

    cfg = RunConfig(outdir=outdir, quick=False, save_gif=True, dpi=600, grid_size=1024)

    print(f"\n{'═'*62}")
    print(f"  TDSE Numerical Methods — ODE Course")
    print(f"  Output: {os.path.abspath(outdir)}")
    print(f"{'═'*62}\n")

    start = time.perf_counter()

    if experiment == "well":
        # 仅运行无限深势阱实验
        experiment_1d_infinite_well(cfg)
    else:
        # 按重要性排序运行全部
        experiment_1d_infinite_well(cfg)       # Exp-01: 主实验（有解析解）
        experiment_1d_tunneling(cfg)           # Exp-02: 扩展实验（无解析解）
        experiment_2d_free(cfg)                # Exp-03: 二维自由传播（有解析解，ADI为主）
        experiment_stability_detailed(cfg)     # Exp-04: 详细稳定性
        experiment_convergence_detailed(cfg)   # Exp-05: 详细收敛性

    elapsed = time.perf_counter() - start
    print(f"\n{'═'*62}")
    print(f"  All done! Total: {elapsed:.1f}s")
    print(f"  Output: {outdir}/")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
