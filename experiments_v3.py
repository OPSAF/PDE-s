#!/usr/bin/env python3
"""
TDSE 数值方法对比实验 — v3 解析解模型版

三个一维模型，均有闭式解析解（无需数值积分）：

  模型一  自由电子高斯波包（含初始啁啾）
          → V=0 | 啁啾高斯初值 | 波包扩散+漂移

  模型二  谐振子相干态（永不扩散）
          → V=½ω²x² | 相干态初值 | 经典轨道振荡

  模型三  正弦电场驱动的量子点电子
          → V=½ω²x² - eE₀cos(Ωt)x | 受迫振荡 | 显式非定态解析解

原子单位制：ħ = 1, m = 1, e = 1
"""

import os
import time
from typing import Optional, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from tdse.config import RunConfig
from tdse.potentials import (
    Array,
    grid,
    normalize,
    probability_mass,
    l1_l2_linf_error,
    l2_error_real_imag,
)
from tdse.solvers import solve


# ──────────────────────────────────────────────────────────────
# 解析解函数（v3.txt 三个模型的闭式表达式）
# ──────────────────────────────────────────────────────────────
#
# 统一使用原子单位：ħ = 1, m = 1, e = 1

def exact_free_chirped_gaussian(x: Array, t: float,
                                 sigma0: float, k0: float, alpha: float) -> Array:
    """
    模型一：自由电子含啁啾高斯波包的精确解。

    方程：i ∂ψ/∂t = -½ ∂²ψ/∂x²   （V = 0）

    初值：ψ(x,0) = (2πσ₀²)^(-1/4) exp[-x²/(4σ₀²) + i k₀ x + i α x²/2]

    闭式解（v3.txt 公式，ħ=m=1）：
        ψ(x,t) = N · D(t)^(-1/2) · exp[Q(t)·(x - k₀t)² + i k₀ x - i k₀² t / 2]

    其中：
        D(t) = 1 + i t/(2σ₀²) + i α t
        Q(t) = (-1/(4σ₀²) + iα/2) / D(t)
        N = (2πσ₀²)^(-1/4)
    """
    # 复数分母
    D = 1.0 + 1j * t / (2.0 * sigma0**2) + 1j * alpha * t
    Q = (-1.0 / (4.0 * sigma0**2) + 0.5j * alpha) / D
    prefactor = (2.0 * np.pi * sigma0**2) ** (-0.25) / np.sqrt(D)

    psi = prefactor * np.exp(
        Q * (x - k0 * t) ** 2
        + 1j * k0 * x
        - 0.5j * k0**2 * t
    )
    return psi.astype(complex)


def exact_harmonic_coherent(x: Array, t: float,
                             omega: float, x0: float, p0: float) -> Array:
    """
    模型二：谐振子相干态的精确解（永不扩散的高斯波包）。

    方程：i ∂ψ/∂t = -½ ∂²ψ/∂x² + ½ ω² x² ψ

    初值：ψ(x,0) = (ω/π)^(1/4) exp[-ω(x-x0)²/2 + i p₀(x - x0/2)]

    闭式解（ħ=m=1）：
        ψ(x,t) = (ω/π)^(1/4) · exp[ -ω(x-x_t)²/2 + i p_t(x - x_t/2)
                                      - iωt/2 + i(p₀x₀ - p_t x_t)/2 ]

    其中经典轨迹：
        x_t = x₀ cos(ωt) + (p₀/ω) sin(ωt)
        p_t = p₀ cos(ωt) - ω x₀ sin(ωt)
    """
    ct = np.cos(omega * t)
    st = np.sin(omega * t)

    x_t = x0 * ct + (p0 / omega) * st
    p_t = p0 * ct - omega * x0 * st

    N = (omega / np.pi) ** 0.25
    psi = N * np.exp(
        -0.5 * omega * (x - x_t) ** 2
        + 1j * p_t * (x - x_t / 2.0)
        - 0.5j * omega * t
        + 0.5j * (p0 * x0 - p_t * x_t)
    )
    return psi.astype(complex)


def exact_driven_harmonic(x: Array, t: float,
                           omega: float, Omega: float, E0: float,
                           x0: float, p0: float) -> Array:
    """
    模型三：正弦驱动谐振子的精确解（显式非定态解）。

    方程：i ∂ψ/∂t = -½ ∂²ψ/∂x² + [½ ω² x² - E₀ cos(Ωt) x] ψ

    初值：与模型二相同（相干态形式）

    闭式解（ħ=m=e=1，非共振 Ω≠ω）：

    定义常数 A = E₀ / [m(ω² - Ω²)] = E₀ / (ω² - Ω²)

    经典轨迹：
        x_c(t) = (x₀ - A) cos(ωt) + (p₀/ω) sin(ωt) + A cos(Ωt)
        p_c(t) = -ω(x₀ - A) sin(ωt) + p₀ cos(ωt) - ΩA sin(Ωt)

    作用量相位 S_c(t)：三角多项式显式表达式（见 v3.txt）

    波函数：
        ψ(x,t) = (ω/π)^(1/4) · exp[ -ω(x-x_c)²/2 + i p_c(x - x_c/2)
                                      + i S_c(t)/ħ - iωt/2 ]
    """
    # 常数 A
    A = E0 / (omega**2 - Omega**2)

    ct_w = np.cos(omega * t)
    st_w = np.sin(omega * t)
    ct_O = np.cos(Omega * t)
    st_O = np.sin(Omega * t)

    # 经典轨迹
    x_c = (x0 - A) * ct_w + (p0 / omega) * st_w + A * ct_O
    p_c = -omega * (x0 - A) * st_w + p0 * ct_w - omega * Omega * A * st_O  # m=1

    # 作用量相位 S_c(t) — 完全由三角函数组成
    denom = omega**2 - Omega**2
    Sc = (
        0.5 * omega * (x0 - A) * p0 * st_w**2
        + 0.5 * (p0**2 / omega - omega * (x0 - A)**2) * st_w * ct_w
        + 0.5 * omega * Omega * A**2 * st_O * ct_O
        + omega * A * (x0 - A) * (omega * st_w * ct_O - Omega * ct_w * st_O) / denom
        + p0 * A * (omega * st_w * st_O + Omega * ct_w * ct_O - Omega) / denom
        + 0.5 * p0 * (x0 - A) - 0.5 * p0 * x0
    )

    N = (omega / np.pi) ** 0.25
    psi = N * np.exp(
        -0.5 * omega * (x - x_c) ** 2
        + 1j * p_c * (x - x_c / 2.0)
        + 1j * Sc
        - 0.5j * omega * t
    )
    return psi.astype(complex)


# ──────────────────────────────────────────────────────────────
# 含时势求解器包装器
# ──────────────────────────────────────────────────────────────

def solve_time_dependent(method: str, psi0: Array,
                          V_fn: Callable[[Array, float], Array],
                          x: Array, t: Array, dx: float, dt: float,
                          store_every: int = 1) -> Tuple[Array, Array]:
    """
    支持含时势 V(x,t) 的统一求解接口。

    各方法对时间依赖的处理：
      FTCS / RK4     : 在每个子步评估 V(x, t_current)
      Backward-Euler : 隐式步用 V(x, t+dt)
      Crank-Nicolson : 平均 V(x,t) 和 V(x, t+dt)
      Split-Step FFT : Strang 分裂中半步用对应时刻的 V
    """
    from tdse.solvers import (
        step_ftcs, step_backward_euler, step_crank_nicolson,
        step_rk4, step_split_step_fft, banded_hamiltonian,
        apply_tridiagonal, solve_banded, hamiltonian_apply,
    )

    method_key = method.lower()
    psi = psi0.astype(complex).copy()
    saved_t = [float(t[0])]
    saved_psi = [psi.copy()]

    for n in range(1, len(t)):
        tn = float(t[n])
        tn_prev = float(t[n - 1])

        v_curr = V_fn(x, tn_prev)
        v_next = V_fn(x, tn)

        if method_key == "ftcs":
            psi = step_ftcs(psi, v_curr, dx, dt)

        elif method_key in ("backward-euler", "be"):
            # BE: (I + i dt H_{n+1}) psi^{n+1} = psi^n
            h = banded_hamiltonian(v_next, dx)
            a = h.copy()
            a[1] = 1.0 + 1j * dt * h[1]
            a[0] = 1j * dt * h[0]
            a[2] = 1j * dt * h[2]
            psi = solve_banded((1, 1), a, psi)

        elif method_key in ("crank-nicolson", "cn"):
            # CN: (I + i dt/2 H_{n+1}) psi^{n+1} = (I - i dt/2 H_n) psi^n
            h_next = banded_hamiltonian(v_next, dx)
            h_curr = banded_hamiltonian(v_curr, dx)

            left = h_next.copy()
            left[1] = 1.0 + 0.5j * dt * h_next[1]
            left[0] = 0.5j * dt * h_next[0]
            left[2] = 0.5j * dt * h_next[2]

            right = h_curr.copy()
            right[1] = 1.0 - 0.5j * dt * h_curr[1]
            right[0] = -0.5j * dt * h_curr[0]
            right[2] = -0.5j * dt * h_curr[2]
            rhs = apply_tridiagonal(right, psi)
            psi = solve_banded((1, 1), left, rhs)

        elif method_key == "rk4":
            def f(y, tt):
                v_tt = V_fn(x, tt)
                return -1j * hamiltonian_apply(y, v_tt, dx)

            k1 = f(psi, tn_prev)
            k2 = f(psi + 0.5 * dt * k1, tn_prev + 0.5 * dt)
            k3 = f(psi + 0.5 * dt * k2, tn_prev + 0.5 * dt)
            k4 = f(psi + dt * k3, tn)
            psi = psi + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

        elif method_key in ("split-step", "split-step-fft", "fft", "ssf"):
            # SSF with time-dependent V:
            # exp(-i V(t+dt/2) dt/2) -> FFT -> exp(-i k^2 dt/2) -> IFFT -> exp(-i V(t+dt/2) dt/2)
            v_mid = V_fn(x, (tn_prev + tn) / 2.0)
            psi_half = np.exp(-0.5j * v_mid * dt) * psi
            k = 2.0 * np.pi * np.fft.fftfreq(len(x), d=dx)
            psi_k = np.fft.fft(psi_half)
            psi_k *= np.exp(-0.5j * k**2 * dt)
            psi_new = np.fft.ifft(psi_k)
            psi = np.exp(-0.5j * v_mid * dt) * psi_new

        else:
            raise ValueError(f"Unknown method: {method}")

        if n % store_every == 0 or n == len(t) - 1:
            saved_t.append(float(t[n]))
            saved_psi.append(psi.copy())

    return np.array(saved_t), np.array(saved_psi)


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

METHODS = ["FTCS", "Backward-Euler", "Crank-Nicolson", "RK4", "Split-Step-FFT"]
COLORS = ["#9B9B9B", "#2E86AB", "#1B998B", "#F18F01", "#C73E1D"]
LABELS = ["FTCS", "BE", "CN", "RK4", "SSF"]


def get_adaptive_dpi(num_subplots: int) -> int:
    if num_subplots <= 3:
        return 600
    elif num_subplots <= 6:
        return 800
    else:
        return 1000


def run_1d_methods_static(methods, psi0, v, x, t, dx, dt):
    """运行静态势的五种方法（使用原有solve接口）。"""
    results = {}
    for method in methods:
        try:
            _, hist = solve(method, psi0, v, x, t, dx, dt, store_every=len(t)-1)
            mass = probability_mass(hist[-1], dx)
            results[method] = {"psi": hist[-1], "stable": True, "mass": mass}
        except Exception as e:
            print(f"  {method} failed: {e}")
            results[method] = {"psi": None, "stable": False, "mass": np.nan}
    return results


def run_1d_methods_td(methods, psi0, V_fn, x, t, dx, dt):
    """运行含时势的五种方法。"""
    results = {}
    for method in methods:
        try:
            _, hist = solve_time_dependent(method, psi0, V_fn, x, t, dx, dt,
                                            store_every=len(t)-1)
            mass = probability_mass(hist[-1], dx)
            results[method] = {"psi": hist[-1], "stable": True, "mass": mass}
        except Exception as e:
            print(f"  {method} failed: {e}")
            results[method] = {"psi": None, "stable": False, "mass": np.nan}
    return results


def plot_three_panel(x, results, methods, colors, labels, title_prefix,
                     outdir, filename, dpi=600, psi_exact=None,
                     show_v=None, v_x=None):
    """标准三面板：|psi|^2, Re[psi], Im[psi]（可选含解析解和势场）"""
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

    if show_v and v_x is not None:
        ax2 = axes[0].twinx()
        vmax_plot = np.max(np.abs(v_x)) if np.max(np.abs(v_x)) > 0 else 1.0
        ax2.plot(x, v_x / vmax_plot * axes[0].get_ylim()[1] * 0.35,
                 'gray', ':', lw=1.0, alpha=0.6)
        ax2.set_yticks([])

    for ax in axes:
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r"$x$", fontsize=11)
    fig.savefig(os.path.join(outdir, filename), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_error_analysis(x, results, methods_all, colors_all, psi_exact, dx,
                         outdir, filename_prefix, dpi=None):
    """完整误差分析：单独对比图 + 6指标柱状图 + CSV表格。"""
    if dpi is None:
        dpi = get_adaptive_dpi(6)

    stable_methods = [m for m in methods_all
                      if results[m]["stable"] and results[m]["psi"] is not None]

    # ── 子图：各稳定方法 vs Exact ──
    n_stable = len(stable_methods)
    if n_stable > 0:
        ncols = min(n_stable, 2)
        nrows = (n_stable + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows),
                                  constrained_layout=True)
        if n_stable == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for idx, method in enumerate(stable_methods):
            ax = axes_flat[idx]
            r = results[method]
            ax.plot(x, np.abs(psi_exact)**2, 'k--', lw=1.8, label='Exact', alpha=0.8)
            ax.plot(x, np.abs(r["psi"])**2, lw=1.5, label=method)
            l1, l2, linf = l1_l2_linf_error(r["psi"], psi_exact, dx)
            ax.set_title(f"{method}: L1={l1:.2e}  L2={l2:.2e}  L∞={linf:.2e}", fontsize=10)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlabel("x")
        # 隐藏多余子图
        for idx in range(n_stable, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        fig.savefig(os.path.join(outdir, f"{filename_prefix}_vs_exact.png"),
                    dpi=get_adaptive_dpi(n_stable), bbox_inches='tight')
        plt.close(fig)

    # ── 误差数据表 + 柱状图 ──
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
    df_err.to_csv(os.path.join(outdir, f"{filename_prefix}_errors.csv"), index=False)

    # 柱状图
    valid_methods = [m for m in methods_all
                     if np.isfinite(df_err.loc[df_err["Method"]==m, "L2"].values[0])]
    metrics = [
        ("L1 Error", "L1", True),
        ("L2 Error", "L2", True),
        ("Linf Error", "Linf", True),
        ("Relative L2", "Rel_L2", True),
        ("L2 Real Part", "L2_Re", True),
        ("L2 Imag Part", "L2_Im", True),
    ]
    color_dict = dict(zip(methods_all, colors_all))

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for idx, (title, col, use_log) in enumerate(metrics):
        ax = axes[idx // 3][idx % 3]
        vals = [df_err.loc[df_err["Method"]==m, col].values[0] for m in valid_methods]
        clrs = [color_dict[m] for m in valid_methods]
        bars = ax.bar(range(len(valid_methods)), vals, color=clrs, edgecolor='white', lw=0.5)
        if use_log:
            vals_finite = [v for v in vals if np.isfinite(v) and v > 0]
            if vals_finite:
                ax.set_yscale('log')
                ax.set_ylim(min(vals_finite)*0.5, max(vals_finite)*5.0)
        ax.set_xticks(range(len(valid_methods)))
        ax.set_xticklabels(valid_methods, fontsize=9, rotation=10)
        ax.set_title(title, fontsize=11); ax.grid(True, alpha=0.3)
        for bi, (bar, val) in enumerate(zip(bars, vals)):
            if np.isfinite(val) and val > 0:
                bh = bar.get_height()
                bx = bar.get_x() + bar.get_width() / 2.
                ax.text(bx, bh * 1.12, f'{val:.2e}', ha='center', va='bottom',
                       fontsize=7, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.15',
                                facecolor=clrs[bi], alpha=0.8, edgecolor='none'),
                       zorder=10)

    fig.suptitle(f"{filename_prefix.replace('_', ' ').title()} - Complete Error Analysis",
                 fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outdir, f"{filename_prefix}_error_analysis.png"),
                dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(df_err.round(8).to_string(index=False))
    return df_err


# ──────────────────────────────────────────────────────────────
# 实验①：模型一 — 自由电子含啁啾高斯波包
# ──────────────────────────────────────────────────────────────

def experiment_model1_free_chirped(cfg):
    """模型一：自由电子高斯波包（含啁啾），V=0，有解析解。"""
    print("=" * 60)
    print("Model-1: Free Chirped Gaussian Wavepacket (V=0)")
    print("=" * 60)

    n = 2048
    x, dx = grid(-20.0, 20.0, n)
    dt = 0.0005
    t_end = 3.0
    t = np.arange(0.0, t_end + 0.5 * dt, dt)

    # 参数（原子单位）
    sigma0 = 1.0       # 初始宽度
    k0 = 2.5           # 初始波数（群速度 = k0）
    alpha = 0.3        # 啁啾参数

    # 初值：啁啾高斯
    psi0 = exact_free_chirped_gaussian(x, 0.0, sigma0, k0, alpha)
    psi0 = normalize(psi0, dx)

    v = np.zeros_like(x, dtype=float)  # V = 0

    print(f"  σ₀={sigma0}, k₀={k0}, α={alpha}, n={n}, dt={dt}, t_end={t_end}")

    results = run_1d_methods_static(METHODS, psi0, v, x, t, dx, dt)

    # 解析解
    psi_exact = exact_free_chirped_gaussian(x, t_end, sigma0, k0, alpha)
    psi_exact = normalize(psi_exact, dx)

    dpi = get_adaptive_dpi(3)

    # 图1a: 全部5方法 + Exact
    plot_three_panel(
        x, results, METHODS, COLORS, LABELS,
        "Free Chirped Gaussian (all)", cfg.outdir,
        "model1a_all.png", dpi=dpi, psi_exact=psi_exact)
    print("  [Fig1a] All 5 methods + Exact: three-value")

    # 图1b: 稳定方法 + Exact
    m_stable = METHODS[1:]  # 去掉FTCS
    c_stable = COLORS[1:]
    l_stable = LABELS[1:]
    plot_three_panel(
        x, results, m_stable, c_stable, l_stable,
        "Free Chirped Gaussian (stable)", cfg.outdir,
        "model1b_stable.png", dpi=dpi, psi_exact=psi_exact)
    print("  [Fig1b] Stable methods + Exact: three-value")

    # 误差分析
    plot_error_analysis(
        x, results, METHODS, COLORS, psi_exact, dx,
        cfg.outdir, "model1", dpi=dpi)
    print("  [Error] Complete error analysis")

    print(f"  Grid: n={n}, dt={dt}, steps={len(t)-1}")

    # ── 动图（CN，三线）──
    if results["Crank-Nicolson"]["stable"]:
        print("  Generating animation (CN, free chirped gaussian)...")
        t_long = np.arange(0.0, t_end + 0.5 * dt, dt)
        _, hist_long = solve("Crank-Nicolson", psi0, v, x, t_long, dx, dt,
                             store_every=int(len(t_long)/80))
        t_snap = t_long[::int(len(t_long)//80)][:len(hist_long)]
        _save_lightweight_1d_gif_three_line(
            cfg.outdir, x, hist_long, t_snap,
            "model1_gif.gif",
            title=f"Free Chirped Gaussian (sigma0={sigma0}, k0={k0})",
            fps=24
        )
        print("  [GIF1] Free chirped gaussian animation")


# ──────────────────────────────────────────────────────────────
# 实验②：模型二 — 谐振子相干态
# ──────────────────────────────────────────────────────────────

def experiment_model2_harmonic_coherent(cfg):
    """模型二：谐振子相干态（永不扩散），V=½ω²x²，有解析解。"""
    print("=" * 60)
    print("Model-2: Harmonic Oscillator Coherent State")
    print("=" * 60)

    n = 2048
    x, dx = grid(-15.0, 15.0, n)
    dt = 0.0005
    t_end = 4.0 * np.pi  # 两个完整振荡周期（T=2π/ω, ω=1 → T=2π）
    t = np.arange(0.0, t_end + 0.5 * dt, dt)

    omega = 1.0           # 谐振子频率
    x0 = 2.0             # 初始位移
    p0 = 1.5             # 初始动量

    # 初值：相干态
    psi0 = exact_harmonic_coherent(x, 0.0, omega, x0, p0)
    psi0 = normalize(psi0, dx)

    # 势场：谐振子
    v = 0.5 * omega**2 * x**2

    print(f"  ω={omega}, x₀={x0}, p₀={p0}, n={n}, dt={dt}, t_end={t_end:.2f} ({t_end/np.pi:.1f}π)")

    results = run_1d_methods_static(METHODS, psi0, v, x, t, dx, dt)

    # 解析解
    psi_exact = exact_harmonic_coherent(x, t_end, omega, x0, p0)
    psi_exact = normalize(psi_exact, dx)

    dpi = get_adaptive_dpi(3)

    # 图2a: 全部 + Exact
    plot_three_panel(
        x, results, METHODS, COLORS, LABELS,
        "Harmonic Coherent (all)", cfg.outdir,
        "model2a_all.png", dpi=dpi, psi_exact=psi_exact,
        show_v=True, v_x=v)
    print("  [Fig2a] All 5 methods + Exact: three-value (+ potential)")

    # 图2b: 稳定方法
    m_stable = METHODS[1:]
    plot_three_panel(
        x, results, m_stable, COLORS[1:], LABELS[1:],
        "Harmonic Coherent (stable)", cfg.outdir,
        "model2b_stable.png", dpi=dpi, psi_exact=psi_exact,
        show_v=True, v_x=v)
    print("  [Fig2b] Stable methods + Exact: three-value")

    # 误差分析
    plot_error_analysis(
        x, results, METHODS, COLORS, psi_exact, dx,
        cfg.outdir, "model2", dpi=dpi)
    print("  [Error] Complete error analysis")

    print(f"  Grid: n={n}, dt={dt}, steps={len(t)-1}")

    # ── 动图（CN，三线）──
    if results["Crank-Nicolson"]["stable"]:
        print("  Generating animation (CN, harmonic coherent)...")
        t_long = np.arange(0.0, t_end + 0.5 * dt, dt)
        _, hist_long = solve("Crank-Nicolson", psi0, v, x, t_long, dx, dt,
                             store_every=int(len(t_long)/80))
        t_snap = t_long[::int(len(t_long)//80)][:len(hist_long)]
        _save_lightweight_1d_gif_three_line(
            cfg.outdir, x, hist_long, t_snap,
            "model2_gif.gif",
            title=f"Harmonic Coherent (omega={omega})",
            fps=24
        )
        print("  [GIF2] Harmonic coherent state animation")


# ──────────────────────────────────────────────────────────────
# 实验③：模型三 — 正弦驱动量子点电子（含时势！）
# ──────────────────────────────────────────────────────────────

def experiment_model3_driven(cfg):
    """模型三：正弦驱动谐振子，V(x,t)=½ω²x²-E₀cos(Ωt)x，有解析解。"""
    print("=" * 60)
    print("Model-3: Sinusoidally Driven Quantum Dot (Time-Dependent V)")
    print("=" * 60)

    n = 2048
    x, dx = grid(-15.0, 15.0, n)
    dt = 0.0005
    t_end = 6.0 * np.pi  # 多个周期以展示受迫行为
    t = np.arange(0.0, t_end + 0.5 * dt, dt)

    omega = 1.0           # 谐振子固有频率
    Omega = 0.6           # 驱动频率（非共振：Ω ≠ ω）
    E0 = 0.8             # 电场幅度
    x0 = 2.0             # 初始位移
    p0 = 1.0             # 初始动量

    # 含时势函数
    def V_driven(x_arr, t_val):
        return 0.5 * omega**2 * x_arr**2 - E0 * np.cos(Omega * t_val) * x_arr

    # 初值：相干态形式
    psi0 = exact_driven_harmonic(x, 0.0, omega, Omega, E0, x0, p0)
    psi0 = normalize(psi0, dx)

    print(f"  ω={omega}, Ω={Omega}, E₀={E0}, x₀={x0}, p₀={p0}")
    print(f"  n={n}, dt={dt}, t_end={t_end:.2f} (non-resonant: Ω/ω={Omega/omega:.2f})")

    # 使用含时势求解器
    results = run_1d_methods_td(METHODS, psi0, V_driven, x, t, dx, dt)

    # 解析解
    psi_exact = exact_driven_harmonic(x, t_end, omega, Omega, E0, x0, p0)
    psi_exact = normalize(psi_exact, dx)

    # 最终时刻的势场（用于绘图显示）
    v_final = V_driven(x, t_end)

    dpi = get_adaptive_dpi(3)

    # 图3a: 全部 + Exact
    plot_three_panel(
        x, results, METHODS, COLORS, LABELS,
        "Driven Harmonic (all)", cfg.outdir,
        "model3a_all.png", dpi=dpi, psi_exact=psi_exact,
        show_v=True, v_x=v_final)
    print("  [Fig3a] All 5 methods + Exact: three-value (+ V at t_end)")

    # 图3b: 稳定方法
    m_stable = METHODS[1:]
    plot_three_panel(
        x, results, m_stable, COLORS[1:], LABELS[1:],
        "Driven Harmonic (stable)", cfg.outdir,
        "model3b_stable.png", dpi=dpi, psi_exact=psi_exact,
        show_v=True, v_x=v_final)
    print("  [Fig3b] Stable methods + Exact: three-value")

    # 误差分析
    plot_error_analysis(
        x, results, METHODS, COLORS, psi_exact, dx,
        cfg.outdir, "model3", dpi=dpi)
    print("  [Error] Complete error analysis")

    print(f"  Grid: n={n}, dt={dt}, steps={len(t)-1}")

    # ── 动图（CN，三线，含时势）──
    if results["Crank-Nicolson"]["stable"]:
        print("  Generating animation (CN, driven harmonic)...")
        t_long = np.arange(0.0, t_end + 0.5 * dt, dt)
        _, hist_long = solve_time_dependent("Crank-Nicolson", psi0, V_driven,
                                             x, t_long, dx, dt,
                                             store_every=int(len(t_long)/80))
        t_snap = t_long[::int(len(t_long)//80)][:len(hist_long)]
        _save_lightweight_1d_gif_three_line(
            cfg.outdir, x, hist_long, t_snap,
            "model3_gif.gif",
            title=f"Driven Harmonic (omega={omega}, Omega={Omega})",
            fps=24
        )
        print("  [GIF3] Driven harmonic animation")


# ──────────────────────────────────────────────────────────────
# 动画生成
# ──────────────────────────────────────────────────────────────

def _save_lightweight_1d_gif_three_line(
    outdir: str, x: Array, psi_hist: Array, saved_t: Array,
    filename: str, title: str = "",
    fps: int = 12,
) -> None:
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

    x_min_plot, x_max_plot = x[0], x[-1]
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


# ──────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────

def main():
    outdir = "tdse_experiments_v3"
    os.makedirs(outdir, exist_ok=True)

    cfg = RunConfig(outdir=outdir, quick=False, save_gif=True, dpi=600, grid_size=1024)

    print(f"\n{'═'*62}")
    print(f"  TDSE Numerical Methods — v3 Analytic Solution Models")
    print(f"  Output: {os.path.abspath(outdir)}")
    print(f"{'═'*62}\n")

    start = time.perf_counter()

    # 三个一维模型，均有闭式解析解
    experiment_model1_free_chirped(cfg)    # 模型一：自由含啁啾高斯
    experiment_model2_harmonic_coherent(cfg)  # 模型二：谐振子相干态
    experiment_model3_driven(cfg)           # 模型三：正弦驱动量子点

    elapsed = time.perf_counter() - start
    print(f"\n{'═'*62}")
    print(f"  All done! Total: {elapsed:.1f}s")
    print(f"  Output: {outdir}/")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
