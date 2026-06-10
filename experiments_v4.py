#!/usr/bin/env python3
"""
TDSE 数值方法对比实验 — v4 严苛基准模型版

两个一维模型，均有闭式解析解，波函数形态更复杂（干涉条纹 + 呼吸振荡）：

  模型四  自由电子双高斯波包干涉
          → V=0 | 双波包反向运动 | 干涉条纹 | 色散敏感

  模型五  谐振子压缩态（呼吸波包）
          → V=½ω²x² | 宽度周期振荡 | 快速形变 | 非定态传播

原子单位制：ħ = 1, m = 1
"""

import os
import time
from typing import Optional

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
# 解析解函数（v4.txt 的两个模型的闭式表达式）
# ──────────────────────────────────────────────────────────────
#
# 统一使用原子单位：ħ = 1, m = 1


def _single_free_gaussian(x: Array, t: float, x0_shift: float, k_sign: float,
                           sigma0: float, k0: float) -> Array:
    """
    单个自由高斯波包的精确解（无啁啾），用于构造模型四的双波包。

    ψ_±(x,t) = N · D(t)^(-1/2) · exp[Q(t)·(x - x_center)² ± i k₀ x - i k₀² t / 2]

    其中 x_center = x0_shift ± k₀ t, k_sign = ±1 控制动量方向。
    """
    D = 1.0 + 1j * t / (2.0 * sigma0**2)
    Q = -1.0 / (4.0 * sigma0**2) / D
    prefactor = (2.0 * np.pi * sigma0**2) ** (-0.25) / np.sqrt(D)

    psi = prefactor * np.exp(
        Q * (x - x0_shift - k_sign * k0 * t) ** 2
        + 1j * k_sign * k0 * x
        - 0.5j * k0**2 * t
    )
    return psi.astype(complex)


def exact_double_gaussian(x: Array, t: float,
                            sigma0: float, d: float, k0: float) -> Array:
    """
    模型四：自由电子双高斯波包干涉的精确解。

    方程：i ∂ψ/∂t = -½ ∂²ψ/∂x²   （V = 0）

    初值：ψ(x,0) = (1/√2)[ψ₊(x,0) + ψ₋(x,0)]
      其中 ψ₊ 中心在 +d/2，动量 +k₀
            ψ₋ 中心在 -d/2，动量 -k₀

    闭式解：线性叠加（每个波包独立自由演化）
      ψ(x,t) = [ψ₊(x,t) + ψ₋(x,t)] / √2
    """
    psi_plus = _single_free_gaussian(x, t, d/2.0, +1.0, sigma0, k0)
    psi_minus = _single_free_gaussian(x, t, -d/2.0, -1.0, sigma0, k0)
    return (psi_plus + psi_minus) / np.sqrt(2.0)


def exact_squeezed_harmonic(x: Array, t: float,
                              omega: float, r: float,
                              x0: float, p0: float) -> Array:
    """
    模型五：谐振子压缩态（呼吸波包）的精确解。

    方程：i ∂ψ/∂t = -½ ∂²ψ/∂x² + ½ ω² x² ψ

    初始压缩态（压缩参数 r > 0 为位置压缩）：
      ψ(x,0) = (ω/π)^(1/4) · sqrt(sech r) · exp[-ω e^{-2r}(x-x₀)²/2 + i p₀(x-x₀/2)]

    闭式解（ħ=m=1）：

    定义复数宽度参数：
      α(t) = [e^{-2r} cos(ωt) + i e^{2r} sin(ωt)] / [cos(ωt) + i e^{-2r} sin(ωt)]

    归一化因子：
      N(t) = 1 / sqrt(cosh r + e^{iωt} sinh r)

    经典轨迹（与相干态相同）：
      x_c(t) = x₀ cos(ωt) + (p₀/ω) sin(ωt)
      p_c(t) = p₀ cos(ωt) - ω x₀ sin(ωt)

    波函数：
      ψ(x,t) = (ω/π)^(1/4) · N(t) · exp[ -ω α(t) (x-x_c)²/2
                                       + i p_c (x - x_c/2) - iωt/2 + C ]
    其中常数 C 保证归一化一致性。
    """
    ct = np.cos(omega * t)
    st = np.sin(omega * t)

    # 复数宽度参数 α(t)
    denom_alpha = ct + 1j * np.exp(-2.0 * r) * st
    alpha_t = (np.exp(-2.0 * r) * ct + 1j * np.exp(2.0 * r) * st) / denom_alpha

    # 归一化因子
    N_t = 1.0 / np.sqrt(np.cosh(r) + np.exp(1j * omega * t) * np.sinh(r))

    # 经典轨迹
    x_c = x0 * ct + (p0 / omega) * st
    p_c = p0 * ct - omega * x0 * st

    # 基础归一化系数
    base_N = (omega / np.pi) ** 0.25

    # 常数相位项（确保 t=0 时与初值一致）
    # 在 t=0 时：α(0)=e^{-2r}, N(0)=1/sqrt(cosh r+sinh r)=e^{-r/2}, x_c=x0, p_c=p0
    # 初值的指数部分: -ω e^{-2r}(x-x0)^2/2 + i p0(x-x0/2)
    # 解的指数部分在 t=0: -ω α(0)(x-x0)^2/2 + i p0(x-x0/2) - i*0 + C
    # 所以 C 应使得整体一致。由于 N(0)*base_N = (ω/π)^(1/4) * e^{-r/2}
    # 而初值有 sqrt(sech r) = e^{-r/2}（近似），所以 C ≈ 0 即可
    const_phase = 0.5j * (p0 * x0 - p_c * x_c)  # 与相干态一致的相位修正

    psi = base_N * N_t * np.exp(
        -0.5 * omega * alpha_t * (x - x_c) ** 2
        + 1j * p_c * (x - x_c / 2.0)
        - 0.5j * omega * t
        + const_phase
    )
    return psi.astype(complex)


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
    """运行静态势的五种方法。"""
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


def plot_three_panel(x, results, methods, colors, labels, title_prefix,
                     outdir, filename, dpi=600, psi_exact=None,
                     show_v=None, v_x=None):
    """标准三面板：|psi|^2, Re[psi], Im[psi]"""
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
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='#FFE4E4', edgecolor='red'))

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
        for idx in range(n_stable, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        fig.savefig(os.path.join(outdir, f"{filename_prefix}_vs_exact.png"),
                    dpi=get_adaptive_dpi(n_stable), bbox_inches='tight')
        plt.close(fig)

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


def _save_lightweight_1d_gif_three_line(
    outdir: str, x: Array, psi_hist: Array, saved_t: Array,
    filename: str, title: str = "",
    fps: int = 12,
) -> None:
    """Save lightweight 1D GIF with THREE lines."""
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
# 实验④：模型四 — 自由电子双高斯波包干涉
# ──────────────────────────────────────────────────────────────

def experiment_model4_double_gaussian(cfg):
    """模型四：自由电子双高斯波包干涉（V=0），干涉条纹检验色散精度。"""
    print("=" * 60)
    print("Model-4: Double Gaussian Wavepacket Interference (V=0)")
    print("=" * 60)

    n = 2048
    x, dx = grid(-25.0, 25.0, n)
    dt = 0.0005
    t_end = 4.0          # 足够长以观察干涉条纹演化
    t = np.arange(0.0, t_end + 0.5 * dt, dt)

    # 双高斯参数
    sigma0 = 0.8         # 较窄的高斯（尖锐干涉条纹）
    d = 6.0              # 初始分离距离
    k0 = 3.0             # 反向初始动量

    # 初值：双高斯叠加
    psi0 = exact_double_gaussian(x, 0.0, sigma0, d, k0)
    psi0 = normalize(psi0, dx)

    v = np.zeros_like(x, dtype=float)  # V = 0

    print(f"  sigma0={sigma0}, d={d}, k0={k0}, n={n}, dt={dt}, t_end={t_end}")

    results = run_1d_methods_static(METHODS, psi0, v, x, t, dx, dt)

    # 解析解
    psi_exact = exact_double_gaussian(x, t_end, sigma0, d, k0)
    psi_exact = normalize(psi_exact, dx)

    dpi = get_adaptive_dpi(3)

    # 图4a: 全部 + Exact
    plot_three_panel(
        x, results, METHODS, COLORS, LABELS,
        "Double Gaussian (all)", cfg.outdir,
        "model4a_all.png", dpi=dpi, psi_exact=psi_exact)
    print("  [Fig4a] All 5 methods + Exact: three-value")

    # 图4b: 稳定方法
    m_stable = METHODS[1:]
    plot_three_panel(
        x, results, m_stable, COLORS[1:], LABELS[1:],
        "Double Gaussian (stable)", cfg.outdir,
        "model4b_stable.png", dpi=dpi, psi_exact=psi_exact)
    print("  [Fig4b] Stable methods + Exact: three-value")

    # 误差分析
    plot_error_analysis(
        x, results, METHODS, COLORS, psi_exact, dx,
        cfg.outdir, "model4", dpi=dpi)
    print("  [Error] Complete error analysis")

    # 动图
    if results["Crank-Nicolson"]["stable"]:
        print("  Generating animation (CN, double gaussian interference)...")
        t_long = np.arange(0.0, t_end + 0.5 * dt, dt)
        _, hist_long = solve("Crank-Nicolson", psi0, v, x, t_long, dx, dt,
                             store_every=int(len(t_long)/80))
        t_snap = t_long[::int(len(t_long)//80)][:len(hist_long)]
        _save_lightweight_1d_gif_three_line(
            cfg.outdir, x, hist_long, t_snap,
            "model4_gif.gif",
            title=f"Double Gaussian Interference (d={d})",
            fps=24
        )
        print("  [GIF4] Double gaussian interference animation")

    print(f"  Grid: n={n}, dt={dt}, steps={len(t)-1}")


# ──────────────────────────────────────────────────────────────
# 实验⑤：模型五 — 谐振子压缩态（呼吸波包）
# ──────────────────────────────────────────────────────────────

def experiment_model5_squeezed(cfg):
    """模型五：谐振子压缩态（呼吸波包），V=½ω²x²，宽度周期振荡。"""
    print("=" * 60)
    print("Model-5: Harmonic Oscillator Squeezed State (Breathing)")
    print("=" * 60)

    n = 2048
    x, dx = grid(-15.0, 15.0, n)
    dt = 0.0005
    t_end = 4.0 * np.pi     # 两个完整呼吸周期（T_breath = π/ω）
    t = np.arange(0.0, t_end + 0.5 * dt, dt)

    omega = 1.0             # 谐振子频率
    r = 0.7                # 压缩参数（r>0: 位置压缩）
    x0 = 1.5               # 初始位移
    p0 = 1.0               # 初始动量

    # 初值：压缩态
    psi0 = exact_squeezed_harmonic(x, 0.0, omega, r, x0, p0)
    psi0 = normalize(psi0, dx)

    # 势场：谐振子
    v = 0.5 * omega**2 * x**2

    print(f"  omega={omega}, r={r}, x0={x0}, p0={p0}")
    print(f"  呼吸周期 T_breath = π/ω = {np.pi/omega:.2f}")
    print(f"  n={n}, dt={dt}, t_end={t_end:.2f} ({t_end/np.pi:.1f}π)")

    results = run_1d_methods_static(METHODS, psi0, v, x, t, dx, dt)

    # 解析解
    psi_exact = exact_squeezed_harmonic(x, t_end, omega, r, x0, p0)
    psi_exact = normalize(psi_exact, dx)

    dpi = get_adaptive_dpi(3)

    # 图5a: 全部 + Exact
    plot_three_panel(
        x, results, METHODS, COLORS, LABELS,
        "Squeezed State (all)", cfg.outdir,
        "model5a_all.png", dpi=dpi, psi_exact=psi_exact,
        show_v=True, v_x=v)
    print("  [Fig5a] All 5 methods + Exact: three-value (+ potential)")

    # 图5b: 稳定方法
    m_stable = METHODS[1:]
    plot_three_panel(
        x, results, m_stable, COLORS[1:], LABELS[1:],
        "Squeezed State (stable)", cfg.outdir,
        "model5b_stable.png", dpi=dpi, psi_exact=psi_exact,
        show_v=True, v_x=v)
    print("  [Fig5b] Stable methods + Exact: three-value")

    # 误差分析
    plot_error_analysis(
        x, results, METHODS, COLORS, psi_exact, dx,
        cfg.outdir, "model5", dpi=dpi)
    print("  [Error] Complete error analysis")

    # 动图
    if results["Crank-Nicolson"]["stable"]:
        print("  Generating animation (CN, squeezed breathing state)...")
        t_long = np.arange(0.0, t_end + 0.5 * dt, dt)
        _, hist_long = solve("Crank-Nicolson", psi0, v, x, t_long, dx, dt,
                             store_every=int(len(t_long)/80))
        t_snap = t_long[::int(len(t_long)//80)][:len(hist_long)]
        _save_lightweight_1d_gif_three_line(
            cfg.outdir, x, hist_long, t_snap,
            "model5_gif.gif",
            title=f"Squeezed Breathing State (r={r})",
            fps=24
        )
        print("  [GIF5] Squeezed breathing state animation")

    print(f"  Grid: n={n}, dt={dt}, steps={len(t)-1}")


# ──────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────

def main():
    outdir = "tdse_experiments_v4"
    os.makedirs(outdir, exist_ok=True)

    cfg = RunConfig(outdir=outdir, quick=False, save_gif=True, dpi=600, grid_size=1024)

    print(f"\n{'═'*62}")
    print(f"  TDSE Numerical Methods — v4 Rigorous Benchmark Models")
    print(f"  Output: {os.path.abspath(outdir)}")
    print(f"{'═'*62}\n")

    start = time.perf_counter()

    experiment_model4_double_gaussian(cfg)   # 模型四：双高斯干涉
    experiment_model5_squeezed(cfg)           # 模型五：压缩态呼吸

    elapsed = time.perf_counter() - start
    print(f"\n{'═'*62}")
    print(f"  All done! Total: {elapsed:.1f}s")
    print(f"  Output: {outdir}/")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
