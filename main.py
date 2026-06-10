"""
main.py — 电子在典型一维/二维势场中的量子动力学：波包演化、隧穿与散射
=====================================================================

基于含时薛定谔方程（TDSE），模拟电子在自由空间、势垒、无限深势阱、
二维圆形障碍物、波导等场景中的波包演化、隧穿、散射和约束传播。

数值方法
--------
一维（5种）:
    FTCS（显式向前Euler）          — 无条件不稳定，反面教材
    Backward Euler（隐式）          — 无条件 L₂ 稳定，有耗散
    Crank-Nicolson（隐式，二阶，酉） — 主力方法，动图使用
    Split-Step Fourier（SSFM，谱）  — 酉，扩展方法
    RK4（4阶显式Runge-Kutta）       — 条件稳定

二维（2种）:
    ADI（交替方向隐式）             — 课本第4章方法推广，已实现
    Split-Step FFT（SSFM）          — 精度基准

实验输出
--------
    Figure 1:  解析解 vs 数值方法对比（|ψ|² + Re[ψ] + Im[ψ] 三值视图）
    Figure 2:  误差收敛性分析（log-log 收敛阶）
    Figure 3:  稳定性扫描（各方法稳定区域热力图）
    Figure 4:  性能对比表
    Figure 5:  自由波包传播动画（三面板：|ψ|² + Re + Im + exact）
    Figure 6:  量子隧穿（三种能量 + T/R 系数 + Re/Im 近场）
    Figure 7:  二维自由高斯传播 + 质量守恒 + 相位角
    Figure 8:  二维圆形障碍物散射 + 角分布分析
    Figure 9:  二维波导约束 vs 自由传播
    补充实验:  质量守恒分析、运行时对比、RTD双势垒共振隧穿、
              2D误差热力图、2D收敛性、ADI vs SSFM对比、
              圆形障碍物半径扫描、波导强度扫描、全方法综合对比

使用方法
--------
    python main.py                    # 完整运行（一维 + 二维）
    python main.py --quick            # 快速模式（低分辨率）
    python main.py --no-gif           # 不生成动画
    python main.py --exp tunneling    # 只运行指定实验
    python main.py --dim 1 --quick    # 快速模式，只跑一维
    python main.py --dim 2            # 只跑二维实验
    python main.py --dpi 150          # 低分辨率输出
    python main.py --list             # 列出所有可用实验
    python main.py --all              # 包括 PINN 实验的全部运行

依赖
----
    numpy scipy matplotlib pandas tqdm
    可选: torch（用于 PINN 实验）
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── TDSE package imports ────────────────────────────────────────────────────
from tdse.config import RunConfig, ensure_outdir, setup_plot_style
from tdse.potentials import print_section

# ── Experiment imports (1D core) ────────────────────────────────────────────
from tdse.experiments import (
    experiment_analytic_vs_numerical,
    experiment_convergence,
    experiment_stability,
    experiment_performance,
    experiment_method_comparison,
    experiment_tunneling,
    experiment_rtd_double_barrier,
    experiment_all_methods_comparison,
    experiment_1d_conservation_analysis,
    experiment_runtime_comparison,
    save_wavepacket_animation_experiment,
)

# ── Experiment imports (2D) ─────────────────────────────────────────────────
from tdse.experiments import (
    experiment_2d_free_propagation,
    experiment_2d_circular_obstacle_with_animation,
    experiment_2d_waveguide,
    experiment_2d_error_heatmap,
    experiment_2d_convergence,
    experiment_circular_obstacle_radius_sweep,
    experiment_waveguide_strength_sweep,
    experiment_2d_adi_vs_ssf,
)  # end of 2D imports


# =============================================================================
# Experiment registry
# =============================================================================

EXPERIMENTS: dict[str, tuple[str, str, callable]] = {
    # (key, category, description, function)
    # ── 1D 核心实验 ──
    "analytic":     ("1D", "Fig 1: Analytic vs Numerical (|ψ|² + Re + Im 三值视图)",
                     experiment_analytic_vs_numerical),
    "convergence":  ("1D", "Fig 2: Error Convergence Study (log-log 收敛阶拟合)",
                     experiment_convergence),
    "stability":    ("1D", "Fig 3: Stability Map (5方法 × 多组 (N,Δt) 热力图)",
                     experiment_stability),
    "performance":  ("1D", "Fig 4: Performance Table (runtime vs grid size)",
                     experiment_performance),
    "animation":    ("1D", "Fig 5: Wavepacket Animation (三面板: |ψ|² + Re + Im)",
                     save_wavepacket_animation_experiment),
    "tunneling":    ("1D", "Fig 6: Quantum Tunneling (E<V₀, E~V₀, E>V₀ + Re/Im)",
                     experiment_tunneling),
    "method_comp":  ("1D", "Method Comparison (harmonic oscillator, 三值视图)",
                     experiment_method_comparison),
    "rtd":          ("1D", "RTD Double-Barrier Resonant Tunneling",
                     experiment_rtd_double_barrier),

    # ── 1D 分析实验 ──
    "conservation": ("Analysis", "Mass Conservation: Error vs Time (FTCS/CN/SSF)",
                     experiment_1d_conservation_analysis),
    "runtime":      ("Analysis", "Runtime Comparison Table",
                     experiment_runtime_comparison),
    "all_methods":  ("Analysis", "All-Methods Comprehensive Comparison (含 Re/Im 分解)",
                     experiment_all_methods_comparison),

    # ── 2D 核心实验 ──
    "2d_free":       ("2D", "Fig 7: 2D Free Gaussian Propagation + Mass + Phase",
                      experiment_2d_free_propagation),
    "2d_obstacle":   ("2D", "Fig 8: 2D Circular Obstacle Scattering + Angular Distribution",
                      experiment_2d_circular_obstacle_with_animation),
    "2d_waveguide":  ("2D", "Fig 9: 2D Waveguide vs Free Propagation + Beam Profile",
                      experiment_2d_waveguide),

    # ── 2D 分析与对比 ──
    "2d_adi_vs_ssf": ("2D", "2D ADI vs Split-Step FFT — Direct Comparison",
                      experiment_2d_adi_vs_ssf),
    "2d_error":      ("2D", "2D Error Heatmap (Re/Im decomposition vs exact)",
                      experiment_2d_error_heatmap),
    "2d_convergence":("2D", "2D Convergence Study (L1/L2/Linf vs grid size)",
                      experiment_2d_convergence),

    # ── 参数扫描 ──
    "radius_sweep":   ("Analysis", "Param Sweep: Circular Obstacle Radius",
                       experiment_circular_obstacle_radius_sweep),
    "waveguide_sweep":("Analysis", "Param Sweep: Waveguide Strength α",
                       experiment_waveguide_strength_sweep),
}


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TDSE 量子动力学模拟 — 波包演化、隧穿与散射",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py                          # 完整运行（一维 + 二维）
    python main.py --quick                  # 快速测试模式
    python main.py --exp tunneling          # 只运行隧穿实验
    python main.py --exp 2d_obstacle        # 只运行二维障碍物散射
    python main.py --dim 1 --quick          # 快速模式，只跑一维
    python main.py --dim 2 --no-gif         # 二维实验，不生成 GIF
    python main.py --list                   # 列出所有可用实验
        """,
    )
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（降低分辨率，减少计算量）")
    parser.add_argument("--no-gif", action="store_true",
                        help="跳过动画生成")
    parser.add_argument("--dpi", type=int, default=600,
                        help="输出图像 DPI（默认 600，出版物级别）")
    parser.add_argument("--exp", type=str, default=None,
                        help="只运行指定实验（名称见 --list）")
    parser.add_argument("--dim", type=str, default=None,
                        choices=["1", "2", "analysis"],
                        help="只运行指定维度的实验（1 / 2 / analysis）")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用实验并退出")
    parser.add_argument("--outdir", type=str, default="tdse_outputs",
                        help="输出目录（默认 tdse_outputs）")
    parser.add_argument("--grid-size", type=int, default=None,
                        help="网格点数（覆盖默认值）")
    parser.add_argument("--all", action="store_true",
                        help="运行包括 PINN 在内的全部实验")
    return parser


def list_experiments() -> None:
    print("\n可用实验列表:")
    print("=" * 80)
    by_cat: dict[str, list[tuple[str, str]]] = {}
    for key, (cat, desc, _func) in EXPERIMENTS.items():
        by_cat.setdefault(cat, []).append((key, desc))
    for cat in ["1D", "2D", "Analysis", "PINN"]:
        if cat in by_cat:
            print(f"\n── {cat} 实验 ──")
            for key, desc in by_cat[cat]:
                print(f"  {key:<20s} {desc}")


# =============================================================================
# Main runner
# =============================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    # ── Configuration ────────────────────────────────────────────────────────
    cfg = RunConfig(
        outdir=args.outdir,
        quick=args.quick,
        save_gif=not args.no_gif,
        dpi=args.dpi,
    )
    if args.grid_size is not None:
        cfg.grid_size = args.grid_size

    ensure_outdir(cfg.outdir)
    setup_plot_style(cfg.dpi)

    # ── Header ───────────────────────────────────────────────────────────────
    banner = r"""
  ╔══════════════════════════════════════════════════════════════════════╗
  ║     电子在典型一维/二维势场中的量子动力学                             ║
  ║     波包演化 · 隧穿 · 散射 · 约束传播                                ║
  ║                                                                      ║
  ║     基于含时薛定谔方程 (TDSE) 的数值模拟                             ║
  ║     i ∂ψ/∂t = −½ ∂²ψ/∂x² + V(x) ψ                                  ║
  ╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"  输出目录:     {os.path.abspath(cfg.outdir)}")
    print(f"  快速模式:     {'是' if cfg.quick else '否'}")
    print(f"  生成 GIF:     {'是' if cfg.save_gif else '否'}")
    print(f"  图像 DPI:     {cfg.dpi}")
    print(f"  网格点数:     {cfg.grid_size}")
    print()

    t_total_start = time.perf_counter()

    # ── Collect experiments to run ───────────────────────────────────────────
    to_run: list[tuple[str, callable]] = []

    if args.exp:
        found = False
        for key, (_cat, _desc, func) in EXPERIMENTS.items():
            if key == args.exp:
                to_run.append((key, func))
                found = True
                break
        if not found:
            print(f"错误: 未找到实验 '{args.exp}'。使用 --list 查看可用实验。")
            sys.exit(1)
    else:
        dim_filter = args.dim
        pin = args.all
        for key, (cat, _desc, func) in EXPERIMENTS.items():
            if cat == "PINN" and not pin:
                continue
            if dim_filter is None:
                to_run.append((key, func))
            elif dim_filter == "1" and cat in ("1D",):
                to_run.append((key, func))
            elif dim_filter == "2" and cat in ("2D",):
                to_run.append((key, func))
            elif dim_filter == "analysis" and cat in ("Analysis",):
                to_run.append((key, func))

    # ── Run ──────────────────────────────────────────────────────────────────
    n_total = len(to_run)
    success_count = 0
    fail_list: list[str] = []

    for idx, (exp_key, exp_func) in enumerate(to_run, 1):
        # 找到描述
        desc = exp_key
        for key, (_cat, d, _func) in EXPERIMENTS.items():
            if key == exp_key:
                desc = d
                break

        print(f"\n{'='*78}")
        print(f"  [{idx}/{n_total}] {desc}")
        print(f"{'='*78}")

        try:
            result = exp_func(cfg)
            success_count += 1
            # 如果返回 DataFrame，打印简要信息
            import pandas as pd
            if isinstance(result, pd.DataFrame) and len(result) > 0:
                pass  # experiment already printed their own output
        except Exception as e:
            print(f"\n  ❌ 实验失败: {e}")
            import traceback
            traceback.print_exc()
            fail_list.append(exp_key)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total_start
    print("\n\n" + "=" * 78)
    print("  运行总结")
    print("=" * 78)
    print(f"  总实验数:     {n_total}")
    print(f"  成功:         {success_count}")
    print(f"  失败:         {len(fail_list)}")
    if fail_list:
        print(f"  失败实验:     {', '.join(fail_list)}")
    print(f"  总运行时间:   {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"  输出目录:     {os.path.abspath(cfg.outdir)}")
    print(f"\n  生成文件:")
    for name in sorted(os.listdir(cfg.outdir)):
        fpath = os.path.join(cfg.outdir, name)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {name}  ({size_kb:.1f} KB)")
    print("=" * 78)


if __name__ == "__main__":
    main()
