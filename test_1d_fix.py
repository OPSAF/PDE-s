#!/usr/bin/env python3
"""Quick test: only run 1D infinite well experiment to verify fig1d fix."""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tdse.potentials import (
    potential_infinite_well_mask,
    gaussian_wavepacket, normalize, grid,
    l1_l2_linf_error, l2_error_real_imag
)
from tdse.solvers import solve

outdir = "tdse_experiments_v2"
os.makedirs(outdir, exist_ok=True)

# ── 复制实验参数 ──
n = 2048
well_left, well_right = -10.0, 10.0
L = well_right - well_left
x, dx = grid(-15.0, 15.0, n)
dt = 0.001
t_end = 8.0
t = np.arange(0.0, t_end + 0.5 * dt, dt)

eigen_n = 3
E_n = (eigen_n * np.pi)**2 / (2.0 * L**2)

phi_n = np.zeros_like(x, dtype=complex)
mask_well = (x >= well_left) & (x <= well_right)
phi_n[mask_well] = np.sqrt(2.0 / L) * np.sin(eigen_n * np.pi * (x[mask_well] - well_left) / L)
phi_n = normalize(phi_n, dx)
psi0 = phi_n.copy()

v = potential_infinite_well_mask(x, a=well_left, b=well_right)

methods_all = ["FTCS", "Backward-Euler", "Crank-Nicolson", "RK4", "Split-Step-FFT"]
colors_all = ["#9B9B9B", "#2E86AB", "#1B998B", "#F18F01", "#C73E1D"]

print("Running 1D methods...")
results = {}
for method in methods_all:
    try:
        t_hist, hist = solve(method, psi0, v, x, t, dx, dt)
        psi_final = hist[-1]
        mass = float(np.sum(np.abs(psi_final)**2) * dx)
        results[method] = {"stable": True, "psi": psi_final, "mass": mass}
        print(f"  {method}: M={mass:.6f}")
    except Exception as e:
        print(f"  {method}: FAILED ({e})")
        results[method] = {"stable": False, "psi": None, "mass": np.nan}

psi_exact = phi_n * np.exp(-1j * E_n * t_end)

# ── 只画 fig1d 误差分析图 ──
print("\nGenerating fig1d error analysis...")

err_rows = []
for method in methods_all:
    r = results[method]
    row = {"Method": method}
    if r["stable"] and r["psi"] is not None:
        l1, l2, linf = l1_l2_linf_error(r["psi"], psi_exact, dx)
        l2_re, l2_im, _ = l2_error_real_imag(r["psi"], psi_exact, dx)
        ref_norm = float(np.sqrt(np.sum(np.abs(psi_exact)**2) * dx))
        rel_l2 = l2 / ref_norm if ref_norm > 0 else np.nan
        row.update({"Mass": r["mass"], "L1": l1, "L2": l2, "Linf": linf,
                    "Rel_L2": rel_l2, "L2_Re": l2_re, "L2_Im": l2_im})
    else:
        row.update({"Mass": np.nan, "L1": np.nan, "L2": np.nan,
                    "Linf": np.nan, "Rel_L2": np.nan, "L2_Re": np.nan, "L2_Im": np.nan})
    err_rows.append(row)

df_err = pd.DataFrame(err_rows)

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
metrics = [
    ("L1 Error", "L1", True),
    ("L2 Error", "L2", True),
    ("Linf Error", "Linf", True),
    ("Relative L2", "Rel_L2", True),
    ("L2 Real Part", "L2_Re", True),
    ("L2 Imag Part", "L2_Im", True),
]

valid_methods = [m for m in methods_all if np.isfinite(df_err.loc[df_err["Method"]==m, "L2"].values[0])]
bar_colors_dict = dict(zip(methods_all, colors_all))

print(f"  Valid methods: {valid_methods}")

for idx, (title, col, use_log) in enumerate(metrics):
    ax = axes[idx // 3][idx % 3]
    vals = [df_err.loc[df_err["Method"]==m, col].values[0] for m in valid_methods]
    clrs = [bar_colors_dict[m] for m in valid_methods]
    bars = ax.bar(range(len(valid_methods)), vals, color=clrs, edgecolor='white', lw=0.5)
    if use_log:
        vals_finite = [v for v in vals if np.isfinite(v) and v > 0]
        if vals_finite:
            ax.set_yscale('log')
            v_min = min(vals_finite) * 0.5
            v_max = max(vals_finite) * 5.0
            ax.set_ylim(v_min, v_max)
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

fig.suptitle("Infinite Well - Complete Error Analysis vs Analytic Solution",
             fontsize=13, y=1.01)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out_path = os.path.join(outdir, "fig1d_inf_well_error_analysis.png")
fig.savefig(out_path, dpi=1000, bbox_inches='tight')
plt.close(fig)

print(f"\nSaved: {out_path}")
print(df_err.round(8).to_string(index=False))
print("\nDone!")
