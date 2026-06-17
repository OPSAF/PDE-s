#!/usr/bin/env python3
"""
1D Time-Dependent Schrödinger Equation — Comprehensive Solver
═══════════════════════════════════════════════════════════════
  i ∂ψ/∂t = -(1/2)∂²ψ/∂x² + V(x)ψ        (ℏ = m = 1)

  Potential  : V(x) = x²/2  (harmonic oscillator)
  IC         : ψ(x,0) = π^(-1/4) exp(-(x-x₀)²/2 + ik₀x)
  Exact soln : coherent state (see exact_psi)

  Methods:  FTCS | Backward Euler | Crank-Nicolson | RK4-MOL | SSFFT
"""

import os, time, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized
warnings.filterwarnings("ignore")

OUT = "v6"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "lines.linewidth": 1.8,
})

# ══════════════════════════════════════════════════════════════
# 1.  GRID, IC, AND TIME PARAMETERS
# ══════════════════════════════════════════════════════════════
L   = 12.0
N   = 512
x   = np.linspace(-L, L, N, endpoint=False)
dx  = x[1] - x[0]

x0, k0 = 2.0, 2.0           # IC parameters

T_final   = 2.0 * np.pi     # one full HO period
dt_main   = 0.001
Nt_main   = int(round(T_final / dt_main))

n_frames   = 120
frame_skip = max(1, Nt_main // n_frames)

print("═" * 62)
print("  1D TDSE  –  Harmonic Oscillator  –  Coherent State")
print("═" * 62)
print(f"  Grid  : N={N}, dx={dx:.5f}, L=±{L}")
print(f"  Time  : T={T_final:.5f}, dt={dt_main}, steps={Nt_main}")
print(f"  IC    : x₀={x0}, k₀={k0}")
print()

# ══════════════════════════════════════════════════════════════
# 2.  EXACT SOLUTION  (coherent state of HO)
# ══════════════════════════════════════════════════════════════
def exact_psi(x, t):
    xc = x0 * np.cos(t) + k0 * np.sin(t)
    pc = k0 * np.cos(t) - x0 * np.sin(t)
    return (np.pi**(-0.25)
            * np.exp(-0.5 * (x - xc)**2)
            * np.exp(1j * pc * x)
            * np.exp(-1j * xc * pc / 2.0)
            * np.exp(-1j * t / 2.0))

psi0 = exact_psi(x, 0.0)
print(f"  Initial norm  : {np.trapezoid(np.abs(psi0)**2, x):.8f}")

# ══════════════════════════════════════════════════════════════
# 3.  POTENTIAL  +  COMPLEX ABSORBING POTENTIAL (CAP)
# ══════════════════════════════════════════════════════════════
V_real = 0.5 * x**2

cap_width    = 2.0
cap_strength = 10.0
cap = np.zeros(N, dtype=float)
xl, xr = -L + cap_width, L - cap_width
cap[x < xl] = cap_strength * ((x[x < xl] - xl) / cap_width)**2
cap[x > xr] = cap_strength * ((x[x > xr] - xr) / cap_width)**2
V_eff = V_real - 1j * cap

# ══════════════════════════════════════════════════════════════
# 4.  ERROR / PROBABILITY METRICS
# ══════════════════════════════════════════════════════════════
l2   = lambda a, b: np.sqrt(np.trapezoid(np.abs(a - b)**2, x))
linf = lambda a, b: np.max(np.abs(a - b))
prob = lambda psi:  np.trapezoid(np.abs(psi)**2, x)

# ══════════════════════════════════════════════════════════════
# 5.  BUILD SPARSE HAMILTONIAN  +  PRE-FACTORED SOLVERS
# ══════════════════════════════════════════════════════════════
def build_hamiltonian(V_eff, dx, N):
    main = (1.0 / dx**2) + V_eff
    off  = -0.5 / dx**2 * np.ones(N - 1)
    return diags([off, main, off], [-1, 0, 1],
                 shape=(N, N), dtype=complex, format="csc")

def apply_dirichlet(M):
    M = M.tolil()
    M[0,:]  = 0;  M[0, 0]   = 1
    M[-1,:] = 0;  M[-1, -1] = 1
    return M.tocsc()

def build_cn_system(H, dt, N):
    I  = eye(N, format="csc", dtype=complex)
    A = apply_dirichlet(I + 0.5j * dt * H)
    B = (I - 0.5j * dt * H).tolil()
    B[0,:] = 0;  B[-1,:] = 0
    return A, B.tocsc()

def build_be_matrix(H, dt, N):
    I = eye(N, format="csc", dtype=complex)
    return apply_dirichlet(I + 1j * dt * H)

H = build_hamiltonian(V_eff, dx, N)

print("  Pre-factorising sparse matrices...", end=" ", flush=True)
t_build = time.time()
A_cn, B_cn = build_cn_system(H, dt_main, N)
solve_cn   = factorized(A_cn)
A_be       = build_be_matrix(H, dt_main, N)
solve_be   = factorized(A_be)
print(f"done ({time.time()-t_build:.2f}s)\n")

# ══════════════════════════════════════════════════════════════
# 6.  METHOD IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════

# ── RHS helper for explicit methods ──────────────────────────
def dpsidt(psi):
    """dψ/dt = -i H ψ  (finite-difference Laplacian)"""
    lap = np.empty_like(psi)
    lap[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2
    lap[0] = lap[-1] = 0.0
    return -1j * (-0.5 * lap + V_eff * psi)

# ── k-grid for SSFFT ─────────────────────────────────────────
k_fft = 2.0 * np.pi * np.fft.fftfreq(N, dx)


def run_ftcs(psi0, dt, n_steps, frame_skip):
    """Forward-Time Centred-Space  (unconditionally unstable)"""
    psi = psi0.copy()
    frames = [psi.copy()]; t_arr = [0.0]; blow_step = None
    for n in range(1, n_steps + 1):
        lap = np.empty_like(psi)
        lap[1:-1] = (psi[2:] - 2*psi[1:-1] + psi[:-2]) / dx**2
        lap[0] = lap[-1] = 0.0
        psi = psi - 1j * dt * (-0.5 * lap + V_eff * psi)
        psi[0] = psi[-1] = 0.0
        if blow_step is None and np.max(np.abs(psi)) > 1e4:
            blow_step = n
        if n % frame_skip == 0:
            frames.append(psi.copy()); t_arr.append(n * dt)
    return frames, t_arr, blow_step


def run_be(psi0, n_steps, frame_skip):
    """Backward Euler  (1st order, unconditionally stable, dissipative)"""
    psi = psi0.copy()
    frames = [psi.copy()]; t_arr = [0.0]
    for n in range(1, n_steps + 1):
        rhs = psi.copy(); rhs[0] = rhs[-1] = 0.0
        psi = solve_be(rhs)
        if n % frame_skip == 0:
            frames.append(psi.copy()); t_arr.append(n * dt_main)
    return frames, t_arr


def run_cn(psi0, n_steps, frame_skip):
    """Crank-Nicolson  (2nd order, unconditionally stable, unitary)"""
    psi = psi0.copy()
    frames = [psi.copy()]; t_arr = [0.0]
    for n in range(1, n_steps + 1):
        rhs = B_cn.dot(psi); rhs[0] = rhs[-1] = 0.0
        psi = solve_cn(rhs)
        if n % frame_skip == 0:
            frames.append(psi.copy()); t_arr.append(n * dt_main)
    return frames, t_arr


def run_rk4(psi0, dt, n_steps, frame_skip):
    """RK4 Method-of-Lines  (4th order, conditionally stable)"""
    psi = psi0.copy()
    frames = [psi.copy()]; t_arr = [0.0]
    for n in range(1, n_steps + 1):
        k1 = dpsidt(psi)
        k2 = dpsidt(psi + 0.5*dt*k1)
        k3 = dpsidt(psi + 0.5*dt*k2)
        k4 = dpsidt(psi + dt*k3)
        psi = psi + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
        psi[0] = psi[-1] = 0.0
        if n % frame_skip == 0:
            frames.append(psi.copy()); t_arr.append(n * dt)
    return frames, t_arr


def run_ssfft(psi0, dt, n_steps, frame_skip):
    """Split-Step Fourier  (Strang: V/2 → K → V/2, 2nd order)"""
    psi   = psi0.copy()
    phV   = np.exp(-1j * V_eff * 0.5 * dt)
    phK   = np.exp(-1j * 0.5  * k_fft**2 * dt)
    frames = [psi.copy()]; t_arr = [0.0]
    for n in range(1, n_steps + 1):
        psi = phV * psi
        psi = np.fft.ifft(phK * np.fft.fft(psi))
        psi = phV * psi
        if n % frame_skip == 0:
            frames.append(psi.copy()); t_arr.append(n * dt)
    return frames, t_arr

# ══════════════════════════════════════════════════════════════
# 7.  RUN ALL METHODS
# ══════════════════════════════════════════════════════════════
METHOD_NAMES = ["FTCS", "BE", "CN", "RK4", "SSFFT"]
COLORS = {"FTCS":"#e84855", "BE":"#f77f00", "CN":"#2176ae",
          "RK4":"#06a77d", "SSFFT":"#9b2dca", "Exact":"#111111"}
STYLES = {"FTCS":"--", "BE":"-.", "CN":"-", "RK4":":", "SSFFT":"-", "Exact":"-"}

data = {}
for name in METHOD_NAMES:
    print(f"  Running {name:6s} …", end=" ", flush=True)
    t0 = time.time()
    blow = None
    if   name == "FTCS":  fs, ts, blow = run_ftcs(psi0, dt_main, Nt_main, frame_skip)
    elif name == "BE":    fs, ts        = run_be  (psi0, Nt_main, frame_skip)
    elif name == "CN":    fs, ts        = run_cn  (psi0, Nt_main, frame_skip)
    elif name == "RK4":   fs, ts        = run_rk4 (psi0, dt_main, Nt_main, frame_skip)
    elif name == "SSFFT": fs, ts        = run_ssfft(psi0, dt_main, Nt_main, frame_skip)
    rt = time.time() - t0
    data[name] = dict(frames=fs, times=ts, runtime=rt, blow=blow)
    info = f"done {rt:.2f}s"
    if blow: info += f"  ← blew up at step {blow} (t={blow*dt_main:.3f})"
    print(info)

print()

# Exact frames aligned with CN time grid
t_grid      = np.array(data["CN"]["times"])
exact_frames = [exact_psi(x, t) for t in t_grid]

# ══════════════════════════════════════════════════════════════
# 8.  COMPUTE ERRORS  &  PROBABILITY CONSERVATION
# ══════════════════════════════════════════════════════════════
P0 = prob(psi0)
for name in METHOD_NAMES:
    ts = data[name]["times"]
    fs = data[name]["frames"]
    nf = min(len(fs), len(exact_frames))
    l2e, lie, pe = [], [], []
    for i in range(nf):
        pn = fs[i]
        pe_ref = exact_psi(x, ts[i])
        amp = np.max(np.abs(pn))
        if np.isnan(amp) or amp > 1e8:
            l2e.append(np.nan); lie.append(np.nan); pe.append(np.nan)
        else:
            l2e.append(l2(pn, pe_ref))
            lie.append(linf(pn, pe_ref))
            pe.append(prob(pn))
    data[name]["l2"]   = np.array(l2e)
    data[name]["linf"] = np.array(lie)
    data[name]["prob"] = np.array(pe)

# ══════════════════════════════════════════════════════════════
# 9.  CONVERGENCE STUDY  (vary dt, measure L2 at T = π/2)
# ══════════════════════════════════════════════════════════════
print("  Convergence study …")
dt_list  = [0.020, 0.010, 0.005, 0.002, 0.001, 0.0005]
T_conv   = np.pi / 2.0
ref_psi  = exact_psi(x, T_conv)
conv_m   = ["BE", "CN", "RK4", "SSFFT"]
conv_err = {m: [] for m in conv_m}

for dt_c in dt_list:
    Nt_c = int(round(T_conv / dt_c))
    for m in conv_m:
        psi = psi0.copy()
        if m == "BE":
            A_c = build_be_matrix(build_hamiltonian(V_eff, dx, N), dt_c, N)
            sv  = factorized(A_c)
            for _ in range(Nt_c):
                r = psi.copy(); r[0]=r[-1]=0; psi = sv(r)
        elif m == "CN":
            Ac, Bc = build_cn_system(build_hamiltonian(V_eff, dx, N), dt_c, N)
            sv = factorized(Ac)
            for _ in range(Nt_c):
                r = Bc.dot(psi); r[0]=r[-1]=0; psi = sv(r)
        elif m == "RK4":
            for _ in range(Nt_c):
                k1=dpsidt(psi); k2=dpsidt(psi+0.5*dt_c*k1)
                k3=dpsidt(psi+0.5*dt_c*k2); k4=dpsidt(psi+dt_c*k3)
                psi += dt_c/6*(k1+2*k2+2*k3+k4); psi[0]=psi[-1]=0
        elif m == "SSFFT":
            pV = np.exp(-1j*V_eff*0.5*dt_c); pK = np.exp(-1j*0.5*k_fft**2*dt_c)
            for _ in range(Nt_c):
                psi = pV*psi; psi = np.fft.ifft(pK*np.fft.fft(psi)); psi = pV*psi
        conv_err[m].append(l2(psi, ref_psi))
    print(f"    dt={dt_c:.4f}  " +
          "  ".join(f"{m}={conv_err[m][-1]:.2e}" for m in conv_m))

print()

# ══════════════════════════════════════════════════════════════
# 10.  GIF ANIMATIONS  (Re ψ | Im ψ | |ψ|²)
# ══════════════════════════════════════════════════════════════
print("  Generating animations …")
ANIM_METHODS = ["CN", "SSFFT"]
ANIM_LABELS  = {"CN":"Crank-Nicolson", "SSFFT":"Split-Step FFT", "Exact":"Exact"}
nf_anim = min(len(data["CN"]["frames"]), len(exact_frames))

def make_gif(qty_key, ylabel, title_str, filename):
    """qty_key: 'real' | 'imag' | 'dens'"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(-L + cap_width - 0.3, L - cap_width + 0.3)
    ax.set_xlabel("x"); ax.set_ylabel(ylabel)

    # y-limits: estimate from initial frame
    if qty_key == "real":
        fn = np.real; yl = (-1.1, 1.1)
    elif qty_key == "imag":
        fn = np.imag; yl = (-1.1, 1.1)
    else:
        fn = lambda p: np.abs(p)**2; yl = (-0.05, 0.75)
    ax.set_ylim(*yl)

    # potential background
    Vn = V_real / V_real.max() * yl[1] * 0.85
    ax.fill_between(x, yl[0], Vn, alpha=0.07, color="gray")
    ax.plot(x, Vn, color="gray", lw=0.8, ls="--", alpha=0.5)

    lines = {}
    for m in ANIM_METHODS:
        col = COLORS[m]; ls = STYLES[m]
        l, = ax.plot([], [], color=col, ls=ls, lw=2.0,
                     label=ANIM_LABELS[m], zorder=3)
        lines[m] = l
    lex, = ax.plot([], [], color=COLORS["Exact"], lw=1.6, ls="-",
                   label="Exact", zorder=5)
    lines["Exact"] = lex

    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    time_txt = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                       fontsize=10, va="top")
    ax.set_title(title_str)
    fig.tight_layout()

    def init():
        for l in lines.values(): l.set_data([], [])
        time_txt.set_text("")
        return list(lines.values()) + [time_txt]

    def update(i):
        t = t_grid[i]
        for m in ANIM_METHODS:
            if i < len(data[m]["frames"]):
                yv = fn(data[m]["frames"][i])
                if np.nanmax(np.abs(yv)) < 1e6:
                    lines[m].set_data(x, yv)
        lines["Exact"].set_data(x, fn(exact_frames[i]))
        time_txt.set_text(f"t = {t:.3f}   (t/T = {t/T_final:.3f})")
        return list(lines.values()) + [time_txt]

    ani = animation.FuncAnimation(fig, update, frames=nf_anim,
                                  init_func=init, interval=60, blit=True)
    writer = animation.PillowWriter(fps=18)
    path = os.path.join(OUT, filename)
    ani.save(path, writer=writer)
    plt.close(fig)
    print(f"    Saved {filename}")

make_gif("real", "Re(ψ)",   "Real part  Re(ψ)",      "tdse_real.gif")
make_gif("imag", "Im(ψ)",   "Imaginary part  Im(ψ)", "tdse_imag.gif")
make_gif("dens", "|ψ|²",    "Probability density |ψ|²", "tdse_density.gif")

# ══════════════════════════════════════════════════════════════
# 11.  ERROR CURVES OVER TIME
# ══════════════════════════════════════════════════════════════
print("  Generating error curves …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax_l2, ax_li = axes

for name in METHOD_NAMES:
    ts  = np.array(data[name]["times"])
    l2v = data[name]["l2"]
    liv = data[name]["linf"]
    nf  = min(len(ts), len(l2v))
    kw  = dict(color=COLORS[name], ls=STYLES[name], label=name, lw=1.8)
    # clip blow-up for visibility
    l2c = np.where(l2v[:nf] > 10, np.nan, l2v[:nf])
    lic = np.where(liv[:nf] > 10, np.nan, liv[:nf])
    ax_l2.semilogy(ts[:nf], l2c, **kw)
    ax_li.semilogy(ts[:nf], lic, **kw)

for ax, title in zip(axes, ["L₂ Error  vs  Time", "L∞ Error  vs  Time"]):
    ax.set_xlabel("t"); ax.set_ylabel("Error")
    ax.set_title(title); ax.legend(fontsize=9)
    ax.set_xlim(0, T_final)
    ax.axvline(np.pi, color="gray", ls=":", lw=1, alpha=0.6, label="t=π")
    ax.axvline(2*np.pi, color="gray", ls="--", lw=1, alpha=0.6)
    ax.grid(True, which="both", alpha=0.3)

fig.suptitle("Error over Time — All Methods", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tdse_errors.png"), bbox_inches="tight")
plt.close(fig)
print("    Saved tdse_errors.png")

# ══════════════════════════════════════════════════════════════
# 12.  PROBABILITY CONSERVATION
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4.5))
for name in METHOD_NAMES:
    ts = np.array(data[name]["times"])
    pv = data[name]["prob"]
    nf = min(len(ts), len(pv))
    pv_c = np.where(np.abs(pv[:nf]) > 10, np.nan, pv[:nf])
    ax.plot(ts[:nf], pv_c,
            color=COLORS[name], ls=STYLES[name], label=name, lw=1.8)
ax.axhline(P0, color="k", ls="--", lw=1.2, alpha=0.5, label=f"P₀={P0:.4f}")
ax.set_xlabel("t"); ax.set_ylabel("∫|ψ|² dx")
ax.set_title("Probability Conservation", fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_xlim(0, T_final)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tdse_probability.png"), bbox_inches="tight")
plt.close(fig)
print("    Saved tdse_probability.png")

# ══════════════════════════════════════════════════════════════
# 13.  CONVERGENCE PLOT
# ══════════════════════════════════════════════════════════════
print("  Generating convergence plot …")
conv_colors = {"BE":"#f77f00","CN":"#2176ae","RK4":"#06a77d","SSFFT":"#9b2dca"}
orders      = {"BE":1, "CN":2, "RK4":4, "SSFFT":2}

fig, ax = plt.subplots(figsize=(8, 5.5))
dt_arr = np.array(dt_list)

for m in conv_m:
    err_arr = np.array(conv_err[m])
    mask = np.isfinite(err_arr) & (err_arr > 0)
    ax.loglog(dt_arr[mask], err_arr[mask],
              "o-", color=conv_colors[m], label=m, lw=2, ms=6)

# Reference lines
ref_x = np.array([dt_list[1], dt_list[-1]])
for p, ls, col in [(1,"--","#aaa"), (2,"-","#888"), (4,":","#666")]:
    ax.loglog(ref_x, ref_x[0]**p / ref_x**p * conv_err["CN"][1],
              ls=ls, color=col, lw=1.2, alpha=0.6,
              label=f"O(Δt^{p})")

ax.set_xlabel("Time step  Δt"); ax.set_ylabel("L₂ Error at t = π/2")
ax.set_title("Convergence Study  —  L₂ Error vs Δt", fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tdse_convergence.png"), bbox_inches="tight")
plt.close(fig)
print("    Saved tdse_convergence.png")

# ══════════════════════════════════════════════════════════════
# 14.  FINAL SNAPSHOT — side-by-side comparison at t = π
# ══════════════════════════════════════════════════════════════
print("  Generating snapshot comparison …")
t_snap = np.pi
# find closest frame
idx_snap = np.argmin(np.abs(t_grid - t_snap))
t_snap_actual = t_grid[idx_snap]

fig = plt.figure(figsize=(14, 9))
gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

quants = [
    ("Re(ψ)",  lambda p: np.real(p),        (-1.1, 1.1)),
    ("Im(ψ)",  lambda p: np.imag(p),        (-1.1, 1.1)),
    ("|ψ|²",   lambda p: np.abs(p)**2,      (-0.05, 0.75)),
]

# top row: Re, Im, |ψ|²  — overlay
for col, (title, fn, yl) in enumerate(quants):
    ax = fig.add_subplot(gs[0, col])
    Vn = V_real / V_real.max() * yl[1] * 0.85
    ax.fill_between(x, yl[0], Vn, alpha=0.07, color="gray")
    ax.plot(x, fn(exact_psi(x, t_snap_actual)),
            color=COLORS["Exact"], lw=2.2, label="Exact", zorder=6)
    for nm in ["CN", "SSFFT", "BE", "RK4"]:
        if idx_snap < len(data[nm]["frames"]):
            yv = fn(data[nm]["frames"][idx_snap])
            if np.nanmax(np.abs(yv)) < 1e6:
                ax.plot(x, yv, color=COLORS[nm], ls=STYLES[nm],
                        lw=1.6, alpha=0.85, label=nm)
    ax.set_xlim(-8, 8); ax.set_ylim(*yl)
    ax.set_title(f"{title}  (t = {t_snap_actual:.3f} ≈ π)")
    ax.set_xlabel("x"); ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(True, alpha=0.25)

# bottom row: |error| for each stable method
for col, nm in enumerate(["BE","CN","SSFFT"]):
    ax = fig.add_subplot(gs[1, col])
    if idx_snap < len(data[nm]["frames"]):
        err_field = np.abs(data[nm]["frames"][idx_snap] - exact_psi(x, t_snap_actual))
        ax.semilogy(x, err_field, color=COLORS[nm], lw=1.8)
    ax.set_xlim(-8, 8); ax.set_xlabel("x")
    ax.set_ylabel("|ψ_num − ψ_exact|")
    ax.set_title(f"Pointwise error — {nm}")
    ax.grid(True, which="both", alpha=0.25)

fig.suptitle("Snapshot Comparison at t = π  (half period)", fontsize=13, fontweight="bold")
fig.savefig(os.path.join(OUT, "tdse_snapshot.png"), bbox_inches="tight", dpi=130)
plt.close(fig)
print("    Saved tdse_snapshot.png")

# ══════════════════════════════════════════════════════════════
# 15.  SUMMARY TABLE (figure)
# ══════════════════════════════════════════════════════════════
print("  Generating summary table …")

# Collect final metrics
rows = []
for nm in METHOD_NAMES:
    l2v   = data[nm]["l2"]
    liv   = data[nm]["linf"]
    pv    = data[nm]["prob"]
    rt    = data[nm]["runtime"]
    blow  = data[nm]["blow"]

    # find last valid index
    valid = ~np.isnan(l2v)
    if valid.any():
        idx_last = np.where(valid)[0][-1]
        l2_fin  = l2v[idx_last]
        li_fin  = liv[idx_last]
        pr_fin  = pv[idx_last]
        t_fin   = data[nm]["times"][idx_last]
        dP      = abs(pr_fin - P0) / P0
    else:
        l2_fin = li_fin = dP = np.nan; t_fin = 0

    stability = "✗ unstable" if nm == "FTCS" else "✓ stable"
    if blow:
        blow_str = f"t≈{blow*dt_main:.3f}"
    else:
        blow_str = "—"

    rows.append([nm,
                 f"{l2_fin:.3e}"  if not np.isnan(l2_fin) else "diverged",
                 f"{li_fin:.3e}"  if not np.isnan(li_fin) else "diverged",
                 f"{dP:.3e}"      if not np.isnan(dP) else "—",
                 f"{rt:.2f} s",
                 blow_str,
                 stability])

col_labels = ["Method", "L₂ Error", "L∞ Error",
              "ΔP/P₀", "Runtime", "Blow-up", "Stability"]

fig, ax = plt.subplots(figsize=(13, 3.4))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
tbl.scale(1.0, 2.1)

# header style
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor("#2b3a55")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold")

# row colours
row_bg = {"FTCS":"#fde8e8","BE":"#fff3e0","CN":"#e3f2fd",
          "RK4":"#e8f5e9","SSFFT":"#f3e5f5"}
for i, row in enumerate(rows):
    nm = row[0]
    for j in range(len(col_labels)):
        tbl[(i+1, j)].set_facecolor(row_bg.get(nm,"#fafafa"))

fig.suptitle("Summary Table — 1D TDSE Methods", fontsize=13, fontweight="bold", y=0.97)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "tdse_summary_table.png"),
            bbox_inches="tight", dpi=130)
plt.close(fig)
print("    Saved tdse_summary_table.png")

# ══════════════════════════════════════════════════════════════
# 16.  PRINT SUMMARY TO CONSOLE
# ══════════════════════════════════════════════════════════════
print()
print("═" * 62)
print("  FINAL SUMMARY")
print("═" * 62)
hdr = f"{'Method':7s}  {'L2 Error':>12s}  {'Linf Error':>12s}  {'ΔP/P₀':>10s}  {'Runtime':>9s}"
print(hdr); print("─" * 62)
for row in rows:
    nm, l2s, lis, dps, rts, bls, stab = row
    print(f"{nm:7s}  {l2s:>12s}  {lis:>12s}  {dps:>10s}  {rts:>9s}  {stab}")
print()

print("  All outputs saved to", OUT)
print("  Files:")
for f in sorted(os.listdir(OUT)):
    fp = os.path.join(OUT, f)
    kb = os.path.getsize(fp) / 1024
    print(f"    {f:35s}  {kb:7.1f} KB")