"""
Case 5: Driven Harmonic Oscillator V(x)=x^2/2 + F*x

Implements displaced coherent-state analytic solution and compares with CN numerical solver.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
import imageio.v2 as imageio
from PIL import Image
import os
import matplotlib as mpl
mpl.rcParams['font.size'] = 12
plt.style.use('seaborn-v0_8-darkgrid')

# Parameters (coherent initial state)
F = 0.5
x0 = -2.0
k0 = 1.0
sigma = 1.0/np.sqrt(2.0)  # choose ground-state width for HO -> coherent state

# Spatial grid
Nx = 2048
L = 100.0
x = np.linspace(-L/2, L/2, Nx)
dx = x[1]-x[0]

# Temporal grid
T = 12.0
dt = 0.01
Nt = int(T/dt)
times = np.linspace(0, T, Nt+1)

def psi0(x):
    norm = (1/(2*np.pi*sigma**2))**0.25
    return norm * np.exp(- (x - x0)**2 /(4*sigma**2) + 1j*k0*x)

def classical_trajectory(t):
    # equation: x'' + x = -F  (m=1)
    # general solution: x = -F + (x0 + F) cos t + k0 sin t
    xc = -F + (x0 + F) * np.cos(t) + k0 * np.sin(t)
    pc = -(x0 + F) * np.sin(t) + k0 * np.cos(t)
    return xc, pc

def action_along_trajectory(tvals):
    # compute classical action S(t) = integral (1/2 p^2 - V(x)) dt along trajectory
    xs, ps = classical_trajectory(tvals)
    V = 0.5 * xs**2 + F * xs
    L = 0.5 * ps**2 - V
    # cumulative integral
    S = np.cumsum(L) * (tvals[1]-tvals[0])
    return S

def psi_analytic_coherent(x_arr, t, S_t=None):
    # displaced ground-state (coherent state) centered at classical xc(t) with momentum pc(t)
    xc, pc = classical_trajectory(t)
    # ground state width
    sigma0 = sigma
    pref = (1/(2*np.pi*sigma0**2))**0.25 * np.exp(1j*(pc*xc/2.0))
    # phase from classical action
    if S_t is None:
        S_t = 0.0
    phi = pref * np.exp(- (x_arr - xc)**2 /(4*sigma0**2) + 1j * pc * (x_arr - xc)) * np.exp(1j * S_t)
    return phi

def crank_nicolson(psi0x):
    # same CN implementation as case4 but for V = 0.5 x^2 + F x
    psi = psi0x.copy()
    V = 0.5 * x**2 + F * x
    Nx_inner = Nx
    off = -1.0/(2.0 * dx**2)
    a = np.full(Nx_inner-1, -1j * dt / 2.0 * off)
    c = np.full(Nx_inner-1, -1j * dt / 2.0 * off)
    H_diag = 1.0/(dx**2) + V
    b = 1.0 + 1j * dt / 2.0 * H_diag
    a_r = np.full(Nx_inner-1, 1j * dt / 2.0 * off)
    c_r = np.full(Nx_inner-1, 1j * dt / 2.0 * off)
    b_r = 1.0 - 1j * dt / 2.0 * H_diag

    def thomas_solve(a, b, c, d):
        n = len(b)
        cp = np.empty(n-1, dtype=complex)
        dp = np.empty(n, dtype=complex)
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        for i in range(1, n-1):
            denom = b[i] - a[i-1] * cp[i-1]
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
        dp[n-1] = (d[n-1] - a[n-2] * dp[n-2]) / (b[n-1] - a[n-2] * cp[n-2])
        x_sol = np.empty(n, dtype=complex)
        x_sol[-1] = dp[-1]
        for i in range(n-2, -1, -1):
            x_sol[i] = dp[i] - cp[i] * x_sol[i+1]
        return x_sol

    snapshots = [psi.copy()]
    for n in range(Nt):
        d = b_r * psi.copy()
        d[1:] += a_r * psi[:-1]
        d[:-1] += c_r * psi[1:]
        psi = thomas_solve(a, b, c, d)
        if (n+1) % 10 == 0:
            snapshots.append(psi.copy())
    return snapshots

def compute_metrics(psi_num, psi_exact):
    dx = x[1]-x[0]
    diff = psi_num - psi_exact
    l2 = np.sqrt(np.sum(np.abs(diff)**2) * dx)
    maxe = np.max(np.abs(diff))
    prob = np.sum(np.abs(psi_num)**2) * dx
    x_avg = np.sum(np.conj(psi_num) * x * psi_num).real * dx
    x2_avg = np.sum(np.conj(psi_num) * (x**2) * psi_num).real * dx
    var_x = x2_avg - x_avg**2
    p_avg = np.sum(np.conj(psi_num) * (-1j * np.gradient(psi_num, dx))).real * dx
    p2_avg = np.sum(np.conj(psi_num) * (-1j * np.gradient(psi_num, dx))**2).real * dx
    var_p = p2_avg - p_avg**2
    return dict(l2=l2, maxe=maxe, prob=prob, x_avg=x_avg, var_x=var_x, p_avg=p_avg, var_p=var_p)

def main():
    psi_init = psi0(x)
    solver = 'SS'
    if solver == 'CN':
        snapshots = crank_nicolson(psi_init)
    else:
        # fallback to split-step spectral for comparison
        k = 2*np.pi*fftfreq(Nx, d=dx)
        Kfact = np.exp(-1j * (k**2) * dt / 2.0)
        V = 0.5 * x**2 + F * x
        expV_half = np.exp(-1j * V * dt / 2.0)
        psi = expV_half * psi_init
        snapshots = [psi.copy()]
        for n in range(Nt):
            psi_k = fft(psi)
            psi_k *= Kfact
            psi = ifft(psi_k)
            psi *= expV_half
            if (n+1) % 10 == 0:
                snapshots.append(psi.copy())

    times_snap = np.linspace(0, T, len(snapshots))
    S = action_along_trajectory(times_snap)
    os.makedirs('tdse_case5_outputs', exist_ok=True)
    frames = []
    metrics = []
    traj_analytical = []
    traj_numerical = []
    for i, psi_num in enumerate(snapshots):
        t = times_snap[i]
        psi_ex = psi_analytic_coherent(x, t, S_t=S[i])
        m = compute_metrics(psi_num, psi_ex)
        metrics.append((t, m))
        dens_num = np.abs(psi_num)**2
        dens_ex = np.abs(psi_ex)**2
        # expectation from numerical
        x_avg_num = np.sum(np.conj(psi_num) * x * psi_num).real * dx
        traj_numerical.append(x_avg_num)
        xc, pc = classical_trajectory(t)
        traj_analytical.append(xc)

        fig, ax = plt.subplots(figsize=(8,4))
        ax.fill_between(x, 0, dens_ex, color='#1f77b4', alpha=0.25)
        ax.plot(x, dens_ex, color='#1f77b4', lw=2, label='analytic')
        ax.plot(x, dens_num, color='#ff7f0e', lw=1.2, label='numerical')
        ax.axvline(xc, color='k', ls='--', lw=1)
        ax.set_xlim(-30, 30)
        ax.set_ylim(0, max(dens_ex.max(), dens_num.max())*1.3)
        ax.legend(loc='upper right')
        ax.set_xlabel('x')
        ax.set_ylabel('Probability density')
        ax.set_title(f't = {t:.3f}    L2={m["l2"]:.2e}  prob={m["prob"]:.6f}')
        ax.text(0.02, 0.95, f"<x>={m['x_avg']:.3f}\nVar(x)={m['var_x']:.3f}\nVar(p)={m['var_p']:.3f}", transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', fc='white', alpha=0.6))
        fname = f'tdse_case5_outputs/frame_{i:04d}.png'
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        pil_im = Image.open(fname).convert('RGBA')
        frames.append(pil_im)

    # normalize frames and save GIF
    widths, heights = zip(*(im.size for im in frames))
    max_w, max_h = max(widths), max(heights)
    norm_frames = []
    for im in frames:
        bg = Image.new('RGBA', (max_w, max_h), (255,255,255,255))
        x_off = (max_w - im.size[0]) // 2
        y_off = (max_h - im.size[1]) // 2
        bg.paste(im, (x_off, y_off), im)
        norm_frames.append(np.array(bg.convert('RGB')))

    gif_path = 'tdse_case5_outputs/animation.gif'
    imageio.mimsave(gif_path, norm_frames, duration=0.08)
    # also save MP4
    mp4_path = 'tdse_case5_outputs/animation.mp4'
    try:
        imageio.mimsave(mp4_path, norm_frames, fps=10)
    except Exception:
        print('MP4 write failed — ensure ffmpeg is installed. GIF created.')

    # save metrics and trajectory comparison
    with open('tdse_case5_outputs/metrics.txt', 'w') as f:
        for t, m in metrics:
            f.write(f"{t:.6f} {m['l2']:.6e} {m['maxe']:.6e} {m['prob']:.12f} {m['x_avg']:.6f} {m['var_x']:.6f} {m['p_avg']:.6f} {m['var_p']:.6f}\n")

    # save trajectory plot
    times_full = times_snap
    traj_analytical = np.array(traj_analytical)
    traj_numerical = np.array(traj_numerical)
    plt.figure()
    plt.plot(times_full, traj_analytical, label='classical/analytic')
    plt.plot(times_full, traj_numerical, '--', label='numerical <x>')
    plt.xlabel('t')
    plt.ylabel('<x>')
    plt.legend()
    plt.savefig('tdse_case5_outputs/trajectory_comparison.png', dpi=150)

    print('Outputs saved in tdse_case5_outputs')

if __name__ == '__main__':
    main()
