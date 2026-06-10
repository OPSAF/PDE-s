"""
Case 4: Inverted Harmonic Oscillator V(x) = -x^2 / 2

Analytic solution computed using the quadratic propagator and performing the Gaussian integral analytically.
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

# Parameters
x0 = 0.0
k0 = 0.0
sigma = 1.0

# Spatial grid
Nx = 2048
L = 100.0
x = np.linspace(-L/2, L/2, Nx)
dx = x[1]-x[0]

# Temporal grid
T = 6.0
dt = 0.01
Nt = int(T/dt)
times = np.linspace(0, T, Nt+1)

def psi0(x):
    norm = (1/(2*np.pi*sigma**2))**0.25
    return norm * np.exp(- (x - x0)**2 /(4*sigma**2) + 1j*k0*x)

def psi_analytic_inv(x_arr, t):
    if abs(t) < 1e-12:
        return psi0(x_arr)
    sinh_t = np.sinh(t)
    cosh_t = np.cosh(t)
    pref = 1.0/np.sqrt(2*np.pi*1j*sinh_t)
    A0 = (1/(2*np.pi*sigma**2))**0.25
    psi = np.zeros_like(x_arr, dtype=complex)
    # constants for integral
    for idx, xval in enumerate(x_arr):
        # coefficients for Gaussian integral over x'
        A = -1.0/(4.0*sigma**2) + 1j * cosh_t/(2.0*sinh_t)
        B = x0/(2.0*sigma**2) + 1j * k0 - 1j * xval / sinh_t
        C = -x0**2/(4.0*sigma**2) + 1j * xval**2 * cosh_t/(2.0*sinh_t)
        # integral: sqrt(pi/(-A)) * exp(-B^2/(4A) + C)
        integral = np.sqrt(np.pi/(-A)) * np.exp(-B**2/(4.0*A) + C)
        psi[idx] = pref * A0 * integral
    return psi

def split_step(psi0x):
    psi = psi0x.copy()
    k = 2*np.pi*fftfreq(Nx, d=dx)
    Kfact = np.exp(-1j * (k**2) * dt / 2.0)
    V = -0.5 * x**2
    expV_half = np.exp(-1j * V * dt / 2.0)
    psi = expV_half * psi
    snapshots = [psi.copy()]
    for n in range(Nt):
        psi_k = fft(psi)
        psi_k *= Kfact
        psi = ifft(psi_k)
        psi *= expV_half
        if (n+1) % 10 == 0:
            snapshots.append(psi.copy())
    return snapshots


def crank_nicolson(psi0x):
    # Crank-Nicolson for H = -1/2 d2/dx2 + V(x)
    psi = psi0x.copy()
    V = -0.5 * x**2
    Nx_inner = Nx
    # coefficients for second derivative
    r = 1.0/(2.0 * dx**2)  # because off-diagonal in T is -1/(2 dx^2)
    # Build tridiagonal for matrix A = I + i dt/2 H
    # H diagonal: 1/dx^2 + V_j ; H offdiag: -1/(2 dx^2)
    off = -1.0/(2.0 * dx**2)
    # Precompute tri-diagonal coefficients for A and B
    a = np.full(Nx_inner-1, -1j * dt / 2.0 * off)  # sub-diagonal
    c = np.full(Nx_inner-1, -1j * dt / 2.0 * off)  # super-diagonal
    H_diag = 1.0/(dx**2) + V
    b = 1.0 + 1j * dt / 2.0 * H_diag
    # RHS matrix coefficients (I - i dt/2 H) applied to psi^n -> use vectors
    a_r = np.full(Nx_inner-1, 1j * dt / 2.0 * off)
    c_r = np.full(Nx_inner-1, 1j * dt / 2.0 * off)
    b_r = 1.0 - 1j * dt / 2.0 * H_diag

    snapshots = [psi.copy()]
    # Thomas algorithm helper for complex tridiagonal solve
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
        x = np.empty(n, dtype=complex)
        x[-1] = dp[-1]
        for i in range(n-2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i+1]
        return x

    for n in range(Nt):
        # compute RHS d = B psi^n (vector)
        d = b_r * psi.copy()
        d[1:] += a_r * psi[:-1]
        d[:-1] += c_r * psi[1:]
        # apply Dirichlet BCs (psi[0]=psi[-1]=0) by setting first/last equations accordingly
        # solve A psi^{n+1} = d
        # for simplicity include boundaries as part of system (they remain small due to domain)
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
    # momentum via spectral derivative
    psi_k = fft(psi_num)
    k = 2*np.pi*fftfreq(Nx, d=dx)
    # momentum operator p = -i d/dx
    p = np.fft.ifft(1j * k * psi_k)
    p_avg = np.sum(np.conj(psi_num) * (-1j * np.gradient(psi_num, dx))).real * dx
    p2_avg = np.sum(np.conj(psi_num) * (-1j * np.gradient(psi_num, dx))**2).real * dx
    var_p = p2_avg - p_avg**2
    return dict(l2=l2, maxe=maxe, prob=prob, x_avg=x_avg, var_x=var_x, p_avg=p_avg, var_p=var_p)

def main():
    psi_init = psi0(x)
    # choose solver: 'CN' or 'SS' (split-step)
    solver = 'CN'
    if solver == 'CN':
        snapshots = crank_nicolson(psi_init)
    else:
        snapshots = split_step(psi_init)
    times_snap = np.linspace(0, T, len(snapshots))
    os.makedirs('tdse_case4_outputs', exist_ok=True)
    frames = []
    metrics = []
    for i, psi_num in enumerate(snapshots):
        t = times_snap[i]
        psi_ex = psi_analytic_inv(x, t)
        m = compute_metrics(psi_num, psi_ex)
        metrics.append((t, m))
        dens_num = np.abs(psi_num)**2
        dens_ex = np.abs(psi_ex)**2
        x_c = 0.0
        fig, ax = plt.subplots(figsize=(8,4))
        ax.fill_between(x, 0, dens_ex, color='#2ca02c', alpha=0.25)
        ax.plot(x, dens_ex, color='#2ca02c', lw=2, label='analytic')
        ax.plot(x, dens_num, color='#d62728', lw=1.2, label='numerical')
        ax.set_xlim(-30, 30)
        ax.set_ylim(0, max(dens_ex.max(), dens_num.max())*1.3)
        ax.legend(loc='upper right')
        ax.set_xlabel('x')
        ax.set_ylabel('Probability density')
        ax.set_title(f't = {t:.3f}    L2={m["l2"]:.2e}  max={m["maxe"]:.2e}  prob={m["prob"]:.6f}')
        ax.text(0.02, 0.95, f"<x>={m['x_avg']:.3f}\nVar(x)={m['var_x']:.3f}\nVar(p)={m['var_p']:.3f}", transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', fc='white', alpha=0.6))
        fname = f'tdse_case4_outputs/frame_{i:04d}.png'
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        pil_im = Image.open(fname).convert('RGBA')
        frames.append(pil_im)

    # normalize and save both GIF and MP4
    if len(frames) == 0:
        print('No frames generated')
        return
    widths, heights = zip(*(im.size for im in frames))
    max_w, max_h = max(widths), max(heights)
    norm_frames = []
    for im in frames:
        bg = Image.new('RGBA', (max_w, max_h), (255,255,255,255))
        x_off = (max_w - im.size[0]) // 2
        y_off = (max_h - im.size[1]) // 2
        bg.paste(im, (x_off, y_off), im)
        norm_frames.append(np.array(bg.convert('RGB')))

    mp4_path = 'tdse_case4_outputs/animation.mp4'
    gif_path = 'tdse_case4_outputs/animation.gif'
    imageio.mimsave(mp4_path, norm_frames, fps=10)
    imageio.mimsave(gif_path, norm_frames, duration=0.08)

    # save metrics
    with open('tdse_case4_outputs/metrics.txt', 'w') as f:
        for t, m in metrics:
            f.write(f"{t:.6f} {m['l2']:.6e} {m['maxe']:.6e} {m['prob']:.12f} {m['x_avg']:.6f} {m['var_x']:.6f} {m['p_avg']:.6f} {m['var_p']:.6f}\n")

    print('Outputs saved in tdse_case4_outputs')

if __name__ == '__main__':
    main()
