"""
Case 6: 2D Isotropic Harmonic Oscillator

Analytic eigenstates and split-step spectral numerical solver in 2D.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq
import imageio.v2 as imageio
from PIL import Image
import os
import scipy.special as sps
import math
import matplotlib as mpl
mpl.rcParams['font.size'] = 10
plt.style.use('seaborn-v0_8-poster')

# Parameters
omega = 1.0
hbar = 1.0

# Grid and time
Nx = 128
Ny = 128
Lx = 12.0
Ly = 12.0
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
dx = x[1]-x[0]
dy = y[1]-y[0]
X, Y = np.meshgrid(x, y, indexing='xy')

T = 6.0
dt = 0.01
Nt = int(T/dt)

def hermite_phi(n, x):
    # physicists' Hermite H_n
    Hn = sps.eval_hermite(n, x)
    norm = 1.0 / (np.pi**0.25 * np.sqrt((2.0**n) * math.factorial(n)))
    return norm * Hn * np.exp(-x**2 / 2.0)

def psi_nm_analytic(n, m, X, Y, t):
    # 1D functions use variable scaled for omega=1
    phi_n = hermite_phi(n, X)
    phi_m = hermite_phi(m, Y)
    E_nm = (n + m + 1) * omega * hbar
    phase = np.exp(-1j * E_nm * t / hbar)
    return (phi_n * phi_m) * phase

def initial_state(n, m):
    return psi_nm_analytic(n, m, X, Y, 0.0)

def split_step_2d(psi0):
    psi = psi0.copy()
    kx = 2.0 * np.pi * fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2
    Kfact = np.exp(-1j * (K2) * dt / 2.0)
    V = 0.5 * (X**2 + Y**2)
    expV_half = np.exp(-1j * V * dt / 2.0)
    psi = expV_half * psi
    snapshots = [psi.copy()]
    for nstep in range(Nt):
        psi_k = fft2(psi)
        psi_k *= Kfact
        psi = ifft2(psi_k)
        psi *= expV_half
        if (nstep+1) % 10 == 0:
            snapshots.append(psi.copy())
    return snapshots

def metrics(psi_num, psi_ex):
    prob_density_diff = np.abs(psi_num - psi_ex)**2
    l2 = np.sqrt(np.sum(np.abs(psi_num - psi_ex)**2) * dx * dy)
    linf = np.max(np.abs(psi_num - psi_ex))
    prob = np.sum(np.abs(psi_num)**2) * dx * dy
    return dict(L2=l2, LInf=linf, prob=prob)

def save_plots_and_animation(n, m, snapshots, times):
    outdir = 'tdse_case6_outputs'
    os.makedirs(outdir, exist_ok=True)
    frames = []
    metrics_list = []
    for i, psi in enumerate(snapshots):
        t = times[i]
        psi_ex = psi_nm_analytic(n, m, X, Y, t)
        dens_num = np.abs(psi)**2
        dens_ex = np.abs(psi_ex)**2
        met = metrics(psi, psi_ex)
        metrics_list.append((t, met))

        # heatmap
        fig, axs = plt.subplots(1,3, figsize=(15,4))
        im0 = axs[0].imshow(dens_num, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='viridis')
        axs[0].set_title('Numerical density')
        plt.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(dens_ex, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='viridis')
        axs[1].set_title('Analytic density')
        plt.colorbar(im1, ax=axs[1])
        # contour
        cs = axs[2].contourf(X, Y, dens_num - dens_ex, levels=20, cmap='RdBu_r')
        axs[2].set_title('Difference (num - ana)')
        plt.colorbar(cs, ax=axs[2])
        fig.suptitle(f'n={n}, m={m}, t={t:.3f}  L2={met["L2"]:.2e}  LInf={met["LInf"]:.2e}')
        fname = os.path.join(outdir, f'frame_{i:04d}.png')
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        pil_im = Image.open(fname).convert('RGBA')
        frames.append(pil_im)

        # additional plots for first and last frame
        if i == 0 or i == len(snapshots)-1:
            # surface plot of analytic density
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig2 = plt.figure(figsize=(6,5))
            ax3 = fig2.add_subplot(111, projection='3d')
            ax3.plot_surface(X, Y, dens_ex, cmap='viridis', linewidth=0, antialiased=False)
            ax3.set_title(f'Analytic density surface t={t:.3f}')
            fig2.savefig(os.path.join(outdir, f'surface_t{i:04d}.png'), dpi=150)
            plt.close(fig2)

    # normalize frames to same size and write GIF/MP4
    widths, heights = zip(*(im.size for im in frames))
    max_w, max_h = max(widths), max(heights)
    norm_frames = []
    for im in frames:
        bg = Image.new('RGBA', (max_w, max_h), (255,255,255,255))
        x_off = (max_w - im.size[0]) // 2
        y_off = (max_h - im.size[1]) // 2
        bg.paste(im, (x_off, y_off), im)
        norm_frames.append(np.array(bg.convert('RGB')))

    gif_path = os.path.join(outdir, 'animation.gif')
    mp4_path = os.path.join(outdir, 'animation.mp4')
    imageio.mimsave(gif_path, norm_frames, duration=0.08)
    try:
        imageio.mimsave(mp4_path, norm_frames, fps=10)
    except Exception:
        print('MP4 write failed — ensure ffmpeg is installed. GIF created.')

    # save metrics
    with open(os.path.join(outdir, 'metrics.txt'), 'w') as f:
        for t, m in metrics_list:
            f.write(f"{t:.6f} {m['L2']:.6e} {m['LInf']:.6e} {m['prob']:.12f}\n")

    print('Saved outputs in', outdir)

def run_case(n, m):
    psi0 = initial_state(n, m)
    snapshots = split_step_2d(psi0)
    times = np.linspace(0, T, len(snapshots))
    save_plots_and_animation(n, m, snapshots, times)

def convergence_test(n, m, sizes=[64, 96, 128]):
    results = []
    for N in sizes:
        global Nx, Ny, x, y, dx, dy, X, Y
        Nx = Ny = N
        x = np.linspace(-Lx/2, Lx/2, Nx)
        y = np.linspace(-Ly/2, Ly/2, Ny)
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        X, Y = np.meshgrid(x, y, indexing='xy')
        psi0 = psi_nm_analytic(n, m, X, Y, 0.0)
        snapshots = split_step_2d(psi0)
        psi_num = snapshots[-1]
        psi_ex = psi_nm_analytic(n, m, X, Y, T)
        L2 = np.sqrt(np.sum(np.abs(psi_num - psi_ex)**2) * dx * dy)
        results.append((N, L2))
    # save convergence
    out = 'tdse_case6_outputs'
    os.makedirs(out, exist_ok=True)
    np.savetxt(os.path.join(out, 'convergence.txt'), results, header='N L2')
    print('Convergence results saved to', out)

