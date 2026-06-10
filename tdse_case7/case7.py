"""
Case 7: 2D Coherent State in Isotropic Harmonic Oscillator

Standalone script moved from case6 coherent-state implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq
import imageio.v2 as imageio
from PIL import Image
import os
import math
import matplotlib as mpl
mpl.rcParams['font.size'] = 10
plt.style.use('seaborn-v0_8-poster')

# Grid and time (same scaling as other cases)
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

# Coherent state helpers
def coherent_initial(center=(2.0, 0.0), momentum=(0.0, 1.0)):
    x0, y0 = center
    px0, py0 = momentum
    psi_x = (1/np.pi)**0.25 * np.exp(- (X[0,:] - x0)**2 / 2.0 + 1j * px0 * (X[0,:] - x0))
    psi_y = (1/np.pi)**0.25 * np.exp(- (Y[:,0] - y0)**2 / 2.0 + 1j * py0 * (Y[:,0] - y0))
    psi = np.outer(psi_y, psi_x)
    return psi

def classical_traj_2d(t, x0, y0, px0, py0):
    xc = x0 * np.cos(t) + px0 * np.sin(t)
    yc = y0 * np.cos(t) + py0 * np.sin(t)
    pcx = -x0 * np.sin(t) + px0 * np.cos(t)
    pcy = -y0 * np.sin(t) + py0 * np.cos(t)
    return xc, yc, pcx, pcy

def psi_coherent_analytic(X, Y, t, x0, y0, px0, py0):
    xc, yc, pcx, pcy = classical_traj_2d(t, x0, y0, px0, py0)
    phi0 = (1/np.pi)**0.5 * np.exp(- ((X - xc)**2 + (Y - yc)**2) / 2.0)
    phase = np.exp(1j * (pcx * (X - xc) + pcy * (Y - yc)))
    return phi0 * phase

def expectations_2d(psi):
    prob = np.sum(np.abs(psi)**2) * dx * dy
    x_avg = np.sum(np.conj(psi) * X * psi).real * dx * dy
    y_avg = np.sum(np.conj(psi) * Y * psi).real * dx * dy
    kx = 2.0 * np.pi * fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    psi_k = fft2(psi)
    dpsi_dx = ifft2(1j * KX * psi_k)
    dpsi_dy = ifft2(1j * KY * psi_k)
    px_avg = np.sum(np.conj(psi) * (-1j * dpsi_dx)).real * dx * dy
    py_avg = np.sum(np.conj(psi) * (-1j * dpsi_dy)).real * dx * dy
    return dict(prob=prob, x_avg=x_avg, y_avg=y_avg, px_avg=px_avg, py_avg=py_avg)

def run_coherent_2d(center=(2.0,0.0), momentum=(0.0,1.0)):
    x0, y0 = center
    px0, py0 = momentum
    psi0 = coherent_initial(center=center, momentum=momentum)
    snapshots = split_step_2d(psi0)
    times = np.linspace(0, T, len(snapshots))
    outdir = 'tdse_case7_outputs'
    os.makedirs(outdir, exist_ok=True)
    frames = []
    traj_num = []
    traj_ana = []
    metrics_list = []
    for i, psi in enumerate(snapshots):
        t = times[i]
        psi_ex = psi_coherent_analytic(X, Y, t, x0, y0, px0, py0)
        dens_num = np.abs(psi)**2
        dens_ex = np.abs(psi_ex)**2
        L2 = np.sqrt(np.sum(np.abs(psi - psi_ex)**2) * dx * dy)
        Linf = np.max(np.abs(psi - psi_ex))
        exp_num = expectations_2d(psi)
        xc, yc, pcx, pcy = classical_traj_2d(t, x0, y0, px0, py0)
        metrics_list.append((t, L2, Linf, exp_num, dict(x_avg=xc, y_avg=yc, px_avg=pcx, py_avg=pcy)))
        traj_num.append((exp_num['x_avg'], exp_num['y_avg']))
        traj_ana.append((xc, yc))

        # heatmap
        fig, axs = plt.subplots(1,3, figsize=(15,4))
        im0 = axs[0].imshow(dens_num, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='viridis')
        axs[0].set_title('Numerical density')
        plt.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(dens_ex, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='viridis')
        axs[1].set_title('Analytic density')
        plt.colorbar(im1, ax=axs[1])
        cs = axs[2].contourf(X, Y, dens_num - dens_ex, levels=20, cmap='RdBu_r')
        axs[2].set_title('Difference (num - ana)')
        plt.colorbar(cs, ax=axs[2])
        fig.suptitle(f'Coherent t={t:.3f} L2={L2:.2e} LInf={Linf:.2e} prob={exp_num["prob"]:.6f}')
        fname = os.path.join(outdir, f'coh_frame_{i:04d}.png')
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        pil_im = Image.open(fname).convert('RGBA')
        frames.append(pil_im)

    widths, heights = zip(*(im.size for im in frames))
    max_w, max_h = max(widths), max(heights)
    norm_frames = []
    for im in frames:
        bg = Image.new('RGBA', (max_w, max_h), (255,255,255,255))
        x_off = (max_w - im.size[0]) // 2
        y_off = (max_h - im.size[1]) // 2
        bg.paste(im, (x_off, y_off), im)
        norm_frames.append(np.array(bg.convert('RGB')))
    gif_path = os.path.join(outdir, 'coherent_animation.gif')
    mp4_path = os.path.join(outdir, 'coherent_animation.mp4')
    imageio.mimsave(gif_path, norm_frames, duration=0.08)
    try:
        imageio.mimsave(mp4_path, norm_frames, fps=10)
    except Exception:
        print('MP4 write failed — ensure ffmpeg is installed. GIF created.')

    traj_num = np.array(traj_num)
    traj_ana = np.array(traj_ana)
    plt.figure()
    plt.plot(traj_ana[:,0], traj_ana[:,1], label='analytic')
    plt.plot(traj_num[:,0], traj_num[:,1], '--', label='numerical')
    plt.xlabel('x'); plt.ylabel('y'); plt.legend()
    plt.title('Coherent state center trajectory')
    plt.savefig(os.path.join(outdir, 'coherent_trajectory.png'), dpi=150)

    with open(os.path.join(outdir, 'metrics.txt'), 'w') as f:
        for t, L2, Linf, exp_num, exp_ex in metrics_list:
            f.write(f"{t:.6f} {L2:.6e} {Linf:.6e} {exp_num['prob']:.12f} {exp_num['x_avg']:.6f} {exp_num['y_avg']:.6f} {exp_num['px_avg']:.6f} {exp_num['py_avg']:.6f} {exp_ex['x_avg']:.6f} {exp_ex['y_avg']:.6f}\n")

    print('Coherent outputs saved in', outdir)

if __name__ == '__main__':
    run_coherent_2d(center=(2.0,0.0), momentum=(0.0,1.0))
