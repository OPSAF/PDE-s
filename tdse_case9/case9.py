"""
Case 9: 2D Uniform Electric Field V(x,y)=F*x

Compares analytic accelerated Gaussian with Split-Step, CN (sparse) and ADI-CN.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq
from PIL import Image
import imageio.v2 as imageio
import os
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Parameters
F = 0.5
x0, y0 = -2.0, 0.0
k0x, k0y = 1.0, 0.0
sigma = 1.0

# Grid
Nx = 128
Ny = 128
Lx = 40.0
Ly = 20.0
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
dx = x[1]-x[0]
dy = y[1]-y[0]
X, Y = np.meshgrid(x, y, indexing='xy')

# Time
T = 6.0
dt = 0.01
Nt = int(T/dt)

def psi0_2d(X, Y):
    norm = 1.0/(2*np.pi*sigma**2)**0.5
    return norm * np.exp(-((X-x0)**2 + (Y-y0)**2)/(4*sigma**2) + 1j*(k0x*(X-x0) + k0y*(Y-y0)))

def psi_free_1d(x_arr, t, x0_local, k0_local, sigma_local):
    w = sigma_local**2 + 1j * t / 2.0
    pref = (1/(2*np.pi*w))**0.25
    xc = x0_local + k0_local * t
    phase = 1j * (k0_local*(x_arr - xc) - 0.5 * k0_local**2 * t)
    expg = np.exp( - (x_arr - xc)**2 / (4.0 * w) + phase )
    return pref * expg

def psi_analytic_2d(X, Y, t):
    # shift in x by 0.5 F t^2 and gauge phase
    shift = 0.5 * F * t**2
    # gauge phase
    phase_g = np.exp(-1j*(F * t * X - (F**2) * t**3 / 6.0))
    psi_x = psi_free_1d(X - shift, t, x0, k0x, sigma)
    psi_y = psi_free_1d(Y, t, y0, k0y, sigma)
    return phase_g * psi_x * psi_y

def split_step_2d(psi0):
    psi = psi0.copy()
    kx = 2.0 * np.pi * fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2
    Kfact = np.exp(-1j * (K2) * dt / 2.0)
    V = F * X
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

def build_2d_H_matrix(Nx, Ny, dx, dy, Vflat):
    # 2D Laplacian with Dirichlet BC flattened row-major
    N = Nx*Ny
    ex = np.ones(Nx)
    Tx = sp.diags([ex, -2*ex, ex], offsets=[-1,0,1], shape=(Nx,Nx)) / dx**2
    ey = np.ones(Ny)
    Ty = sp.diags([ey, -2*ey, ey], offsets=[-1,0,1], shape=(Ny,Ny)) / dy**2
    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)
    Lap = sp.kron(Iy, Tx) + sp.kron(Ty, Ix)
    H = -0.5 * Lap + sp.diags(Vflat)
    return H.tocsr()

def cn_sparse(psi0):
    psi = psi0.copy()
    N = Nx*Ny
    V = (F * X).ravel()
    H = build_2d_H_matrix(Nx, Ny, dx, dy, V)
    I = sp.eye(N)
    A = (I + 1j * dt / 2.0 * H).tocsc()
    B = (I - 1j * dt / 2.0 * H).tocsc()
    snapshots = [psi.copy()]
    for nstep in range(Nt):
        b = B.dot(psi.ravel())
        sol = spla.spsolve(A, b)
        psi = sol.reshape((Ny, Nx))
        if (nstep+1) % 10 == 0:
            snapshots.append(psi.copy())
    return snapshots

def adi_cn(psi0):
    # ADI-Peaceman-Rachford: (I + i dt/2 Ax) u* = (I - i dt/2 Ay - i dt V) u^n
    # then (I + i dt/2 Ay) u^{n+1} = (I - i dt/2 Ax) u*
    psi = psi0.copy()
    Nx_inner = Nx
    Ny_inner = Ny
    ax_off = -1.0/(4.0 * dx**2) * 1j * dt
    bx = 1.0 + 1j * dt / 2.0 * (1.0/dx**2)
    ay_off = -1.0/(4.0 * dy**2) * 1j * dt
    by = 1.0 + 1j * dt / 2.0 * (1.0/dy**2)

    def thomas(a, b, c, d):
        n = len(b)
        cp = np.empty(n-1, dtype=complex)
        dp = np.empty(n, dtype=complex)
        cp[0] = c[0]/b[0]
        dp[0] = d[0]/b[0]
        for i in range(1, n-1):
            denom = b[i] - a[i-1]*cp[i-1]
            cp[i] = c[i]/denom
            dp[i] = (d[i] - a[i-1]*dp[i-1])/denom
        dp[n-1] = (d[n-1] - a[n-2]*dp[n-2])/(b[n-1] - a[n-2]*cp[n-2])
        x = np.empty(n, dtype=complex)
        x[-1] = dp[-1]
        for i in range(n-2, -1, -1):
            x[i] = dp[i] - cp[i]*x[i+1]
        return x

    snapshots = [psi.copy()]
    V = F * X
    for nstep in range(Nt):
        # compute Ay u^n (finite diff in y)
        Ay_u = np.zeros_like(psi, dtype=complex)
        Ay_u[1:-1,:] = (psi[2:,:] - 2*psi[1:-1,:] + psi[:-2,:]) / dy**2
        # RHS1 = psi - i dt/2 * Ay_u - i dt * V * psi
        rhs1 = psi - 1j * dt / 2.0 * Ay_u - 1j * dt * V * psi
        # solve (I + i dt/2 Ax) u* for each row (x-direction)
        ustar = np.zeros_like(psi, dtype=complex)
        a = np.full(Nx-1, ax_off)
        c = np.full(Nx-1, ax_off)
        b = np.full(Nx, bx)
        for j in range(Ny):
            d = rhs1[j,:].copy()
            ustar[j,:] = thomas(a, b, c, d)
        # compute Ax u* (finite diff in x)
        Ax_ustar = np.zeros_like(ustar, dtype=complex)
        Ax_ustar[:,1:-1] = (ustar[:,2:] - 2*ustar[:,1:-1] + ustar[:,:-2]) / dx**2
        rhs2 = ustar + 1j * dt / 2.0 * Ax_ustar
        # solve (I + i dt/2 Ay) u^{n+1} for each column (y-direction)
        unew = np.zeros_like(psi, dtype=complex)
        a_y = np.full(Ny-1, ay_off)
        c_y = np.full(Ny-1, ay_off)
        b_y = np.full(Ny, by)
        for i in range(Nx):
            dcol = rhs2[:,i].copy()
            unew[:,i] = thomas(a_y, b_y, c_y, dcol)
        psi = unew
        if (nstep+1) % 10 == 0:
            snapshots.append(psi.copy())
    return snapshots

def compute_metrics(psi_num, psi_ex):
    diff = psi_num - psi_ex
    l2 = np.sqrt(np.sum(np.abs(diff)**2) * dx * dy)
    linf = np.max(np.abs(diff))
    prob = np.sum(np.abs(psi_num)**2) * dx * dy
    # expectations
    x_avg = np.sum(np.conj(psi_num) * X * psi_num).real * dx * dy
    y_avg = np.sum(np.conj(psi_num) * Y * psi_num).real * dx * dy
    return dict(L2=l2, LInf=linf, prob=prob, x_avg=x_avg, y_avg=y_avg)

def save_outputs(name, snapshots):
    out = 'tdse_case9_outputs'
    os.makedirs(out, exist_ok=True)
    frames = []
    metrics_list = []
    times = np.linspace(0, T, len(snapshots))
    for i, psi in enumerate(snapshots):
        t = times[i]
        psi_ex = psi_analytic_2d(X, Y, t)
        dens_num = np.abs(psi)**2
        dens_ex = np.abs(psi_ex)**2
        m = compute_metrics(psi, psi_ex)
        metrics_list.append((t, m))
        fig, axs = plt.subplots(1,3, figsize=(15,4))
        im0 = axs[0].imshow(dens_num, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='viridis')
        axs[0].set_title('Numerical density')
        plt.colorbar(im0, ax=axs[0])
        im1 = axs[1].imshow(dens_ex, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], cmap='viridis')
        axs[1].set_title('Analytic density')
        plt.colorbar(im1, ax=axs[1])
        cs = axs[2].contourf(X, Y, dens_num - dens_ex, levels=20, cmap='RdBu_r')
        axs[2].set_title('Difference')
        plt.colorbar(cs, ax=axs[2])
        fig.suptitle(f'{name} t={t:.3f} L2={m["L2"]:.2e} LInf={m["LInf"]:.2e} prob={m["prob"]:.6f}')
        fname = os.path.join(out, f'{name}_frame_{i:04d}.png')
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        frames.append(Image.open(fname).convert('RGBA'))

    widths, heights = zip(*(im.size for im in frames))
    max_w, max_h = max(widths), max(heights)
    norm_frames = []
    for im in frames:
        bg = Image.new('RGBA', (max_w, max_h), (255,255,255,255))
        x_off = (max_w - im.size[0]) // 2
        y_off = (max_h - im.size[1]) // 2
        bg.paste(im, (x_off, y_off), im)
        norm_frames.append(np.array(bg.convert('RGB')))

    gif = os.path.join(out, f'animation_{name}.gif')
    mp4 = os.path.join(out, f'animation_{name}.mp4')
    imageio.mimsave(gif, norm_frames, duration=0.08)
    try:
        imageio.mimsave(mp4, norm_frames, fps=10)
    except Exception:
        print('MP4 write failed — ensure ffmpeg is installed. GIF created.')

    with open(os.path.join(out, f'metrics_{name}.txt'), 'w') as f:
        for t, mm in metrics_list:
            f.write(f"{t:.6f} {mm['L2']:.6e} {mm['LInf']:.6e} {mm['prob']:.12f} {mm['x_avg']:.6f} {mm['y_avg']:.6f}\n")

    print('Saved outputs for', name)

def spatial_convergence(method, Ns=[64,96,128]):
    results = []
    for N in Ns:
        global Nx, Ny, x, y, dx, dy, X, Y
        Nx = Ny = N
        x = np.linspace(-Lx/2, Lx/2, Nx)
        y = np.linspace(-Ly/2, Ly/2, Ny)
        dx = x[1]-x[0]; dy = y[1]-y[0]
        X, Y = np.meshgrid(x, y, indexing='xy')
        psi0 = psi0_2d(X, Y)
        if method == 'SS':
            snaps = split_step_2d(psi0)
        elif method == 'CN':
            snaps = cn_sparse(psi0)
        else:
            snaps = adi_cn(psi0)
        psi_num = snaps[-1]
        psi_ex = psi_analytic_2d(X, Y, T)
        L2 = np.sqrt(np.sum(np.abs(psi_num - psi_ex)**2) * dx * dy)
        results.append((N, L2))
    out = 'tdse_case9_outputs'
    os.makedirs(out, exist_ok=True)
    np.savetxt(os.path.join(out, f'spatial_convergence_{method}.txt'), results, header='N L2')
    print('Saved spatial convergence for', method)

def temporal_convergence(method, dts=[0.02, 0.01, 0.005]):
    results = []
    for dt_local in dts:
        global dt, Nt
        dt_old = dt
        dt = dt_local
        Nt = int(T/dt)
        psi0 = psi0_2d(X, Y)
        if method == 'SS':
            snaps = split_step_2d(psi0)
        elif method == 'CN':
            snaps = cn_sparse(psi0)
        else:
            snaps = adi_cn(psi0)
        psi_num = snaps[-1]
        psi_ex = psi_analytic_2d(X, Y, T)
        L2 = np.sqrt(np.sum(np.abs(psi_num - psi_ex)**2) * dx * dy)
        results.append((dt_local, L2))
        dt = dt_old
        Nt = int(T/dt)
    out = 'tdse_case9_outputs'
    os.makedirs(out, exist_ok=True)
    np.savetxt(os.path.join(out, f'temporal_convergence_{method}.txt'), results, header='dt L2')
    print('Saved temporal convergence for', method)

if __name__ == '__main__':
    psi0 = psi0_2d(X, Y)
    print('Running Split-Step...')
    snaps_ss = split_step_2d(psi0)
    save_outputs('SS', snaps_ss)
    print('Running ADI-CN...')
    snaps_adi = adi_cn(psi0)
    save_outputs('ADI', snaps_adi)
    print('Running CN (sparse)...')
    snaps_cn = cn_sparse(psi0)
    save_outputs('CN', snaps_cn)
    # convergence studies
    spatial_convergence('SS', Ns=[64,96,128])
    spatial_convergence('ADI', Ns=[64,96,128])
    spatial_convergence('CN', Ns=[64,96,128])
    temporal_convergence('SS', dts=[0.02, 0.01, 0.005])
    temporal_convergence('ADI', dts=[0.02, 0.01, 0.005])
    temporal_convergence('CN', dts=[0.02, 0.01, 0.005])
