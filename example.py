"""
Example usage of the modular TDSE package.

This file demonstrates how to use the tdse package
to run TDSE simulations.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import tdse


def example_basic_simulation() -> None:
    """Basic example: free Gaussian wave packet propagation."""
    print("=" * 60)
    print("Example 1: Basic Free Gaussian Propagation")
    print("=" * 60)

    config = tdse.TDSEConfig(quick=True, outdir="example_output", save_gif=False)

    tdse.ensure_outdir(config.outdir)
    tdse.setup_plot_style(config.dpi)

    n = 384
    x, dx = tdse.grid(-30.0, 30.0, n)
    dt = 0.0025
    t = np.arange(0.0, 2.0 + 0.5 * dt, dt)

    x0, sigma, k0 = -8.0, 1.2, 2.0
    psi0 = tdse.gaussian_wavepacket(x, x0, sigma, k0, dx)
    v = tdse.potential_free(x)

    print(f"\nGrid: N={n}, dx={dx:.4f}")
    print(f"Initial state: x0={x0}, σ={sigma}, k0={k0}")
    print(f"Simulation: t_end={t[-1]:.2f}, dt={dt}")

    saved_t, psi_hist = tdse.solve("Crank-Nicolson", psi0, v, x, t, dx, dt)

    psi_exact = tdse.exact_free_gaussian(x, t[-1], x0, sigma, k0, dx)

    viz = tdse.TDSEVisualizer(config)
    viz.plot_wavepacket(
        x, psi_hist[-1],
        title="Free Gaussian Propagation",
        filename="example_basic.png",
        show_exact=psi_exact
    )

    print(f"\n[OK] Results saved to {config.outdir}/example_basic.png")


def example_barrier_scattering() -> None:
    """Example: barrier scattering with transmission/reflection analysis."""
    print("\n" + "=" * 60)
    print("Example 2: Barrier Scattering Analysis")
    print("=" * 60)

    config = tdse.TDSEConfig(quick=True, outdir="example_output", save_gif=False)

    x, dx = tdse.grid(-30.0, 30.0, 384)
    dt = 0.002
    t = np.arange(0.0, 3.0 + 0.5 * dt, dt)

    barrier_left, barrier_right = -0.5, 0.5
    V0, sigma, k0 = 1.0, 1.2, 2.0

    psi0 = tdse.gaussian_wavepacket(x, -10.0, sigma, k0, dx)
    v = tdse.potential_rect_barrier(x, V0, barrier_left, barrier_right)

    print(f"\nBarrier: V0={V0}, [{barrier_left}, {barrier_right}]")
    print(f"Wave packet: x0=-10.0, σ={sigma}, k0={k0}")

    saved_t, psi_hist = tdse.solve("Split-Step-FFT", psi0, v, x, t, dx, dt)
    psi_final = psi_hist[-1]

    T, R, total = tdse.compute_transmission_reflection(
        psi_final, x, dx,
        barrier_left=barrier_left,
        barrier_right=barrier_right,
        buffer=0.5
    )

    print(f"\nScattering results:")
    print(f"  Transmission (T): {T:.6f}")
    print(f"  Reflection (R):   {R:.6f}")
    print(f"  Total (T+R):     {total:.6f}")

    analysis = tdse.analyze_barrier_scattering(
        psi_final, x, dx,
        barrier_center=0.0,
        barrier_width=1.0,
        buffer=0.5
    )

    print(f"\nDetailed analysis:")
    print(f"  Unaccounted: {analysis['unaccounted']:.6f}")
    print(f"  Barrier amplitude: {analysis['barrier_amplitude']:.6f}")


def example_factory_usage() -> None:
    """Example: using factories to create potentials and solvers."""
    print("\n" + "=" * 60)
    print("Example 3: Factory Pattern Usage")
    print("=" * 60)

    print(f"\nAvailable potentials: {tdse.TDSEPotentialFactory.list_potentials()}")
    print(f"Available solvers: {tdse.TDSESolverFactory.list_solvers()}")

    pot_factory = tdse.TDSEPotentialFactory()
    solver_factory = tdse.TDSESolverFactory()

    free_pot = pot_factory.create("free")
    barrier_pot = pot_factory.create("barrier", v0=1.0, a=-0.5, b=0.5)

    print(f"\nCreated potentials:")
    print(f"  {free_pot.name}: {free_pot.info()}")
    print(f"  {barrier_pot.name}: {barrier_pot.info()}")

    cn_solver = solver_factory.create("CN")
    ssf_solver = solver_factory.create("Split-Step-FFT")

    print(f"\nCreated solvers:")
    print(f"  {cn_solver.name}: {cn_solver.info()}")
    print(f"  {ssf_solver.name}: {ssf_solver.info()}")


def example_2d_simulation() -> None:
    """Example: 2D free Gaussian propagation."""
    print("\n" + "=" * 60)
    print("Example 4: 2D Free Gaussian Propagation")
    print("=" * 60)

    import numpy as np
    from tdse.solvers import solve_2d, step_split_step_fft_2d
    from tdse.potentials import mass_2d

    nx = ny = 48
    x = np.linspace(-15.0, 15.0, nx, endpoint=False)
    y = np.linspace(-15.0, 15.0, ny, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y, indexing="ij")
    kx_1d = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx_1d, ky_1d, indexing="ij")

    x0, y0, sigma, kx0, ky0 = -5.0, -2.0, 1.2, 2.0, 1.0
    psi0 = tdse.gaussian_wavepacket_2d(X, Y, x0, y0, sigma, kx0, ky0, dx, dy)
    V = np.zeros_like(X)

    dt = 0.01
    n_steps = 50

    print(f"\nGrid: {nx}×{ny}")
    print(f"Initial position: ({x0}, {y0})")
    print(f"Wave packet: σ={sigma}, kx0={kx0}, ky0={ky0}")
    print(f"Steps: {n_steps}, dt={dt}")

    psi = psi0.copy()
    for _ in range(n_steps):
        psi = step_split_step_fft_2d(psi, V, KX, KY, dt)

    mass = mass_2d(psi, dx, dy)
    print(f"\nFinal mass: {mass:.6f}")
    print(f"Mass conserved: {abs(mass - 1.0) < 1e-10}")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" " * 15 + "TDSE Package Examples")
    print("=" * 70)

    example_basic_simulation()
    example_barrier_scattering()
    example_factory_usage()
    example_2d_simulation()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

