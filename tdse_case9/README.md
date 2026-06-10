Case 9 — 2D Uniform Electric Field

Potential: V(x,y) = F * x (uniform electric field in x-direction)

This folder contains `case9.py` which implements the exact accelerated Gaussian solution in 2D
and compares three numerical methods: Split-Step Fourier (SS), Crank–Nicolson (CN, sparse solve),
and Alternating Direction Implicit Crank–Nicolson (ADI-CN). It performs spatial/temporal convergence
studies, stability analysis, and produces density maps, contour plots, trajectory plots and MP4/GIF animations.

Run:

python case9.py

Outputs: `tdse_case9_outputs/` with frames, `animation_<method>.mp4`, `animation_<method>.gif`, and metrics files.
