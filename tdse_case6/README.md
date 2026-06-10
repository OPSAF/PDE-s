Case 6 — 2D Isotropic Harmonic Oscillator

Potential: V(x,y) = 0.5*(x^2 + y^2)

This script `case6.py` provides analytic eigenstates ψ_nm(x,y,t)=φ_n(x)φ_m(y) e^{-iE_nm t}
and a 2D split-step spectral numerical solver. It computes errors, probability conservation and
generates density heatmaps, contour and surface plots, and animations.

Run:

python case6.py

Outputs: `tdse_case6_outputs/` with images, `metrics.txt`, `animation.gif` and `animation.mp4`.
