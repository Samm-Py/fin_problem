# Validation and example runs

Checked on 2026-09-28 using `/root/anaconda3/envs/jetgp/bin/python`
(Python 3.9, NumPy/SciPy, the existing compiled `pyoti.sparse` extension).

`python -m unittest -v test_fin.py`: **8 tests passed**.
The suite includes OTI vs. complex-step sensitivities, an independent
Crank–Nicolson finite-difference PDE solution, series convergence,
initial/wall/steady limits, distribution moments, and active-space identities.
All production derivatives use OTI.

Example results and figures in `results/` use 4,096 LHS samples, seed 42,
and 100 bootstrap replicates. The independent comparison in
`results/convergence/` uses 8,192 LHS samples, seed 123.
Both evaluate the insulated tip. Dimensions use the 99% gradient-energy rule.

| Case | Time (s) | Dimension 4096 / 8192 | Energy retained (4096) | Leading eigenvalue relative difference | Largest active-space angle |
|---|---:|---:|---:|---:|---:|
| 1 | 35 | 1 / 1 | 99.650% | 0.003% | 0.015° |
| 1 | 450 | 2 / 2 | 99.327% | 0.586% | 4.509° |
| 2 | 35 | 3 / 3 | 99.619% | 1.398% | 1.547° |
| 2 | 450 | 3 / 3 | 99.347% | 2.977% | 1.107° |

The same dimension was selected in each comparison. This is a finite-sample
stability check, not proof of convergence. In particular, Case 2 at 35 s has
noticeable sampling variation in the smaller eigenvalues and individual
loadings; close parameter activity rankings can interchange. Use larger samples
and independent seeds when precise loadings are required. For Case 1 at 450 s,
the second and third eigenvalues are relatively close, so the second direction
is less stable than the leading direction; inspect both the spectrum and the
bootstrap diagnostics before fixing the reduced dimension.

The Case 2 ambient triangle is assumed symmetric because the supplied table
specifies mean and coefficient of variation without mode/endpoints.
