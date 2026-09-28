# Fin temperature sensitivities and time-dependent active subspaces

Python implementation of the analytical solution in Balcer et al.,
*HYPAD-UQ: A Derivative-Based Uncertainty Quantification Method Using a
Hypercomplex Finite Element Method*, J. Verif. Valid. Uncert. 8(2), 021002 (2023),
[DOI: 10.1115/1.4062459](https://doi.org/10.1115/1.4062459).
Equations (69)–(71) and Table 1 were checked against the supplied PDF:
`C:\Users\Rober\Downloads\vvuq_008_02_021002.pdf`
(accessible here at `/mnt/c/Users/Rober/Downloads/vvuq_008_02_021002.pdf`).

The quantity of interest is **temperature in kelvin at a specified time**, at
the insulated fin tip by default. The active-subspace calculation is an
extension of the paper's example; it is not a reproduction of its Sobol' indices.

## Run

In the current workspace, the `jetgp` environment already provides a working
`pyoti.sparse` build from otilib:

```bash
cd /root/fin_problem
python run.py --time 35 --case both --plot
```

If your shell selects another Python interpreter, use
`/root/anaconda3/envs/jetgp/bin/python` in place of `python`.

```bash
# Prompt for time; compute both cases
python run.py

# Reuse the same deterministic sampling design at several times
python run.py --time 0 10 35 100 450 --case both --samples 4096 --plot

# Larger sample for a convergence comparison
python run.py --time 35 --case 2 --samples 16384 --seed 123 --output results/refined

# A different location: X = x/b, measured FROM THE TIP
python run.py --time 35 --case 1 --position 0.5

# Use exactly the paper's 100-term truncation
python run.py --time 35 --terms 100 --fixed-terms

# Scientific checks
python -m unittest -v test_fin.py
```

Console output reports directions, eigenvalues and a parameter activity ranking.
The `results/` directory receives JSON files with all seven directions, the
gradient second-moment matrix, input definitions, selected dimension, activity
scores and bootstrap diagnostics. `--plot` also saves spectrum/loading PNGs.
Re-running the same case, time and position in the same output directory
replaces its result files; use `--output` to retain sampling comparisons.

Dependencies are NumPy, SciPy, and Matplotlib (only needed for plots), plus
otilib's compiled **`pyoti.sparse`** module. `requirements.txt` lists the ordinary
Python dependencies. Otilib must be installed separately for the Python/NumPy
version you use; installing an unrelated package named `otilib` is not a substitute.
An existing otilib build can be exposed with
`PYTHONPATH=/path/to/otilib/build python run.py --time 35`.
The code uses OTI first-order numbers for every production sensitivity and does
not fall back to finite differences. The local tested extension is under
`/root/Research/jetgp_2/otilib-master/build/pyoti`.

## Analytical model

The input vector, with the same order throughout the code, is

```text
p = [k, Cp, rho, hU, T_inf, T_W, b]
```

Following the paper, thickness `delta = 0.00475 m` and depth `L = 1 m` are
deterministic. With `X = x/b`, `tau = t*k/(b²*rho*Cp)`,
`omega² = 2*hU*b²/(k*delta*L)`, and `lambda_j = pi*(j-1/2)`, the solution is

```text
theta_ss = cosh(omega*X) / cosh(omega)
theta_tr = -2 sum_j (-1)^(j+1) lambda_j/(omega²+lambda_j²)
                    * cos(lambda_j*X) * exp(-(omega²+lambda_j²)*tau)
T = T_inf + (T_W - T_inf)*(theta_ss + theta_tr)
```

This specializes Eq. (71) to the paper's initial condition `T0 = T_inf`.
The hyperbolic ratio is evaluated using equivalent decaying exponentials to
avoid overflow. The paper's coordinate has **X=0 at the insulated tip and X=1
at the hot wall**. For optional interior locations, X is held fixed while b
varies; these are locations attached to the fin, not fixed physical distances.

The default retains at least 100 series terms. At small positive times it
increases the count until `lambda_N²*tau >= 45`; this is a practical decay
criterion, not a rigorous error bound. A 20,000-term safety limit gives a clear
error for excessively small positive times. Python callers can raise
`max_terms` in `temperature` / `temperature_and_gradient`. At exactly t=0,
interior temperature is imposed exactly as `T_inf`, avoiding truncation error
in the initial Fourier series. The wall is imposed exactly as `T_W`, including
t=0 (the boundary step convention). Very early tip heating below floating-point
resolution cannot be resolved by this double-precision series.

## Two input distributions

All seven inputs are independent. Coefficients of variation (CV) define
`sigma = mean * CV`; they are **not uniform half-widths**.

| Input | Mean (SI) | Case 1 | CV 1 | Case 2 | CV 2 |
|---|---:|---|---:|---|---:|
| k | 7.1 | Normal | 0.10 | Lognormal | 0.20 |
| Cp | 580 | Normal | 0.03 | Lognormal | 0.20 |
| rho | 4430 | Normal | 0.03 | Lognormal | 0.20 |
| hU | 114 | Normal | 0.10 | Lognormal | 0.20 |
| T_inf | 283 | Normal | 0.001 | Symmetric triangular* | 0.01 |
| T_W | 389 | Normal | 0.05 | Uniform | 0.20 |
| b | 0.051 | Normal | 0.01 | Uniform | 0.20 |

*Table 1 supplies a triangular family, mean and CV, without its mode/endpoints.
This implementation assumes symmetry: mode=mean, endpoints=mean ± sqrt(6)*sigma.
If an asymmetric triangle was intended, its additional shape information is
needed. Uniform endpoints are mean ± sqrt(3)*sigma. The lognormal parameters
are computed from the physical mean and standard deviation, not treated as
log-space moments.*

Sampling defaults to randomized Latin hypercube sampling (LHS). `--sampling mc`
uses independent Monte Carlo instead. Gaussian marginals are retained as stated
in the paper. If a nonpositive physical sample is drawn, the calculation stops
with an error instead of silently changing the distribution by clipping/redrawing.

## Meaning of the directions

For each time, define standardized physical inputs `z_i = (p_i-mu_i)/sigma_i`.
These have zero mean and unit variance in both cases, but Case 2 is still
non-Gaussian. The chain rule gives

```text
g = grad_z T = diag(sigma) grad_p T
C(t) = E[g g.T] ≈ (1/N) sum_s g_s g_s.T
C W = W diag(eigenvalues)
```

This is an **uncentered** gradient second moment. All seven partial derivatives
are extracted from one OTI evaluation per input sample. Directions are columns
of W, sorted by decreasing eigenvalue. For example, the first important
coordinate is `y1 = w1.T @ z`. Moving along `w1` in standardized space corresponds
to physical displacement `diag(sigma) @ w1`. Results include both that physical
displacement and the coefficients `w1/sigma` for writing y1 in physical inputs.
Physical displacements mix units and are not Euclidean-normalized.

This coordinate convention matters: eigenvectors calculated using unscaled
physical derivatives or a nonlinear Gaussian transform would answer a different
question. Case 1 and Case 2 also have different standard deviations, so a
comparison of their standardized directions includes both distribution shape
and uncertainty scale.

The default selects the smallest number r of directions capturing 99% of
`trace(C)`. Change this with `--energy`. This is **gradient energy**, not explained
output variance. Inspect eigenvalue gaps and stability before choosing a reduced
model. A direction's sign is arbitrary; the code makes its largest-magnitude
entry positive for repeatability. If eigenvalues are close/repeated, individual
vectors can rotate even while their joint subspace remains stable. Near-zero
eigenvectors are not meaningful important directions.

Parameter activity is `sum_{j<=r} eigenvalue_j * W_ij²`, also reported as normalized
fractions and a ranking. It summarizes parameters; the eigenvectors describe
important **combinations** of parameters. Neither is a Sobol' index.

The optional bootstrap (`--bootstrap 100`, or 0 to disable) reports eigenvalue
percentiles and distances between the estimated active-space projectors.
Projector distance equals the sine of the largest principal angle: zero means
agreement, one means a 90-degree difference. Row bootstrap of an LHS design is
only a stability diagnostic, not a calibrated confidence interval. Increase
sample counts and vary seeds to assess integration convergence.

## Python API

Run from this directory (or add it to your Python import path):

```python
from model import MEAN, temperature, temperature_and_gradient
from active_subspace import estimate_active_subspace

T = temperature(MEAN, time=35.0)
T, dT_dp = temperature_and_gradient(MEAN, time=35.0)
result = estimate_active_subspace(time=35.0, case=2, samples=4096, seed=42)
W_active = result["active_directions"]  # shape (7, r); columns are directions
print(result["parameters"], W_active)
```

Use `distributions.sample_inputs` and the API's `inputs=` argument to explicitly
reuse a physical sample across times. The CLI already reproduces the same design
by keeping case, sample count and seed fixed for all requested times.

## Validation

`test_fin.py` checks exact distribution moments/supports, OTI gradients against
complex-step derivatives at sampled inputs and multiple times/positions,
initial/wall/steady-state limits, series convergence, and an independent
finite-difference transient PDE solution. It also checks the exact rank-one
active space at t=0 and at the wall, the eigenpair equations, reproducibility,
and invalid inputs. Complex-step and the PDE solver are validation tools only.
