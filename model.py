"""Transient fin solution: Balcer et al. (2023), Sec. 3.1, Eqs. (69)-(71)."""
from functools import lru_cache
import math

import numpy as np

PARAMETERS = ("k", "Cp", "rho", "hU", "T_inf", "T_W", "b")
UNITS = ("W/(m K)", "J/(kg K)", "kg/m^3", "W/(m^2 K)", "K", "K", "m")
MEAN = np.array([7.1, 580.0, 4430.0, 114.0, 283.0, 389.0, 0.051])
THICKNESS = 0.00475  # m
DEPTH = 1.0  # m; retain the paper's definition of omega^2, including L.


@lru_cache(maxsize=1)
def oti_module():
    """otilib's Python import name is pyoti; no substitute AD backend is used."""
    try:
        import pyoti.sparse as oti
    except ImportError as exc:
        raise ImportError(
            "otilib's pyoti.sparse extension is required. Activate your OTI "
            "environment or add its build directory to PYTHONPATH; see README.md."
        ) from exc
    return oti


def _real(x):
    return float(x.real) if hasattr(x, "real") else float(x)


def _validate(parameters, time, position, terms):
    if len(parameters) != 7:
        raise ValueError(f"Expected seven parameters in order {PARAMETERS}.")
    values = np.array([_real(v) for v in parameters])
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("All physical inputs must be finite and positive (temperatures in K).")
    if not np.isfinite(time) or time < 0:
        raise ValueError("Time must be finite and nonnegative, in seconds.")
    if not np.isfinite(position) or not 0 <= position <= 1:
        raise ValueError("position is X=x/b measured from the tip, and must be in [0,1].")
    if not isinstance(terms, (int, np.integer)) or terms < 1:
        raise ValueError("terms must be a positive integer.")


def temperature(parameters, time, *, position=0.0, terms=100, adaptive=True,
                max_terms=20000, backend=None):
    """Temperature in kelvin, accepting floats, complex values, or OTI scalars.

    X=0 is the insulated tip; X=1 is the prescribed-temperature wall. X is
    held fixed when b is differentiated. Initially T0=T_inf. At t=0 the
    wall takes its imposed value and every interior point has T_inf.
    `terms=100, adaptive=False` reproduces the paper's series truncation.
    Adaptive mode increases the term count at small positive times.
    """
    _validate(parameters, time, position, terms)
    k, cp, rho, hu, ambient, wall, length = parameters
    if position == 1.0:
        return wall
    if time == 0:
        return ambient
    if backend is None:
        backend = oti_module() if hasattr(k, "get_deriv") else np
    tau = time * k / (length * length * rho * cp)
    omega2 = 2.0 * hu * length * length / (k * THICKNESS * DEPTH)
    omega = backend.sqrt(omega2)
    # Equivalent to cosh(omega*X)/cosh(omega), without large positive exp.
    steady = (backend.exp(-omega * (1.0 - position))
              + backend.exp(-omega * (1.0 + position))) / (1.0 + backend.exp(-2.0 * omega))
    if adaptive:
        # Resolve exp(-lambda_j^2*tau) through exp(-45). This also leaves
        # margin for the polynomial factors introduced by differentiation.
        required = max(terms, math.ceil(math.sqrt(45.0 / _real(tau)) / math.pi + 0.5))
        if required > max_terms:
            raise ValueError(f"This very small time requires {required} series terms; "
                             f"increase max_terms (currently {max_terms}) or use a later time.")
        terms = required
    transient = 0.0
    for j in range(1, terms + 1):
        lam = math.pi * (j - 0.5)
        rate = omega2 + lam * lam
        coefficient = -2.0 * (-1.0 if j % 2 == 0 else 1.0) * lam * math.cos(lam * position)
        transient = transient + coefficient / rate * backend.exp(-rate * tau)
    return ambient + (wall - ambient) * (steady + transient)


def temperature_and_gradient(parameters, time, **kwargs):
    """Return T and seven physical partial derivatives from one OTI evaluation."""
    _validate(parameters, time, kwargs.get("position", 0.0), kwargs.get("terms", 100))
    oti = oti_module()
    seeded = [float(value) + oti.e(i + 1, order=1) for i, value in enumerate(parameters)]
    result = temperature(seeded, time, backend=oti, **kwargs)
    gradient = np.array([result.get_deriv(i + 1) for i in range(7)], dtype=float)
    if not np.isfinite(result.real) or not np.all(np.isfinite(gradient)):
        raise FloatingPointError("Nonfinite OTI temperature or derivatives.")
    return float(result.real), gradient
