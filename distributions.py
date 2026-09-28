"""Independent input distributions from Table 1 of Balcer et al. (2023)."""
import numpy as np
from scipy import stats
from scipy.stats import qmc

from model import MEAN, PARAMETERS, UNITS

COV = {
    1: np.array([0.10, 0.03, 0.03, 0.10, 0.001, 0.05, 0.01]),
    2: np.array([0.20, 0.20, 0.20, 0.20, 0.01, 0.20, 0.20]),
}


def standard_deviations(case):
    if case not in COV:
        raise ValueError("case must be 1 or 2.")
    return MEAN * COV[case]


def marginals(case):
    std = standard_deviations(case)
    if case == 1:
        return [stats.norm(loc=m, scale=s) for m, s in zip(MEAN, std)]
    result = []
    for i, (mean, sigma) in enumerate(zip(MEAN, std)):
        if i < 4:
            log_sigma = np.sqrt(np.log1p((sigma / mean) ** 2))
            result.append(stats.lognorm(s=log_sigma, scale=mean * np.exp(-log_sigma**2 / 2)))
        elif i == 4:
            # Table 1 specifies only mean and CV: assume a symmetric triangle.
            half_width = np.sqrt(6.0) * sigma
            result.append(stats.triang(c=0.5, loc=mean-half_width, scale=2*half_width))
        else:
            half_width = np.sqrt(3.0) * sigma
            result.append(stats.uniform(loc=mean-half_width, scale=2*half_width))
    return result


def sample_inputs(case, samples=4096, seed=42, sampling="lhs"):
    """Draw physical inputs. LHS stratifies each independent marginal."""
    if not isinstance(samples, (int, np.integer)) or samples < 2:
        raise ValueError("samples must be an integer >= 2.")
    distributions = marginals(case)
    if sampling == "lhs":
        u = qmc.LatinHypercube(d=7, seed=seed).random(samples)
    elif sampling == "mc":
        u = np.random.default_rng(seed).random((samples, 7))
    else:
        raise ValueError("sampling must be 'lhs' or 'mc'.")
    u = np.clip(u, np.finfo(float).eps, 1-np.finfo(float).eps)
    values = np.column_stack([dist.ppf(u[:, i]) for i, dist in enumerate(distributions)])
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("A distribution produced a nonphysical sample. Samples are not "
                         "silently clipped or redrawn, since that would alter Table 1.")
    return values


def describe(case):
    std = standard_deviations(case)
    families = ["normal"]*7 if case == 1 else ["lognormal"]*4 + ["symmetric triangular", "uniform", "uniform"]
    return [dict(name=name, unit=unit, mean=float(mean), std=float(sigma), distribution=family)
            for name, unit, mean, sigma, family in zip(PARAMETERS, UNITS, MEAN, std, families)]
