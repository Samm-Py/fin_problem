"""Estimate C(t)=E[grad_z T(t) grad_z T(t)^T], z=(p-mu)/sigma."""
import numpy as np

from distributions import describe, sample_inputs, standard_deviations
from model import MEAN, PARAMETERS, temperature_and_gradient


def eigensystem(matrix):
    eigenvalues, directions = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    directions = directions[:, order]
    # Eigenvector sign is arbitrary; fix the largest loading to be positive.
    for j in range(directions.shape[1]):
        if directions[np.argmax(np.abs(directions[:, j])), j] < 0:
            directions[:, j] *= -1
    return eigenvalues, directions


def estimate_active_subspace(time, case=1, *, samples=4096, seed=42,
                             sampling="lhs", position=0.0, terms=100,
                             adaptive=True, energy=0.99, bootstrap=100,
                             inputs=None):
    """Return JSON-compatible directions, eigenvalues, and sampling diagnostics.

    Directions are columns, ordered by descending eigenvalue, in standardized
    physical coordinates z=(p-mu)/sigma. Both cases use this same affine
    convention; no latent-Gaussian transform is applied to Case 2.
    Optional `inputs` allows reuse of an (n,7) physical sample across times.
    """
    if not np.isfinite(energy) or not 0 < energy <= 1:
        raise ValueError("energy must be in (0,1].")
    if not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer.")
    sigma = standard_deviations(case)
    points = sample_inputs(case, samples, seed, sampling) if inputs is None else np.asarray(inputs, dtype=float)
    if points.ndim != 2 or points.shape[1] != 7 or points.shape[0] < 2:
        raise ValueError("inputs must have shape (n,7), n >= 2.")
    samples = len(points)
    temperatures = np.empty(samples)
    gradients = np.empty_like(points)
    for i, point in enumerate(points):
        temperatures[i], physical_gradient = temperature_and_gradient(
            point, time, position=position, terms=terms, adaptive=adaptive)
        gradients[i] = sigma * physical_gradient
    # An uncentered second moment is essential: centering would erase a
    # constant nonzero gradient (for example, the t=0 ambient direction).
    matrix = gradients.T @ gradients / samples
    eigenvalues, directions = eigensystem(matrix)
    total = eigenvalues.sum()
    if total <= 0:
        raise ValueError("All sampled gradients vanish; no important direction is identifiable.")
    fractions = eigenvalues / total
    cumulative = np.cumsum(fractions)
    cumulative[-1] = 1.0
    dimension = min(7, int(np.searchsorted(cumulative, energy)) + 1)
    active = directions[:, :dimension]
    activity = (active**2) @ eigenvalues[:dimension]
    result = dict(
        time_seconds=float(time), case=int(case), position_X=float(position),
        samples=samples, seed=seed, sampling=sampling if inputs is None else "provided",
        parameters=list(PARAMETERS), input_distributions=describe(case),
        coordinates="z_i = (p_i - mean_i) / std_i",
        mean=MEAN.tolist(), std=sigma.tolist(),
        series_minimum_terms=terms, adaptive_series=adaptive,
        gradient_backend="otilib: pyoti.sparse, order=1",
        matrix=matrix.tolist(), eigenvalues=eigenvalues.tolist(),
        eigenvalue_fractions=fractions.tolist(), cumulative_energy=cumulative.tolist(),
        directions=directions.tolist(), active_directions=active.tolist(),
        energy_threshold=energy, selected_dimension=dimension,
        activity_scores=activity.tolist(), activity_fractions=(activity/activity.sum()).tolist(),
        parameter_ranking=[PARAMETERS[i] for i in np.argsort(activity)[::-1]],
        # z displacement w corresponds to physical displacement diag(sigma)*w.
        physical_displacements=(sigma[:, None] * active).tolist(),
        active_coordinate_coefficients=(active / sigma[:, None]).tolist(),
        temperature_mean_K=float(temperatures.mean()),
        temperature_std_K=float(temperatures.std(ddof=1)),
        notes=["Eigenvectors are columns. Signs are arbitrary; repeated eigenvalues permit rotations.",
               "Energy measures squared-gradient energy, not a fraction of temperature variance.",
               "Selected dimension is an energy heuristic; inspect the spectrum and stability."],
    )
    if case == 2:
        result["notes"].append("The ambient-temperature triangle is assumed symmetric; Table 1 gives only mean and CV.")
    if bootstrap:
        rng = np.random.default_rng(seed)
        boot_values = np.empty((bootstrap, 7))
        distances = np.empty(bootstrap)
        for i in range(bootstrap):
            g = gradients[rng.integers(samples, size=samples)]
            boot_values[i], boot_directions = eigensystem(g.T @ g / samples)
            boot_active = boot_directions[:, :dimension]
            distances[i] = np.linalg.norm(active @ active.T - boot_active @ boot_active.T, ord=2)
        result["bootstrap"] = dict(
            replicates=bootstrap,
            eigenvalue_percentiles_2_5_97_5=np.percentile(boot_values, [2.5, 97.5], axis=0).tolist(),
            active_projector_distance_percentiles_50_95=np.percentile(distances, [50, 95]).tolist(),
            interpretation="Projector distance is sin(largest principal angle); 0 means agreement.",
            caveat="Row bootstrap is a stability diagnostic; for LHS it is not a calibrated confidence interval.",
        )
    return result
