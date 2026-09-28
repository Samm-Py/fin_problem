"""Scientific checks: python -m unittest -v test_fin.py."""
import unittest

import numpy as np
from scipy.linalg import solve_banded

from active_subspace import estimate_active_subspace
from distributions import marginals, sample_inputs, standard_deviations
from model import DEPTH, MEAN, THICKNESS, temperature, temperature_and_gradient


class FinTests(unittest.TestCase):
    def test_distribution_moments(self):
        for case in (1, 2):
            for i, marginal in enumerate(marginals(case)):
                self.assertAlmostEqual(marginal.mean(), MEAN[i], places=10)
                self.assertAlmostEqual(marginal.std(), standard_deviations(case)[i], places=10)
        # Validate bounded distributions by their actual support, not a sample.
        for i in (5, 6):
            lo, hi = marginals(2)[i].support()
            self.assertAlmostEqual(lo, MEAN[i] - np.sqrt(3)*standard_deviations(2)[i])
            self.assertAlmostEqual(hi, MEAN[i] + np.sqrt(3)*standard_deviations(2)[i])

    def test_oti_against_complex_step(self):
        for case in (1, 2):
            for point in sample_inputs(case, 3, seed=123):
                for t in (1.0, 35.0, 450.0):
                    for position in (0.0, 0.4):
                        value, gradient = temperature_and_gradient(point, t, position=position)
                        self.assertAlmostEqual(value, temperature(point, t, position=position), places=10)
                        reference = np.empty(7)
                        for j in range(7):
                            z = point.astype(complex)
                            step = point[j]*1e-25
                            z[j] += 1j*step
                            reference[j] = temperature(z, t, position=position).imag / step
                        # Compare derivatives after unit standardization.
                        np.testing.assert_allclose(gradient*standard_deviations(case),
                                                   reference*standard_deviations(case), rtol=2e-9, atol=2e-10)

    def test_initial_wall_and_steady_limits(self):
        value, gradient = temperature_and_gradient(MEAN, 0.0)
        self.assertEqual(value, MEAN[4])
        np.testing.assert_array_equal(gradient, [0, 0, 0, 0, 1, 0, 0])
        for t in (0.0, 1.0, 35.0, 1e6):
            value, gradient = temperature_and_gradient(MEAN, t, position=1.0)
            self.assertEqual(value, MEAN[5])
            np.testing.assert_array_equal(gradient, [0, 0, 0, 0, 0, 1, 0])
        omega = np.sqrt(2*MEAN[3]*MEAN[6]**2/(MEAN[0]*THICKNESS*DEPTH))
        value, gradient = temperature_and_gradient(MEAN, 1e6)
        self.assertAlmostEqual(value, MEAN[4]+(MEAN[5]-MEAN[4])/np.cosh(omega), places=10)
        np.testing.assert_allclose(gradient[[1, 2]], 0, atol=1e-13)
        # Equal wall and ambient values must not cause division by zero.
        point = MEAN.copy()
        point[5] = point[4]
        self.assertEqual(temperature(point, 35), point[4])

    def test_series_convergence(self):
        for t in (0.01, 1, 35, 450):
            for position in (0, 0.95):
                value, grad = temperature_and_gradient(MEAN, t, position=position)
                ref, ref_grad = temperature_and_gradient(MEAN, t, position=position, terms=1500, adaptive=False)
                self.assertAlmostEqual(value, ref, places=10)
                np.testing.assert_allclose(grad, ref_grad, rtol=1e-8, atol=1e-8)

    def test_independent_finite_difference_pde(self):
        # Crank-Nicolson solution of dtheta/dtau = theta_XX - omega² theta,
        # insulated at X=0 and theta=1 at X=1. This checks the formula's
        # sign, time scale, coordinate convention and convection coefficient.
        n = 200
        dx = 1/n
        omega2 = 2*MEAN[3]*MEAN[6]**2/(MEAN[0]*THICKNESS*DEPTH)
        final_tau = 35*MEAN[0]/(MEAN[6]**2*MEAN[2]*MEAN[1])
        steps = 2000
        dt = final_tau/steps
        diagonal = np.full(n, -2/dx**2-omega2)
        upper = np.full(n-1, 1/dx**2)
        upper[0] *= 2  # ghost point enforces zero tip flux
        lower = np.full(n-1, 1/dx**2)
        lhs = np.zeros((3, n))
        lhs[1] = 1-dt*diagonal/2
        lhs[0, 1:] = -dt*upper/2
        lhs[2, :-1] = -dt*lower/2
        theta = np.zeros(n)
        for _ in range(steps):
            rhs = (1+dt*diagonal/2)*theta
            rhs[:-1] += dt*upper*theta[1:]/2
            rhs[1:] += dt*lower*theta[:-1]/2
            rhs[-1] += dt/dx**2
            theta = solve_banded((1, 1), lhs, rhs)
        numerical = MEAN[4] + (MEAN[5]-MEAN[4])*theta[0]
        self.assertLess(abs(numerical-temperature(MEAN, 35)), 0.003)

    def test_subspace_initial_and_wall(self):
        for case in (1, 2):
            for time, position, index in ((0, 0, 4), (35, 1, 5)):
                result = estimate_active_subspace(time, case, samples=16, bootstrap=5, position=position)
                self.assertEqual(result["selected_dimension"], 1)
                expected = np.eye(7)[:, index]
                np.testing.assert_allclose(np.array(result["directions"])[:, 0], expected)
                self.assertAlmostEqual(result["eigenvalues"][0], standard_deviations(case)[index]**2)

    def test_subspace_matrix_and_reproducibility(self):
        for case in (1, 2):
            result = estimate_active_subspace(35, case, samples=32, bootstrap=5)
            other = estimate_active_subspace(35, case, samples=32, bootstrap=5)
            self.assertEqual(result, other)
            matrix = np.array(result["matrix"])
            vectors = np.array(result["directions"])
            values = np.array(result["eigenvalues"])
            np.testing.assert_allclose(vectors.T@vectors, np.eye(7), atol=1e-14)
            np.testing.assert_allclose(matrix@vectors, vectors*values, atol=1e-11)
            self.assertTrue(np.all(values >= 0))
            self.assertAlmostEqual(sum(result["activity_fractions"]), 1.0)

    def test_invalid_inputs(self):
        for time in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                temperature(MEAN, time)
        for position in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                temperature(MEAN, 35, position=position)
        with self.assertRaises(ValueError):
            sample_inputs(3)
        with self.assertRaises(ValueError):
            temperature(MEAN, 1e-20)


if __name__ == "__main__":
    unittest.main()
