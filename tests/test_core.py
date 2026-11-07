import unittest
import numpy as np

from bankpulse.analysis import (
    mean_confidence_interval,
    least_squares_trend,
    reserve_cost,
    marginal_holding_cost,
)
from bankpulse.bayes import beta_posterior
from bankpulse.simulation import simulate_daily_demand
from bankpulse.models import ReservePolicy


class TestBankPulse(unittest.TestCase):

    def test_mean_confidence_interval_contains_mean(self):
        values = [10, 12, 14, 16, 18]
        mean, lower, upper = mean_confidence_interval(values)
        self.assertEqual(mean, 14.0)
        self.assertLessEqual(lower, mean)
        self.assertGreaterEqual(upper, mean)

    def test_least_squares_linear_data(self):
        intercept, slope, fitted = least_squares_trend([2, 4, 6, 8])
        self.assertAlmostEqual(intercept, 2.0)
        self.assertAlmostEqual(slope, 2.0)
        np.testing.assert_allclose(fitted, [2, 4, 6, 8])

    def test_bayesian_update(self):
        result = beta_posterior(2, 8, 4, 20)
        self.assertEqual(result["alpha_posterior"], 6)
        self.assertEqual(result["beta_posterior"], 24)
        self.assertAlmostEqual(result["posterior_mean"], 0.2)

    def test_reserve_cost(self):
        result = reserve_cost(100, [50, 100, 150], 0.1, 0.01)
        self.assertAlmostEqual(result["shortfall_probability"], 1 / 3)
        self.assertAlmostEqual(result["average_shortfall"], 50 / 3)

    def test_marginal_holding_cost(self):
        self.assertEqual(marginal_holding_cost(0.002), 0.002)

    def test_simulation_nonnegative(self):
        rng = np.random.default_rng(1)
        values = simulate_daily_demand(20, 1000, 50, rng)
        self.assertEqual(len(values), 50)
        self.assertTrue(np.all(values >= 0))

    def test_reserve_policy_validation(self):
        policy = ReservePolicy("BR001", 0.95, 10000)
        self.assertEqual(policy.reserve_amount, 10000)
        with self.assertRaises(ValueError):
            ReservePolicy("BR001", 1.2, 10000)


if __name__ == "__main__":
    unittest.main()
