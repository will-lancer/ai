from __future__ import annotations

import unittest

from lean_reward_hacking.stats import (
    bimodality_summary,
    dip_test,
    fit_two_component_gaussian_mixture,
    mixture_bic_parametric_bootstrap,
    mixture_bootstrap,
    paired_rates,
    silverman_test,
    wilson_interval,
)


class StatsTests(unittest.TestCase):
    def test_paired_rates_use_discordant_pairs_for_gap(self) -> None:
        rates = paired_rates(2, 3, 1, 4)
        self.assertEqual(rates["n_pairs"], 10)
        self.assertAlmostEqual(float(rates["c_on"]), 0.5)
        self.assertAlmostEqual(float(rates["c_off"]), 0.3)
        self.assertAlmostEqual(float(rates["gap"]), 0.2)

    def test_wilson_interval_is_bounded(self) -> None:
        low, high = wilson_interval(9, 10)
        self.assertIsNotNone(low)
        self.assertIsNotNone(high)
        self.assertGreaterEqual(float(low), 0.0)
        self.assertLessEqual(float(high), 1.0)
        self.assertLess(float(low), float(high))

    def test_mixture_fit_and_bootstrap_are_deterministic(self) -> None:
        values = [-2.0, -1.9, -2.1, 1.9, 2.0, 2.1]
        first = fit_two_component_gaussian_mixture(values).as_dict()
        second = fit_two_component_gaussian_mixture(values).as_dict()
        self.assertEqual(first, second)
        bootstrap_a = mixture_bootstrap(values, replicates=8, seed=19)
        bootstrap_b = mixture_bootstrap(values, replicates=8, seed=19)
        self.assertEqual(bootstrap_a, bootstrap_b)
        self.assertGreater(float(first["bic_delta"]), 0.0)

    def test_mixture_fit_uses_deterministic_multiple_starts(self) -> None:
        values = [-3.0, -2.8, -3.2, 2.8, 3.0, 3.2]
        multi = fit_two_component_gaussian_mixture(values, n_starts=5)
        repeat = fit_two_component_gaussian_mixture(values, n_starts=5)
        self.assertEqual(multi.as_dict(), repeat.as_dict())
        self.assertLess(float(multi.means[0]), 0.0)
        self.assertGreater(float(multi.means[1]), 0.0)

    def test_mixture_bootstrap_reports_registered_null_bic_p_value(self) -> None:
        values = [-2.0, -1.9, -2.1, 1.9, 2.0, 2.1]
        result = mixture_bic_parametric_bootstrap(values, replicates=32, seed=19)
        self.assertEqual(result["calibration"], "parametric_bootstrap_fitted_one_component_gaussian_null")
        self.assertEqual(result["statistic"], "bic_delta")
        self.assertEqual(result["null_model"], "one_component_gaussian_mle")
        self.assertEqual(result["delta_bic_p_value"], result["bic_delta_p_value"])
        self.assertIn("null_parameters", result)
        self.assertEqual(result["null_successful_replicates"], 32)
        self.assertEqual(result["bic_delta_p_value"], result["p_value"])
        self.assertGreaterEqual(float(result["p_value"]), 0.0)
        self.assertLessEqual(float(result["p_value"]), 1.0)

    def test_mixture_null_calibration_is_deterministic(self) -> None:
        values = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        first = mixture_bootstrap(values, replicates=16, seed=7)
        second = mixture_bootstrap(values, replicates=16, seed=7)
        self.assertEqual(first, second)

    def test_optional_modality_methods_report_status(self) -> None:
        values = [-1.0, -0.9, -1.1, 1.0, 0.9, 1.1]
        dip = dip_test(values, bootstrap_replicates=2, seed=3)
        silverman = silverman_test(values, bootstrap_replicates=2, seed=3)
        self.assertIn(dip.status, {"ok", "unavailable", "insufficient_data"})
        self.assertIn(silverman.status, {"ok_approx", "unavailable", "insufficient_data"})

    def test_default_modality_summary_marks_dip_as_primary(self) -> None:
        result = bimodality_summary(
            [-1.0, -0.9, 0.9, 1.0], dip_bootstrap=0, mixture_bootstrap_replicates=0
        )
        self.assertEqual(result["primary_method"], "hartigan_dip")
        self.assertEqual(result["primary"]["method"], "hartigan_dip")


if __name__ == "__main__":
    unittest.main()
