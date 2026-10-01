"""Causality and parameter-state checks for the independent paper sweep."""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd

from MonthFu.experiments.src.paper_sweep import (
    PaperSpec, _stationary_coefficients, direction_block_comparison, direction_regression_metrics, fit_css_arima,
    fit_damped_holt, fit_paper, paper_features, predict_paper)
from MonthFu.src.modeling import add_targets


class PaperSweepTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.prices = pd.DataFrame({"Date": pd.bdate_range("2016-01-01", periods=600),
                                    "Close": 150*np.exp(np.cumsum(rng.normal(0.0001, 0.01, 600)))})
        self.features, self.groups = paper_features(self.prices)
        self.frame = add_targets(self.features, 30, "calendar")

    def test_feature_prefix_invariance_and_no_target_columns(self):
        prefix, _ = paper_features(self.prices.iloc[:400])
        pd.testing.assert_frame_equal(prefix, self.features.iloc[:400])
        self.assertFalse(any("target" in column for cols in self.groups.values() for column in cols))

    def test_supervised_fit_only_uses_strictly_mature_labels(self):
        cutoff = self.frame.Date.iloc[400]
        member = fit_paper(PaperSpec("test_ar", "ar", window=0), self.frame, self.groups, cutoff)
        self.assertLess(pd.Timestamp(member["diagnostic"]["latest_training_label_end"]), cutoff)
        changed = self.frame.copy()
        changed.loc[changed.target_end_date >= cutoff, "target_return"] = 999999
        alternative = fit_paper(PaperSpec("test_ar", "ar", window=0), changed, self.groups, cutoff)
        origins = self.frame.iloc[400:410]
        np.testing.assert_allclose(predict_paper(member, self.frame, origins), predict_paper(alternative, changed, origins))

    def test_arima_orders_stateful_forecast_and_future_invariance(self):
        cutoff = self.frame.Date.iloc[400]
        member = fit_paper(PaperSpec("css", "arima", window=252, order=(2, 1, 1)), self.frame, self.groups, cutoff)
        self.assertTrue(member["diagnostic"]["optimizer_success"])
        origins = self.frame.iloc[[400, 407, 411]]
        predictions = predict_paper(member, self.frame, origins)
        altered = self.frame.copy()
        altered.loc[altered.Date > origins.Date.max(), "Close"] *= 10
        np.testing.assert_allclose(predictions, predict_paper(member, altered, origins))
        np.testing.assert_allclose(predictions, predict_paper(member, self.frame, origins))
        dense = predict_paper(member, self.frame, self.frame.iloc[400:412])
        np.testing.assert_allclose(predictions, dense[[0, 7, 11]])

    def test_no_drift_random_walk_and_stationarity(self):
        state = fit_css_arima(self.prices.Close, order=(0, 1, 0), drift=False)
        self.assertEqual(state.forecast_return(21), 0.0)
        for reflection in [[0.8], [0.9, -0.9], [-0.9, 0.9]]:
            coefficients = _stationary_coefficients(np.array(reflection))
            roots = np.roots(np.r_[1, -coefficients])
            self.assertTrue((np.abs(roots) < 1).all())

    def test_holt_states_and_predictions_do_not_mutate(self):
        state = fit_damped_holt(self.prices.Close.iloc[:252], season=5, damping=0.98)
        before = deepcopy(state)
        self.assertTrue(np.isfinite(state.forecast_return(22)))
        self.assertEqual(state.last_log_price, before.last_log_price)
        np.testing.assert_array_equal(state.seasonal, before.seasonal)
        state.update_price(self.prices.Close.iloc[252])
        self.assertAlmostEqual(state.last_log_price, float(np.log(self.prices.Close.iloc[252])))

    def test_invalid_optimizer_fit_is_explicit(self):
        cutoff = self.frame.Date.iloc[400]
        member = fit_paper(PaperSpec("css", "arima", window=252), self.frame, self.groups, cutoff)
        member["diagnostic"]["optimizer_success"] = False
        with self.assertRaisesRegex(ValueError, "Invalid optimizer"):
            predict_paper(member, self.frame, self.frame.iloc[400:402])

    def test_binary_zero_convention_and_precision_recall(self):
        result = direction_regression_metrics([-0.2, -0.1, 0, 0.1, 0.2], [-0.1, 0, -0.4, 0.1, -0.1])
        self.assertEqual(result["rows"], 5)
        self.assertEqual(result["direction_rows"], 4)
        self.assertEqual(result["up_precision"], 0.5)
        self.assertEqual(result["up_recall"], 0.5)
        self.assertEqual(result["down_precision"], 0.5)
        self.assertEqual(result["down_recall"], 0.5)

    def test_paired_direction_blocks_zero_for_identical_models(self):
        actual = np.linspace(-0.2, 0.2, 120)
        predicted = np.sin(np.arange(120))*0.1
        result = direction_block_comparison(actual, predicted, predicted, block=30, draws=50)
        for values in result["differences"].values():
            self.assertEqual(values, {"measured": 0.0, "lower_95": 0.0, "upper_95": 0.0})


if __name__ == "__main__":
    unittest.main()
