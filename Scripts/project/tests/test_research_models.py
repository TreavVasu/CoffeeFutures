from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "Scripts" / "project" / "src"))

from research_models import (  # noqa: E402
    ExtremeLearningMachineRegressor,
    ThresholdAutoregressiveRegressor,
    fit_arima111,
    fit_garch11,
    fit_holt_winters,
    fit_simple_exp_smoothing,
    rolling_polynomial_return_forecast,
    walk_forward_garch_forecast,
    walk_forward_return_forecast,
)


class ResearchModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(7)
        cls.features = rng.normal(size=(360, 8))
        cls.target = 0.02 * cls.features[:, 0] - 0.01 * cls.features[:, 2] + rng.normal(
            scale=0.004, size=360
        )
        returns = rng.normal(loc=0.0003, scale=0.012, size=440)
        cls.prices = 120.0 * np.exp(np.cumsum(returns))

    def test_elm_is_sklearn_compatible_and_finite(self) -> None:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ExtremeLearningMachineRegressor(n_hidden=48, alpha=2.0, random_state=4),
                ),
            ]
        )
        model.fit(self.features[:300], self.target[:300])
        prediction = model.predict(self.features[300:])
        self.assertEqual(prediction.shape, (60,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_threshold_model_fits_both_regimes(self) -> None:
        model = ThresholdAutoregressiveRegressor(
            alpha=1.0,
            min_regime_size=40,
            threshold_quantiles=(0.60, 0.75),
        )
        model.fit(self.features, self.target)
        prediction = model.predict(self.features[-20:])
        self.assertTrue(np.isfinite(prediction).all())
        self.assertGreater(model.threshold_, 0)

    def test_price_models_produce_finite_horizon_forecasts(self) -> None:
        train = self.prices[:320]
        context = self.prices[320:325]
        evaluation = self.prices[325:350]
        for kind in ["arima_111", "simple_exp_smoothing", "holt_winters"]:
            _, prediction = walk_forward_return_forecast(
                kind, train, context, evaluation, horizon=5
            )
            self.assertEqual(prediction.shape, (25,))
            self.assertTrue(np.isfinite(prediction).all())

    def test_polynomial_trends_are_finite(self) -> None:
        for degree in [1, 2]:
            prediction = rolling_polynomial_return_forecast(
                self.prices[:325], self.prices[325:350], horizon=5, degree=degree
            )
            self.assertTrue(np.isfinite(prediction).all())

    def test_garch_is_stationary_and_finite(self) -> None:
        state = fit_garch11(self.prices[:350])
        self.assertLess(state.alpha + state.beta, 0.999)
        _, prediction = walk_forward_garch_forecast(
            self.prices[:350], self.prices[350:355], self.prices[355:380], horizon=5
        )
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction > 0).all())

    def test_direct_fits_return_valid_state(self) -> None:
        states = [
            fit_arima111(self.prices),
            fit_simple_exp_smoothing(self.prices),
            fit_holt_winters(self.prices),
        ]
        for state in states:
            self.assertTrue(np.isfinite(state.forecast_return(5)))


if __name__ == "__main__":
    unittest.main()
