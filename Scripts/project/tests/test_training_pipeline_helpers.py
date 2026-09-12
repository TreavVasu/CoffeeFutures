from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "Scripts" / "project" / "scripts"))

from train_research_paper_ensemble import (  # noqa: E402
    XGBRegressor,
    add_local_regime_features,
    assign_regimes,
    fit_regime_definitions,
)


class TrainingPipelineHelperTests(unittest.TestCase):
    @staticmethod
    def make_frame(rows: int = 900) -> pd.DataFrame:
        rng = np.random.default_rng(19)
        index = np.arange(rows, dtype=float)
        daily_return = rng.normal(0.0002, 0.012, rows)
        frame = pd.DataFrame(
            {
                "Date": pd.bdate_range("2018-01-01", periods=rows),
                "Close": 120.0 * np.exp(np.cumsum(daily_return)),
                "return_20d": 0.025 * np.sin(index / 34.0) + rng.normal(0, 0.01, rows),
                "volatility_20": 0.018 + 0.008 * (1.0 + np.sin(index / 57.0)),
                "volume_vs_ma_20": 1.0 + 0.3 * np.cos(index / 23.0),
                "npi_z_3y": 1.4 * np.sin(index / 71.0),
                "npi_abs_z_3y": np.abs(1.4 * np.sin(index / 71.0)),
                "weather_brazil_minas_gerais_temperature_2m_max_roll20": 28 + 3 * np.sin(index / 80.0),
                "weather_brazil_minas_gerais_vapour_pressure_deficit_max_roll20": 1.2 + 0.4 * np.sin(index / 67.0),
                "weather_brazil_minas_gerais_soil_moisture_0_to_100cm_mean_roll20": 0.32 + 0.05 * np.cos(index / 75.0),
                "weather_brazil_minas_gerais_precipitation_sum_roll20": 4.0 + 2.0 * np.cos(index / 61.0),
            }
        )
        return frame

    def test_xgboost_import_is_available(self) -> None:
        self.assertIsNotNone(XGBRegressor)

    def test_regime_features_do_not_read_future_rows(self) -> None:
        original = self.make_frame()
        changed_future = original.copy()
        changed_future.loc[changed_future.index[-20:], "volatility_20"] *= 50.0
        changed_future.loc[changed_future.index[-20:], "return_20d"] += 2.0

        first = add_local_regime_features(original)
        second = add_local_regime_features(changed_future)
        regime_columns = [column for column in first.columns if column.startswith("regime_")]
        assert_frame_equal(
            first.loc[first.index[:-20], regime_columns],
            second.loc[second.index[:-20], regime_columns],
        )

    def test_regime_thresholds_are_calibrated_and_assignable(self) -> None:
        frame = add_local_regime_features(self.make_frame())
        calibration = frame.iloc[:700].copy()
        holdout = frame.iloc[700:].copy()
        definitions = fit_regime_definitions(calibration)
        assignments = assign_regimes(holdout, definitions)

        expected = {
            "volatility_regime",
            "trend_regime",
            "cot_positioning_regime",
            "liquidity_regime",
            "brazil_weather_regime",
        }
        self.assertTrue(expected.issubset(definitions))
        self.assertEqual(len(assignments), len(holdout))
        self.assertFalse(assignments["volatility_regime"].eq("unavailable").all())
        self.assertIn("trend_volatility_regime", assignments.columns)


if __name__ == "__main__":
    unittest.main()
