from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MonthFu/src"))

from features import _feature_groups, interaction_features, load_prices, price_features  # noqa: E402


def synthetic_prices(rows=500):
    rng = np.random.default_rng(13)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, rows)))
    opening = close * np.exp(rng.normal(0, 0.005, rows))
    return pd.DataFrame({"Date": pd.bdate_range("2015-01-01", periods=rows),
                         "Open": opening, "High": np.maximum(opening, close) * 1.01,
                         "Low": np.minimum(opening, close) * 0.99, "Close": close,
                         "Volume": rng.integers(1000, 100000, rows).astype(float),
                         "price_invalid_ohlc": 0.0, "price_low_volume": 0.0})


class FeatureTests(unittest.TestCase):
    def test_future_ohlcv_mutation_does_not_change_earlier_features(self):
        prices = synthetic_prices()
        changed = prices.copy()
        changed.loc[300:, ["Open", "High", "Low", "Close"]] *= 10
        changed.loc[300:, "Volume"] *= 100
        original, mutation = price_features(prices), price_features(changed)
        assert_frame_equal(original.iloc[:300], mutation.iloc[:300], check_exact=True)
        self.assertNotEqual(original.iloc[300].price_return_21, mutation.iloc[300].price_return_21)

    def test_prefix_rebuild_identical_to_full_history(self):
        prices = synthetic_prices()
        full = price_features(prices)
        assert_frame_equal(price_features(prices.iloc[:300]), full.iloc[:300], check_exact=True)

    def test_monthly_returns_and_gaps_count_observed_sessions(self):
        prices = synthetic_prices().drop(index=[10, 20]).reset_index(drop=True)
        result = price_features(prices)
        for window in [21, 28, 30]:
            expected = prices.Close.iloc[100] / prices.Close.iloc[100-window] - 1
            self.assertAlmostEqual(result.iloc[100][f"price_return_{window}"], expected)
        self.assertEqual(result.iloc[10].price_session_gap_days,
                         (prices.Date.iloc[10] - prices.Date.iloc[9]).days)
        self.assertTrue(result.price_return_30.iloc[:30].isna().all())

    def test_price_quality_masking_and_source_field_whitelist(self):
        prices = synthetic_prices(100).drop(columns=["price_invalid_ohlc", "price_low_volume"])
        prices.loc[3, "High"] = 1
        prices.loc[4, "Volume"] = 1
        prices.loc[5, "Close"] = -2
        prices["target_return_30"] = 99999
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            prices.to_csv(path, index=False)
            cleaned, audit = load_prices(path)
        self.assertEqual(audit["rejected_rows"], 1)
        self.assertTrue(cleaned.loc[cleaned.Date.eq(prices.Date.iloc[3]), ["Open", "High", "Low"]].isna().all(axis=None))
        self.assertTrue(cleaned.loc[cleaned.Date.eq(prices.Date.iloc[4]), "Volume"].isna().all())
        self.assertNotIn("target_return_30", cleaned)

    def test_duplicate_session_dates_rejected(self):
        prices = synthetic_prices(100)
        prices.loc[10, "Date"] = prices.Date.iloc[9]
        with self.assertRaisesRegex(ValueError, "unique increasing"):
            price_features(prices)

    def test_default_groups_exclude_news_and_targets(self):
        prices = synthetic_prices()
        market = price_features(prices)
        cot = pd.DataFrame({"Date": prices.Date, "cot_legacy_noncommercial_net_oi": 0.1,
                            "cot_legacy_noncommercial_net_change_4w": 0.01})
        external = pd.DataFrame({"Date": prices.Date, "weather_brazil_minas_gerais_rain_30d": 50,
                                 "news_daily_event_count_log_21s": 1.0})
        frame = prices.drop(columns=["price_invalid_ohlc", "price_low_volume"]).merge(market, on="Date").merge(cot, on="Date").merge(external, on="Date")
        interactions = interaction_features(frame)
        frame = pd.concat([frame, interactions], axis=1)
        frame["target_return"] = 1e8
        groups = _feature_groups(frame, market, cot, external, interactions)
        for name, columns in groups.items():
            self.assertNotIn("target_return", columns)
            self.assertEqual(len(columns), len(set(columns)))
            self.assertFalse(any(column.startswith("news_") for column in columns))
        self.assertIn("Open", groups["previous_technical"])
        self.assertFalse(any(c in {"Open", "High", "Low", "Close", "Volume"} for c in groups["previous_history"]))
        self.assertNotIn("Open", groups["engineered"])
        self.assertIn("price_return_21", groups["monthly_core"])
        self.assertIn("interaction_legacy_noncommercial_position_trend_21", groups["monthly_core"])


if __name__ == "__main__":
    unittest.main()
