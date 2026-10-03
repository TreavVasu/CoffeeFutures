from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Scripts/project/src"))

from external_return_features import (  # noqa: E402
    _prior_seasonal_zscore, build_external_features, build_weather_features,
)


class ExternalReturnFeatureTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.DataFrame({"Date": pd.bdate_range("2020-01-01", periods=100).delete(5)})
        dates = pd.date_range("2016-01-01", "2020-08-01")
        self.weather = pd.DataFrame({"Date": dates})
        values = {
            "temperature_2m_mean": 20.0, "temperature_2m_max": 33.0,
            "temperature_2m_min": 15.0, "precipitation_sum": 4.0,
            "soil_moisture_0_to_100cm_mean": 0.3,
            "vapour_pressure_deficit_max": 2.0, "et0_fao_evapotranspiration": 3.0,
        }
        for variable, value in values.items():
            self.weather[f"weather_brazil_minas_gerais_{variable}"] = value + 0.01 * np.sin(np.arange(len(dates)))
        self.daily = pd.DataFrame({"trading_date": self.calendar.Date, "event_count": 10.0,
                                   "bullish_events": 6.0, "bearish_events": 4.0})
        self.weekly = pd.DataFrame({"period_end": ["2020-01-05", "2020-01-12"],
                                    "event_count": [50, 50], "bullish_event_count": [30, 30],
                                    "bearish_event_count": [20, 20], "explicit_coffee_event_count": [3, 3],
                                    "mean_event_intensity_score_0_1": [0.6, 0.6],
                                    "top_event_digest": ["coffee drought score=0.9", "coffee recovery"]})

    def test_weather_change_cannot_arrive_before_five_calendar_days(self):
        changed = self.weather.copy()
        mask = changed.Date >= pd.Timestamp("2020-01-10")
        changed.loc[mask, "weather_brazil_minas_gerais_temperature_2m_mean"] += 20
        before, _ = build_weather_features(self.weather, self.calendar)
        after, _ = build_weather_features(changed, self.calendar)
        assert_frame_equal(before.loc[before.Date < "2020-01-15"], after.loc[after.Date < "2020-01-15"])
        column = "weather_brazil_minas_gerais_temp_7d"
        self.assertNotEqual(before.loc[before.Date.eq("2020-01-15"), column].iloc[0],
                            after.loc[after.Date.eq("2020-01-15"), column].iloc[0])

    def test_weather_climatology_never_uses_later_days_or_current_month(self):
        dates = pd.date_range("2010-01-01", "2020-12-31")
        original = pd.Series(10 + np.sin(np.arange(len(dates))), index=dates)
        changed = original.copy()
        changed.loc["2020-01-16":] = 10000
        baseline, mutation = _prior_seasonal_zscore(original), _prior_seasonal_zscore(changed)
        assert_series_equal(baseline.loc[:"2020-01-15"], mutation.loc[:"2020-01-15"])
        self.assertTrue(baseline.loc[:"2011-12-31"].isna().all())
        self.assertTrue(baseline.loc["2012-01-01":].notna().all())

    def test_weather_stale_gap_is_missing_without_backfill(self):
        weather = self.weather.loc[self.weather.Date <= "2020-01-01"]
        result, _ = build_weather_features(weather, self.calendar)
        column = "weather_brazil_minas_gerais_temp_7d"
        self.assertTrue(result.loc[result.Date > "2020-01-13", column].isna().all())
        self.assertTrue(result.loc[result.Date > "2020-01-13", "weather_brazil_minas_gerais_stale"].eq(1).all())
        early_calendar = pd.DataFrame({"Date": pd.bdate_range("2015-12-25", "2016-01-20")})
        result, _ = build_weather_features(self.weather, early_calendar)
        self.assertTrue(result.loc[result.Date < "2016-01-06", column].isna().all())

    def test_news_layer_is_fully_removed(self):
        """The GDELT layer is gone: no builders, constants or feature columns."""
        module = sys.modules["external_return_features"]
        for removed in ["build_daily_news_features", "build_weekly_news_features",
                        "_text_news_features", "DAILY_NEWS_COLUMNS", "WEEKLY_NEWS_COLUMNS",
                        "NEWS_LAG_SESSIONS", "NEWS_MAX_SOURCE_AGE_DAYS"]:
            self.assertFalse(hasattr(module, removed), f"{removed} should be removed")
        frame, metadata = build_external_features(
            pd.DataFrame({"Date": self.calendar.Date, "Close": 100.0}), ROOT)
        self.assertFalse([c for c in frame.columns if c.startswith("news_")])
        self.assertNotIn("news", metadata)
        self.assertIn("news", metadata["excluded_sources"])


if __name__ == "__main__":
    unittest.main()
