from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "MonthFu/src"))

from external_return_features import (  # noqa: E402
    _prior_seasonal_zscore, build_daily_news_features,
    build_weather_features, build_weekly_news_features,
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

    def test_heavy_weather_features_preserve_prefix_availability(self):
        baseline, _ = build_weather_features(self.weather, self.calendar)
        cutoff = pd.Timestamp("2020-02-01")
        partial = self.weather.loc[self.weather.Date <= cutoff]
        rebuilt, _ = build_weather_features(partial, self.calendar)
        early = self.calendar.Date <= cutoff + pd.Timedelta(days=5)
        assert_frame_equal(baseline.loc[early], rebuilt.loc[early], check_exact=True)
        self.assertIn("weather_brazil_minas_gerais_dry_heat_stress_30d", baseline)
        self.assertIn("weather_brazil_minas_gerais_rain_28d", baseline)

    def test_missing_weather_day_interrupts_dry_spell(self):
        rain = pd.Series([0.0, 0.0, np.nan, 0.0, 0.0, 5.0, 0.0])
        from external_return_features import _dry_spell
        result = _dry_spell(rain)
        self.assertEqual(result.tolist()[:2], [1.0, 2.0])
        self.assertTrue(np.isnan(result.iloc[2]))
        self.assertEqual(result.tolist()[3:], [1.0, 2.0, 0.0, 1.0])

    def test_news_lag_uses_observed_sessions_and_ignores_future_scores(self):
        original = build_daily_news_features(self.daily, self.calendar)
        changed = self.daily.copy()
        changed.loc[10:, "event_count"] = 1000
        result = build_daily_news_features(changed, self.calendar)
        assert_frame_equal(original.iloc[:16], result.iloc[:16])
        self.assertNotEqual(original.iloc[16]["news_daily_event_count_log_5s"], result.iloc[16]["news_daily_event_count_log_5s"])
        self.daily["future_return_5d"] = 1e8
        self.daily["signed_impact_score"] = -1e8
        assert_frame_equal(original, build_daily_news_features(self.daily, self.calendar))

    def test_weekly_news_release_and_stale_handling(self):
        result = build_weekly_news_features(self.weekly, self.calendar)
        # Jan 3 is source week's final session, Jan 8 is missing: six sessions later is Jan 14.
        column = "news_weekly_event_count_log"
        self.assertTrue(result.loc[result.Date < "2020-01-14", column].isna().all())
        self.assertTrue(result.loc[result.Date.eq("2020-01-14"), column].notna().all())
        self.assertTrue(result.loc[result.Date > "2020-02-16", column].isna().all())
        changed = self.weekly.copy()
        changed["deterministic_period_score_0_1"] = 1e9
        changed["top_event_digest"] = changed.top_event_digest.str.replace("score=0.9", "score=-1000 future_return=0.99")
        assert_frame_equal(result, build_weekly_news_features(changed, self.calendar))


if __name__ == "__main__":
    unittest.main()
