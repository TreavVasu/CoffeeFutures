from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "Scripts" / "project" / "scripts"))

from integrate_news_scores_final_model import (  # noqa: E402
    BASE_PREDICTION,
    TARGET_RETURN,
    chronological_split,
    daily_news_features,
    fit_models,
    resolve_path,
    text_news_features,
    weekly_news_features,
)


class NewsIntegrationTests(unittest.TestCase):
    def setUp(self):
        # A missing Monday ensures all delays count price sessions, not weekdays.
        self.calendar = pd.bdate_range("2024-01-01", periods=90).delete(5)
        self.daily = pd.DataFrame({
            "trading_date": self.calendar,
            "event_count": 10.0,
            "bullish_events": 7.0,
            "bearish_events": 3.0,
            "signed_impact_score": 100.0,
        })
        self.weekly = pd.DataFrame({
            "period_id": ["2024-01-01", "2024-01-08"],
            "period_end": ["2024-01-07", "2024-01-14"],
            "event_count": [50, 40],
            "bullish_event_count": [30, 25],
            "bearish_event_count": [20, 15],
            "explicit_coffee_event_count": [5, 4],
            "mean_event_intensity_score_0_1": [0.5, 0.6],
            "deterministic_period_score_0_1": [0.7, 0.8],
            "top_event_digest": ["coffee drought", "ceasefire agreement"],
        })

    def test_news_is_unavailable_until_selection_returns_mature(self):
        original = daily_news_features(self.daily, self.calendar)
        changed = self.daily.copy()
        changed.loc[10:, "event_count"] = 1000
        actual = daily_news_features(changed, self.calendar)
        assert_frame_equal(original.iloc[:16], actual.iloc[:16])
        self.assertNotEqual(original.iloc[16, 1], actual.iloc[16, 1])
        self.assertTrue(original.iloc[:6, 1].isna().all())

    def test_price_derived_score_columns_are_ignored(self):
        daily = self.daily.copy()
        daily["signed_impact_score"] = np.arange(len(daily)) * 1e6
        daily["future_return_5d"] = 1e9
        weekly = self.weekly.copy()
        weekly["deterministic_period_score_0_1"] = -1e9
        assert_frame_equal(daily_news_features(daily, self.calendar), daily_news_features(self.daily, self.calendar))
        assert_frame_equal(weekly_news_features(weekly, self.calendar), weekly_news_features(self.weekly, self.calendar))

    def test_weekly_release_observes_session_calendar(self):
        features = weekly_news_features(self.weekly, self.calendar)
        # Friday Jan 5 + five return sessions + one release session = Tuesday Jan 16.
        self.assertEqual(features.iloc[0]["weekly_available_date"], pd.Timestamp("2024-01-16"))
        joined = pd.merge_asof(pd.DataFrame({"Date": self.calendar}), features,
                               left_on="Date", right_on="weekly_available_date")
        self.assertTrue(joined.loc[joined.Date < "2024-01-16", "period_id"].isna().all())

    def test_split_purges_overlapping_labels_even_with_missing_prediction_rows(self):
        frame = pd.DataFrame({"Date": self.calendar[:80], "target_available_date": self.calendar[5:85]})
        frame = frame.drop(index=[12, 14, 36])
        train, test = chronological_split(frame, 0.5)
        self.assertLess(train.target_available_date.max(), test.Date.min())
        self.assertTrue(set(train.Date).isdisjoint(set(test.Date)))
        with self.assertRaises(ValueError):
            chronological_split(frame.head(20), 0.5)

    def test_base_control_matches_calibration_with_uninformative_news(self):
        x = np.linspace(-0.02, 0.02, 40)
        train = pd.DataFrame({BASE_PREDICTION: x, TARGET_RETURN: 0.003 + 0.8 * x, "news": 1.0})
        models, specs = fit_models(train, ["news"])
        np.testing.assert_allclose(
            models["calibrated_base"].predict(train[specs["calibrated_base"]]),
            models["final_news_blend"].predict(train[specs["final_news_blend"]]), atol=1e-12,
        )

    def test_duplicate_news_dates_and_outside_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            daily_news_features(pd.concat([self.daily, self.daily.iloc[:1]]), self.calendar)
        with self.assertRaises(ValueError):
            resolve_path("../outside.csv")

    def test_text_scores_ignore_embedded_prices_and_match_whole_words(self):
        features = text_news_features(pd.Series([
            "Coffee drought score=0.1 future_return=0.2", "Coffee drought score=0.9 future_return=-0.8",
            "reward warm coffeehouse", "ceasefire agreement",
        ]))
        np.testing.assert_array_equal(features.iloc[0], features.iloc[1])
        self.assertEqual(features.iloc[2].sum(), 0)
        self.assertGreater(features.iloc[3]["news_weekly_text_support"], 0)


if __name__ == "__main__":
    unittest.main()
