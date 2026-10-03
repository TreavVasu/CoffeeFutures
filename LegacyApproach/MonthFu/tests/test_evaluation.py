"""Behavioral checks for monthly label maturity and chronological evaluation."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import modeling
from features import price_features


class MonthlyTargetTests(unittest.TestCase):
    def setUp(self):
        # The deliberately irregular observed calendar includes a holiday/data gap.
        self.prices = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05",
                                    "2024-01-08", "2024-01-09", "2024-01-11",
                                    "2024-01-16"]),
            "Close": [100., 110., 105., 120., 125., 130., 150.],
        })

    def test_calendar_target_uses_next_observed_close(self):
        result = modeling.add_targets(self.prices, horizon=7, unit="calendar")
        self.assertEqual(result.target_end_date.iloc[0], pd.Timestamp("2024-01-09"))
        # Jan 10 has no observation: use Jan 11, rather than interpolate a price.
        self.assertEqual(result.target_end_date.iloc[1], pd.Timestamp("2024-01-11"))
        self.assertAlmostEqual(result.target_return.iloc[1], 130./110.-1)
        self.assertTrue(result.target_return.iloc[-1:].isna().all())
        self.assertTrue(result.target_end_date.iloc[-1:].isna().all())
        self.assertEqual(result.Close.tolist(), self.prices.Close.tolist())

    def test_sessions_count_observations_rather_than_calendar_days(self):
        result = modeling.add_targets(self.prices, horizon=3, unit="sessions")
        self.assertEqual(result.target_end_date.iloc[0], pd.Timestamp("2024-01-08"))
        self.assertEqual(result.target_end_date.iloc[3], pd.Timestamp("2024-01-16"))
        np.testing.assert_allclose(result.target_return.iloc[:4],
                                   np.array([120./100., 125./110., 130./105., 150./120.])-1)
        self.assertTrue(result.target_return.iloc[-3:].isna().all())

    def test_invalid_or_conflicting_calendar_is_rejected(self):
        for horizon, unit in [(0, "calendar"), (-1, "sessions"), (7, "weeks")]:
            with self.subTest(horizon=horizon, unit=unit), self.assertRaises(ValueError):
                modeling.add_targets(self.prices, horizon, unit)
        for invalid in [self.prices.iloc[::-1], pd.concat([self.prices, self.prices.tail(1)])]:
            with self.assertRaises(ValueError):
                modeling.add_targets(invalid, 7, "calendar")


class PurgedEvaluationTests(unittest.TestCase):
    def test_boundary_label_at_cutoff_is_not_available_for_fitting(self):
        cutoff = pd.Timestamp("2024-02-01")
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-02-01"]),
            "target_end_date": pd.to_datetime(["2024-01-31", "2024-02-01", None, "2024-02-03"]),
            "target_return": [.1, .2, np.nan, .3],
        })
        train = modeling.mature_training_rows(frame, cutoff)
        self.assertEqual(train.Date.tolist(), [pd.Timestamp("2024-01-01")])

    def test_each_fold_purges_actual_calendar_target_endpoints(self):
        dates = pd.bdate_range("2019-01-02", periods=250)
        dates = dates.delete(np.arange(95, 106))
        prices = pd.DataFrame({"Date": dates, "Close": 100+np.arange(len(dates), dtype=float)})
        frame = modeling.add_targets(prices, horizon=28, unit="calendar")
        cutoff = dates[210]
        folds = modeling.expanding_folds(frame, cutoff, splits=3, test_size=25, min_train=50)
        self.assertEqual(len(folds), 3)
        previous_train_size = 0
        previous_valid_end = pd.Timestamp.min
        for train, valid in folds:
            self.assertTrue(train.target_end_date.lt(valid.Date.min()).all())
            self.assertTrue(valid.target_end_date.lt(cutoff).all())
            self.assertEqual(len(valid), 25)
            self.assertGreater(len(train), previous_train_size)
            self.assertGreater(valid.Date.min(), previous_valid_end)
            previous_train_size, previous_valid_end = len(train), valid.Date.max()

    def test_recent_training_window_respects_both_age_and_maturity(self):
        dates = pd.bdate_range("2020-01-01", "2024-01-01")
        frame = modeling.add_targets(pd.DataFrame({"Date": dates, "Close": np.arange(len(dates))+100}), 30, "calendar")
        cutoff = pd.Timestamp("2023-12-01")
        train = modeling.mature_training_rows(frame, cutoff, train_years=1)
        self.assertGreaterEqual(train.Date.min(), pd.Timestamp("2022-12-01"))
        self.assertLess(train.target_end_date.max(), cutoff)

    def test_insufficient_history_does_not_produce_unpurged_fallback(self):
        dates = pd.bdate_range("2024-01-01", periods=80)
        frame = modeling.add_targets(pd.DataFrame({"Date": dates, "Close": np.arange(80)+100}), 30, "calendar")
        with self.assertRaises(ValueError):
            modeling.expanding_folds(frame, dates[-1], splits=3, test_size=20, min_train=30)

    def test_nonoverlap_sampling_uses_real_dates_on_irregular_calendar(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01", "2024-01-04", "2024-01-11",
                                    "2024-01-15", "2024-01-23", "2024-01-29"]),
            "target_end_date": pd.to_datetime(["2024-01-11", "2024-01-15", "2024-01-23",
                                               "2024-01-29", "2024-02-02", "2024-02-06"]),
        })
        positions = modeling.nonoverlap_positions(frame, offset=0)
        sample = frame.iloc[positions]
        self.assertEqual(sample.Date.iloc[0], pd.Timestamp("2024-01-01"))
        self.assertGreaterEqual(len(sample), 2)
        for previous_end, next_origin in zip(sample.target_end_date.iloc[:-1], sample.Date.iloc[1:]):
            self.assertGreaterEqual(next_origin, previous_end)


class FeatureIsolationTests(unittest.TestCase):
    def test_price_features_cannot_change_when_future_prices_are_perturbed(self):
        dates = pd.bdate_range("2020-01-01", periods=330)
        close = 100+np.sin(np.arange(330)/6)+np.arange(330)*.02
        prices = pd.DataFrame({"Date": dates, "Close": close, "Open": close-.1,
                               "High": close+.4, "Low": close-.5,
                               "Volume": np.arange(330)+1000.,
                               "price_invalid_ohlc": 0., "price_low_volume": 0.})
        original = price_features(prices)
        changed = prices.copy()
        changed.loc[271:, ["Close", "Open", "High", "Low", "Volume"]] *= 1000
        recomputed = price_features(changed)
        pd.testing.assert_frame_equal(original.iloc[:271], recomputed.iloc[:271])

    def test_feature_screening_and_imputation_use_training_rows_only(self):
        train = pd.DataFrame({"x": [1., np.nan, 3., 4., 5.],
                              "future_only": [np.nan]*5,
                              "constant": [7.]*5,
                              "target_return": [.01, .02, -.01, .03, -.02]})
        member = modeling.fit_candidate(modeling.Candidate("isolation_ridge", "price", "ridge", 10),
                                        train, ["x", "future_only", "constant"])
        self.assertEqual(member["features"], ["x"])
        np.testing.assert_allclose(member["estimator"].named_steps["impute"].statistics_, [3.5])
        future = pd.DataFrame({"x": [np.nan], "future_only": [999999.], "constant": [-999999.]})
        prediction = modeling.predict_member(member, future)
        future[["future_only", "constant"]] *= -1000
        np.testing.assert_allclose(prediction, modeling.predict_member(member, future))


class DependenceAndMetricTests(unittest.TestCase):
    def test_invalid_metric_and_bootstrap_pairs_fail_without_silent_dropping(self):
        for actual, prediction in [([], []), ([.1], [.1, .2]), ([np.nan], [0.]), ([.1], [np.inf])]:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                modeling.metrics(actual, prediction)
        for block, draws in [(0, 10), (10, 0)]:
            with self.assertRaises(ValueError):
                modeling.block_comparison([.1], [0.], [0.], block=block, draws=draws)
        with self.assertRaises(ValueError):
            modeling.block_comparison([.1], [0.], [np.nan], block=1, draws=10)

    def test_zero_prediction_has_neutral_direction_and_zero_reference_skill(self):
        result = modeling.metrics([.1, -.1, 0.], [0., 0., 0.])
        self.assertAlmostEqual(result["direction_accuracy"], 1/3)
        self.assertAlmostEqual(result["r2_vs_zero"], 0.)

    def test_zero_relative_skill_is_scale_invariant_for_horizon_comparison(self):
        actual = np.array([.03, -.02, .06, -.04])
        prediction = actual*.4
        first = modeling.metrics(actual, prediction)
        scaled = modeling.metrics(actual*2, prediction*2)
        self.assertAlmostEqual(first["r2_vs_zero"], scaled["r2_vs_zero"])
        self.assertAlmostEqual(first["rmse"]*2, scaled["rmse"])

    def test_paired_block_bootstrap_identical_models_has_exact_zero_difference(self):
        actual = np.sin(np.arange(150)/8)*.05
        prediction = np.cos(np.arange(150)/8)*.02
        result = modeling.block_comparison(actual, prediction, prediction, block=40, draws=100)
        self.assertEqual(result["rmse_difference"], 0.)
        np.testing.assert_equal(result["rmse_difference_95pct_block_interval"], [0., 0.])
        self.assertEqual(result["block_sessions"], 40)

    def test_paired_block_bootstrap_sign_reports_selected_improvement(self):
        actual = np.full(150, .05)
        result = modeling.block_comparison(actual, actual, np.zeros(150), block=40, draws=100)
        self.assertAlmostEqual(result["rmse_difference"], -.05)
        np.testing.assert_allclose(result["rmse_difference_95pct_block_interval"], [-.05, -.05])


class CausalStateTests(unittest.TestCase):
    def test_holt_winters_predictions_do_not_depend_on_later_prices_or_mutate_bundle(self):
        dates = pd.bdate_range("2023-01-02", periods=190)
        prices = pd.DataFrame({"Date": dates, "Close": 120+np.sin(np.arange(190)/9)*3,
                               "forecast_steps": 21.})
        member = modeling.fit_candidate(
            modeling.Candidate("previous_hw", "previous_technical", "holt_winters"),
            prices.iloc[:170], [], price_history=prices.iloc[:170])
        evaluation = prices.iloc[170:].copy()
        original_state = member["state"].seasonal.copy()
        prediction = modeling.predict_member(member, evaluation)
        changed = evaluation.copy()
        changed.iloc[10:, changed.columns.get_loc("Close")] *= 10
        alternative = modeling.predict_member(member, changed)
        np.testing.assert_allclose(prediction[:10], alternative[:10])
        np.testing.assert_allclose(prediction, modeling.predict_member(member, evaluation))
        np.testing.assert_equal(original_state, member["state"].seasonal)

    def test_holt_winters_cannot_replay_dates_before_fitted_state(self):
        dates = pd.bdate_range("2023-01-02", periods=40)
        frame = pd.DataFrame({"Date": dates, "Close": 100+np.arange(40)*.01,
                              "forecast_steps": 21.})
        member = modeling.fit_candidate(
            modeling.Candidate("previous_hw", "previous_technical", "holt_winters"),
            frame, [], price_history=frame)
        with self.assertRaises(ValueError):
            modeling.predict_member(member, frame.iloc[:1])


if __name__ == "__main__":
    unittest.main()
