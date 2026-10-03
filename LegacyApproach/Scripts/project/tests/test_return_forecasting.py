from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Scripts" / "project" / "src"))

from return_forecasting import (  # noqa: E402
    Candidate,
    add_targets,
    block_comparison,
    expanding_folds,
    fit_candidate,
    load_prices,
    mature_training_rows,
    metrics,
    predict_member,
    price_features,
)


class ReturnTargetTests(unittest.TestCase):
    @staticmethod
    def irregular_prices(rows: int = 100) -> pd.DataFrame:
        dates = pd.bdate_range("2024-01-05", periods=rows + 3)
        dates = dates.difference(pd.to_datetime(["2024-01-15", "2024-02-05", "2024-02-19"]))
        return pd.DataFrame({"Date": dates[:rows], "Close": np.arange(rows, dtype=float) + 100})

    def test_thirty_session_targets_replace_stale_labels_and_keep_unlabeled_tail(self) -> None:
        prices = self.irregular_prices()
        prices["target_return"] = 99.0
        prices["target_close"] = -1.0
        prices["target_end_date"] = pd.Timestamp("2099-01-01")
        result = add_targets(prices, horizon=30, unit="sessions")

        self.assertAlmostEqual(result.target_return.iloc[0], 0.30)
        self.assertEqual(result.target_end_date.iloc[0], prices.Date.iloc[30])
        self.assertEqual(result.target_close.iloc[69], prices.Close.iloc[99])
        self.assertEqual(result.target_return.notna().sum(), 70)
        self.assertTrue(result.iloc[-30:][["target_return", "target_close", "target_end_date"]].isna().all().all())
        self.assertEqual(len(result), len(prices))
        self.assertTrue(prices.target_return.eq(99).all(), "Label construction must not mutate its source.")

    def test_calendar_horizon_uses_first_observed_session_on_or_after_day_thirty(self) -> None:
        prices = self.irregular_prices()
        result = add_targets(prices, horizon=30, unit="calendar")

        # Jan 5 + 30 days is Sunday Feb 4; this fixture also omits Monday Feb 5.
        self.assertEqual(result.target_end_date.iloc[0], pd.Timestamp("2024-02-06"))
        end_close = prices.loc[prices.Date.eq("2024-02-06"), "Close"].iloc[0]
        self.assertAlmostEqual(result.target_return.iloc[0], end_close / 100 - 1)
        observed = result.target_end_date.notna()
        self.assertTrue((result.loc[observed, "target_end_date"] >= result.loc[observed, "Date"] + pd.Timedelta(days=30)).all())
        self.assertTrue(result.loc[prices.Date + pd.Timedelta(days=30) > prices.Date.max(), "target_return"].isna().all())
        self.assertNotEqual(result.target_end_date.iloc[0], add_targets(prices, 30, "sessions").target_end_date.iloc[0])

    def test_target_builder_rejects_ambiguous_calendar_and_invalid_horizon(self) -> None:
        prices = self.irregular_prices()
        for frame in [prices.iloc[::-1], pd.concat([prices, prices.iloc[-1:]])]:
            with self.assertRaises(ValueError):
                add_targets(frame)
        for horizon, unit in [(0, "sessions"), (-1, "calendar"), (30, "weeks")]:
            with self.assertRaises(ValueError):
                add_targets(prices, horizon, unit)

    def test_each_fold_excludes_training_labels_ending_on_validation_start(self) -> None:
        dates = pd.bdate_range("2015-01-01", periods=600).delete([7, 100, 200, 300])
        raw = pd.DataFrame({"Date": dates, "Close": 100 + np.arange(len(dates)) * .1})
        for unit in ["sessions", "calendar"]:
            with self.subTest(unit=unit):
                frame = add_targets(raw, horizon=30, unit=unit)
                holdout_start = frame.Date.iloc[520]
                folds = expanding_folds(frame, holdout_start, splits=3, test_size=60, min_train=150)
                self.assertEqual(len(folds), 3)
                seen_validation = []
                for train, valid in folds:
                    boundary = valid.Date.iloc[0]
                    self.assertTrue(train.target_end_date.lt(boundary).all())
                    self.assertTrue(train.Date.lt(boundary).all())
                    self.assertTrue(valid.target_end_date.lt(holdout_start).all())
                    self.assertEqual(len(valid), 60)
                    self.assertFalse(train.index.isin(frame.index[frame.target_end_date.eq(boundary)]).any())
                    seen_validation.extend(valid.Date.tolist())
                self.assertEqual(len(seen_validation), len(set(seen_validation)))

    def test_mature_rows_require_both_observation_and_label_before_cutoff(self) -> None:
        cutoff = pd.Timestamp("2024-04-01")
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2020-01-01", "2024-03-01", "2024-03-02", "2024-03-03", "2024-04-02"]),
            "target_end_date": pd.to_datetime(["2020-02-01", "2024-03-28", "2024-04-01", None, "2024-03-29"]),
            "target_return": [.1, .2, .3, np.nan, .4],
        })
        self.assertEqual(mature_training_rows(frame, cutoff).index.tolist(), [0, 1])
        self.assertEqual(mature_training_rows(frame, cutoff, train_years=1).index.tolist(), [1])


class ReturnFeatureTests(unittest.TestCase):
    @staticmethod
    def prices(rows: int = 420) -> pd.DataFrame:
        rng = np.random.default_rng(27)
        close = 100 * np.exp(np.cumsum(rng.normal(0, .012, rows)))
        return pd.DataFrame({
            "Date": pd.bdate_range("2020-01-01", periods=rows),
            "Close": close, "Open": close * .995, "High": close * 1.015,
            "Low": close * .98, "Volume": rng.integers(100, 5000, rows).astype(float),
            "price_invalid_ohlc": 0.0, "price_low_volume": 0.0,
        })

    def test_future_price_and_volume_changes_cannot_change_existing_features(self) -> None:
        prices = self.prices()
        cutoff = 330
        changed = prices.copy()
        changed.loc[cutoff:, ["Close", "Open", "High", "Low"]] *= 17
        changed.loc[cutoff:, "Volume"] *= 100

        original_features = price_features(prices)
        assert_frame_equal(original_features.iloc[:cutoff], price_features(changed).iloc[:cutoff])
        assert_frame_equal(original_features.iloc[:cutoff], price_features(prices.iloc[:cutoff]))
        self.assertFalse(any(c.startswith("target_") for c in original_features))

    def test_feature_screening_and_imputation_use_training_rows_only(self) -> None:
        train = pd.DataFrame({
            "signal": [np.nan, 1., 2., 3., 4., 5., 6., 7., 8., 9.],
            "constant_before_validation": 5.0,
            "available_only_after_training": np.nan,
            "too_sparse": [1.] + [np.nan] * 9,
            "target_return": np.linspace(-.03, .03, 10),
        })
        future = pd.DataFrame({
            "signal": [1e8, np.nan], "constant_before_validation": [0., 1.],
            "available_only_after_training": [100., 200.], "too_sparse": [2., 3.],
        })
        columns = future.columns.tolist()
        member = fit_candidate(Candidate("test_ridge", "price", "ridge", 10), train, columns)
        self.assertEqual(member["features"], ["signal"])
        self.assertEqual(member["estimator"].named_steps["impute"].statistics_.tolist(), [5.])
        predictions = predict_member(member, future)
        self.assertTrue(np.isfinite(predictions).all())
        future.loc[:, columns[1:]] = -1e12
        np.testing.assert_array_equal(predictions, predict_member(member, future))

    def test_invalid_ohlc_and_nontrading_volume_are_masked_without_losing_closes(self) -> None:
        source = self.prices(8).drop(columns=["price_invalid_ohlc", "price_low_volume"])
        source.loc[1, "High"] = source.Close.iloc[1] - 1
        source.loc[2, "Low"] = source.Close.iloc[2] + 1
        source.loc[3, "Open"] = -1
        source.loc[4, "Volume"] = 0
        source.loc[5, "Volume"] = 1
        source.loc[6, "Volume"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            source.to_csv(path, index=False)
            loaded, audit = load_prices(path)
        np.testing.assert_allclose(loaded.Close, source.Close)
        self.assertTrue(loaded.loc[[1, 2, 3], ["Open", "High", "Low"]].isna().all().all())
        self.assertTrue(loaded.loc[[1, 2, 3], "price_invalid_ohlc"].eq(1).all())
        self.assertTrue(loaded.loc[[4, 5, 6], "Volume"].isna().all())
        self.assertEqual(audit["invalid_ohlc_rows_masked"], 3)
        self.assertEqual(audit["zero_or_one_volume_rows_masked"], 3)

    def test_nonfinite_ohlc_and_volume_cannot_survive_source_validation(self) -> None:
        source = self.prices(5).drop(columns=["price_invalid_ohlc", "price_low_volume"])
        source.loc[1, "High"] = np.inf
        source.loc[2, "Open"] = np.nan
        source.loc[3, "Low"] = -np.inf
        source.loc[4, "Volume"] = np.inf
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            source.to_csv(path, index=False)
            loaded, _ = load_prices(path)
        self.assertTrue(loaded.loc[[1, 2, 3], ["Open", "High", "Low"]].isna().all().all())
        self.assertTrue(loaded.loc[[1, 2, 3], "price_invalid_ohlc"].eq(1).all())
        self.assertTrue(pd.isna(loaded.Volume.iloc[4]))
        self.assertEqual(loaded.price_low_volume.iloc[4], 1)


class ReturnEvaluationTests(unittest.TestCase):
    def test_block_comparator_is_paired_reproducible_and_has_correct_sign(self) -> None:
        actual = np.tile([-.02, .01, .03, -.04], 30)
        zero = np.zeros_like(actual)
        identical = block_comparison(actual, zero, zero, block=30, draws=80)
        self.assertEqual(identical["rmse_difference"], 0)
        self.assertEqual(identical["rmse_difference_95pct_block_interval"], [0., 0.])

        perfect = block_comparison(actual, actual, zero, block=30, draws=80)
        self.assertLess(perfect["rmse_difference"], 0)
        self.assertLess(perfect["rmse_difference_95pct_block_interval"][1], 0)
        self.assertEqual(perfect, block_comparison(actual, actual, zero, block=30, draws=80))
        reversed_comparison = block_comparison(actual, zero, actual, block=30, draws=80)
        self.assertGreater(reversed_comparison["rmse_difference_95pct_block_interval"][0], 0)
        self.assertAlmostEqual(perfect["rmse_difference"], -reversed_comparison["rmse_difference"])

    def test_metrics_treat_zero_forecasts_as_neutral_and_compare_return_error(self) -> None:
        score = metrics([-.1, 0., .1], [0., 0., 0.])
        self.assertAlmostEqual(score["direction_accuracy"], 1 / 3)
        self.assertEqual(score["r2_vs_zero"], 0.)
        self.assertIsNone(score["correlation"])
        perfect = metrics([-.1, 0., .1], [-.1, 0., .1])
        self.assertEqual(perfect["rmse"], 0.)
        self.assertEqual(perfect["direction_accuracy"], 1.)


if __name__ == "__main__":
    unittest.main()
