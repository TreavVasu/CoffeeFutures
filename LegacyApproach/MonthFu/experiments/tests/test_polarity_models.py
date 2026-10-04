"""Checks for the two separately-trained direction heads and their scoring."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "MonthFu/experiments/src"))

from polarity_metrics import (  # noqa: E402
    block_bootstrap_difference, decide, evaluate_calls, margin_baseline,
    polarity_report, select_side_threshold, select_threshold_pair,
)
from polarity_models import (  # noqa: E402
    POLARITIES, DirectionSpec, candidates, fit_direction, predict_direction,
    polarity_banks, training_rows, usable_features,
)


class SpecTests(unittest.TestCase):
    def test_both_polarities_are_searched_independently(self):
        specs = candidates(["basic", "candles"])
        self.assertEqual(len(specs), 2 * 2 * 3 * 2)
        self.assertEqual({s.polarity for s in specs}, set(POLARITIES))
        up = {s.name for s in specs if s.polarity == "up"}
        down = {s.name for s in specs if s.polarity == "down"}
        self.assertEqual(up & down, set())
        # The same family/variant on both sides is a coincidence, not a coupling.
        self.assertEqual({s.family for s in specs if s.polarity == "up"},
                         {s.family for s in specs if s.polarity == "down"})

    def test_each_spec_targets_its_own_label_column(self):
        self.assertEqual(DirectionSpec("up", "basic", "linear", 0).target_column, "direction_up")
        self.assertEqual(DirectionSpec("down", "basic", "linear", 0).target_column, "direction_down")

    def test_invalid_specs_are_rejected(self):
        for args in [("sideways", "basic", "linear", 0), ("up", "basic", "mlp", 0),
                     ("up", "basic", "linear", 7)]:
            with self.assertRaises(ValueError):
                DirectionSpec(*args)


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "target_return": [.05, -.03, 0.0, np.nan, .01],
            "direction_up": [1.0, 0.0, np.nan, np.nan, 1.0],
            "direction_down": [0.0, 1.0, np.nan, np.nan, 0.0],
        })

    def test_both_sides_see_identical_rows(self):
        up, down = training_rows(self.frame, "up"), training_rows(self.frame, "down")
        self.assertEqual(list(up.index), list(down.index))
        self.assertEqual(list(up.index), [0, 1, 4])

    def test_zero_and_undefined_returns_are_excluded_for_both_sides(self):
        for polarity in POLARITIES:
            rows = training_rows(self.frame, polarity)
            self.assertNotIn(2, rows.index)   # exactly zero
            self.assertNotIn(3, rows.index)   # undefined

    def test_unknown_polarity_or_missing_column_is_rejected(self):
        with self.assertRaises(ValueError):
            training_rows(self.frame, "sideways")
        with self.assertRaisesRegex(ValueError, "missing"):
            training_rows(self.frame.drop(columns=["direction_up"]), "up")


class FeatureScreenTests(unittest.TestCase):
    def test_screening_uses_training_rows_only(self):
        frame = pd.DataFrame({"kept": [1.0, 2.0, 3.0, 4.0],
                              "constant": [5.0] * 4,
                              "sparse": [1.0, np.nan, np.nan, np.nan],
                              "absent": [np.nan] * 4})
        self.assertEqual(usable_features(frame, ["kept", "constant", "sparse", "absent"]), ["kept"])

    def test_candles_group_is_side_specific_but_others_are_shared(self):
        candles = ["candle_hammer", "candle_hanging_man", "candle_hammer_count@5"]
        shared = ["price_return_5"]
        up, down = polarity_banks(candles, shared, "up"), polarity_banks(candles, shared, "down")
        self.assertIn("candle_hammer", up["candles"])
        self.assertNotIn("candle_hanging_man", up["candles"])
        self.assertIn("candle_hanging_man", down["candles"])
        self.assertNotIn("candle_hammer", down["candles"])
        # Shared groups must be identical so a comparison isolates the effect.
        self.assertEqual(up["basic"], down["basic"])
        self.assertEqual(up["engineered"], down["engineered"])


class DecisionRuleTests(unittest.TestCase):
    def test_each_side_is_compared_against_its_own_threshold(self):
        up = np.array([.9, .1, .9, .1, .6])
        down = np.array([.1, .9, .6, .6, .6])
        # Rows 0,1 clear one side each; row 2 and row 4 have both sides firing so
        # the rule declines; row 3 clears DOWN only.
        np.testing.assert_array_equal(decide(up, down, .5, .5), [1.0, -1.0, 0.0, -1.0, 0.0])

    def test_conflicting_strengths_become_neutral_rather_than_arbitrary(self):
        self.assertEqual(decide([.8], [.8], .5, .5)[0], 0.0)   # both fire at once
        self.assertEqual(decide([.2], [.2], .5, .5)[0], 0.0)   # neither fires

    def test_thresholds_are_independent(self):
        up, down = np.array([.55]), np.array([.20])
        self.assertEqual(decide(up, down, .50, .30)[0], 1.0)
        self.assertEqual(decide(up, down, .60, .30)[0], 0.0)

    def test_invalid_inputs_are_rejected(self):
        for args in [([.5, np.nan], [.5], .5, .5), ([.5], [.5, .5], .5, .5)]:
            with self.assertRaises(ValueError):
                decide(*args)
        with self.assertRaises(ValueError):
            decide([.5], [.5], 1.4, .5)


class ScoringTests(unittest.TestCase):
    def test_counts_match_a_hand_calculation(self):
        y = np.array([1., 1., 1., 0., 0., 0., 0.])
        call = np.array([1., 1., -1., 1., -1., -1., 0.])
        # tp: rows 0,1 | fp: row 3 | fn: row 2 | tn: rows 4,5,6 (row 6 abstains)
        score = evaluate_calls(y, call)
        self.assertEqual([score[k] for k in ["tp", "tn", "fp", "fn"]], [2, 3, 1, 1])
        self.assertAlmostEqual(score["accuracy"], 5 / 7)
        self.assertAlmostEqual(score["coverage"], 6 / 7)
        self.assertEqual(score["neutral_rows"], 1)
        # Unconditional recall uses full class support, not accepted rows only.
        self.assertAlmostEqual(score["up_unconditional_recall"], 2 / 3)
        self.assertAlmostEqual(score["down_unconditional_recall"], 2 / 4)

    def test_abstention_cannot_inflate_the_neutral_accuracy(self):
        y = np.array([1., 0., 1., 0.])
        everything = evaluate_calls(y, np.array([1., -1., 1., -1.]))
        half = evaluate_calls(y, np.array([1., -1., 0., 0.]))
        self.assertAlmostEqual(everything["accuracy"], 1.0)
        self.assertLess(half["accuracy"], 1.0)
        self.assertAlmostEqual(half["neutral_accuracy"], 0.5)

    def test_margin_baseline_matches_a_direct_comparison(self):
        y = np.array([1., 0., 1.])
        up, down = np.array([.6, .2, .6]), np.array([.4, .4, .9])
        # Direct comparison calls [+1, -1, -1]; only the first two are correct.
        self.assertEqual(margin_baseline(y, up, down)["accuracy"], 2 / 3)

    def test_labels_are_binary_and_calls_are_validated(self):
        with self.assertRaises(ValueError):
            evaluate_calls([1., 0., np.nan], [1., -1., 1.])
        with self.assertRaises(ValueError):
            evaluate_calls([1., 0.], [1., 2.])

    def test_report_excludes_zero_returns_once_for_both_sides(self):
        frame = pd.DataFrame({"target_return": [.02, -.01, 0.0, .03, -.02]})
        report = polarity_report(frame, [.8, .2, .5, .7, .3], [.2, .8, .5, .3, .7], .5, .5)
        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["excluded_neutral_rows"], 1)
        self.assertEqual(report["accuracy"], 1.0)
        self.assertIn("margin_baseline", report)


class ThresholdSelectionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(99)
        rows = 300
        self.dates = pd.bdate_range("2010-01-04", periods=rows)
        self.ends = pd.date_range("2011-06-01", periods=rows, freq="B")
        self.labels = (rng.random(rows) < .55).astype(float)
        self.scores = np.clip(.3 + .5 * self.labels + rng.normal(0, .12, rows), 0, 1)

    def test_selection_records_its_window_and_objective(self):
        picked = select_side_threshold(self.labels, self.scores, dates=self.dates,
                                       target_end_dates=self.ends, holdout_start="2022-01-01",
                                       polarity="up")
        self.assertEqual(picked["objective"], "up_f1")
        self.assertEqual(picked["holdout_start"], "2022-01-01")
        self.assertEqual(picked["selection_source"], "pre_holdout_oof")
        self.assertGreater(len(picked["search"]), 1)
        self.assertGreaterEqual(picked["threshold"], .3)
        self.assertLessEqual(picked["threshold"], .7)

    def test_each_side_optimises_its_own_class(self):
        up = select_side_threshold(self.labels, self.scores, dates=self.dates,
                                   target_end_dates=self.ends, holdout_start="2022-01-01",
                                   polarity="up")
        down = select_side_threshold(1 - self.labels, 1 - self.scores, dates=self.dates,
                                     target_end_dates=self.ends, holdout_start="2022-01-01",
                                     polarity="down")
        self.assertEqual(up["objective"], "up_f1")
        self.assertEqual(down["objective"], "down_f1")

    def test_holdout_origins_and_unmatured_labels_are_refused(self):
        common = dict(dates=self.dates, target_end_dates=self.ends,
                      holdout_start="2022-01-01", polarity="up")
        with self.assertRaisesRegex(ValueError, "pre-holdout"):
            select_side_threshold(self.labels, self.scores,
                                  dates=self.dates, target_end_dates=self.ends,
                                  holdout_start="2009-01-01", polarity="up")
        with self.assertRaisesRegex(ValueError, "matured"):
            select_side_threshold(self.labels, self.scores,
                                  dates=self.dates,
                                  target_end_dates=pd.DatetimeIndex(["2030-01-01"] * len(self.ends)),
                                  holdout_start="2022-01-01", polarity="up")
        self.assertTrue(common)


class FitTests(unittest.TestCase):
    def _labelled(self, rows: int = 240):
        rng = np.random.default_rng(11)
        close = 100 * np.exp(np.cumsum(rng.normal(0, .02, rows)))
        ret = np.r_[0.0, np.diff(np.log(close))]
        frame = pd.DataFrame({"Date": pd.bdate_range("2016-01-04", periods=rows),
                              "Close": close, "Open": close * (1 + rng.normal(0, .004, rows)),
                              "feature_a": rng.normal(size=rows),
                              "feature_b": np.arange(rows, dtype=float)})
        frame["target_return"] = ret
        observed = np.isfinite(ret) & (ret != 0)
        frame["direction_up"] = np.where(observed, (ret > 0).astype(float), np.nan)
        frame["direction_down"] = np.where(observed, (ret < 0).astype(float), np.nan)
        return frame

    def test_each_side_fits_on_its_own_label_and_predicts_probabilities(self):
        frame = self._labelled()
        for polarity in POLARITIES:
            member = fit_direction(DirectionSpec(polarity, "basic", "linear", 0), frame,
                                   ["feature_a", "feature_b"], quick=True)
            self.assertEqual(member["spec"].polarity, polarity)
            self.assertEqual(sorted(member["features"]), ["feature_a", "feature_b"])
            scores = predict_direction(member, frame)
            self.assertEqual(len(scores), len(frame))
            self.assertTrue(np.isfinite(scores).all())
            self.assertTrue(((scores >= 0) & (scores <= 1)).all())
            # Balanced class weights keep neither side degenerate.
            self.assertGreater(member["positive_rate"], 0.)
            self.assertLess(member["positive_rate"], 1.)

    def test_no_usable_features_is_an_error_not_a_silent_fit(self):
        frame = self._labelled().assign(feature_a=1.0, feature_b=np.nan)
        with self.assertRaisesRegex(ValueError, "No usable features"):
            fit_direction(DirectionSpec("up", "basic", "linear", 0), frame,
                          ["feature_a", "feature_b"], quick=True)


class ThresholdPairTests(unittest.TestCase):
    """The two cuts feed one rule, so they must be chosen together."""

    def setUp(self):
        rng = np.random.default_rng(7)
        rows = 400
        self.dates = pd.bdate_range("2010-01-04", periods=rows)
        self.ends = pd.date_range("2012-01-02", periods=rows, freq="B")
        self.labels = (rng.random(rows) < .5).astype(float)
        self.up = np.clip(.5 + .3 * (self.labels * 2 - 1) + rng.normal(0, .15, rows), 0, 1)
        self.down = np.clip(.5 - .3 * (self.labels * 2 - 1) + rng.normal(0, .15, rows), 0, 1)

    def _select(self, **kwargs):
        return select_threshold_pair(
            self.labels, self.up, self.down, dates=self.dates,
            target_end_dates=self.ends, holdout_start="2022-01-01", **kwargs)

    def test_pair_is_selected_on_the_joint_objective(self):
        picked = self._select()
        self.assertEqual(picked["objective"], "joint_macro_f1_of_the_combined_call")
        self.assertEqual(picked["selection_source"], "pre_holdout_development")
        self.assertIn("up_threshold", picked)
        self.assertIn("down_threshold", picked)
        self.assertGreater(picked["search_size"], 1)

    def test_informative_scores_beat_a_one_sided_rule(self):
        picked = self._select()
        score = evaluate_calls(self.labels, decide(self.up, self.down,
                                                   picked["up_threshold"],
                                                   picked["down_threshold"]))
        one_sided = evaluate_calls(self.labels, np.ones(len(self.labels)))
        self.assertGreater(score["macro_f1"], one_sided["macro_f1"])
        # Both sides must actually be called, not just UP.
        self.assertGreater(score["down_recall"], 0.2)
        self.assertGreater(score["up_recall"], 0.2)

    def test_joint_search_avoids_the_degenerate_asymmetric_pair(self):
        # Two independent side-F1 cuts would both drift toward calling their own
        # class; the joint search must not return that collapse.
        picked = self._select()
        self.assertLess(abs(picked["up_threshold"] - picked["down_threshold"]), 0.3)

    def test_holdout_and_unmatured_rows_are_refused(self):
        for dates, ends in [
            (self.dates, pd.DatetimeIndex(["2030-01-01"] * len(self.dates))),
            (pd.DatetimeIndex(["2030-01-01"] * len(self.labels)), self.ends),
        ]:
            with self.assertRaises(ValueError):
                select_threshold_pair(self.labels, self.up, self.down, dates=dates,
                                      target_end_dates=ends, holdout_start="2022-01-01")

    def test_nan_scores_are_refused(self):
        broken = self.up.copy()
        broken[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            select_threshold_pair(self.labels, broken, self.down, dates=self.dates,
                                  target_end_dates=self.ends, holdout_start="2022-01-01")

    def test_out_of_range_grid_is_refused(self):
        with self.assertRaises(ValueError):
            self._select(up_grid=np.array([0.5, 1.4]))


class BootstrapTests(unittest.TestCase):
    def test_identical_calls_give_a_zero_difference(self):
        y = np.array([1., 0., 1., 0., 1., 0.])
        call = np.array([1., -1., 1., -1., 1., -1.])
        result = block_bootstrap_difference(y, call, call, block=2, draws=50)
        for entry in result["metrics"].values():
            self.assertAlmostEqual(entry["difference"], 0.0)

    def test_a_perfect_rule_beats_a_reversed_one(self):
        y = np.array([1., 0., 1., 0., 1., 0., 1., 0.])
        good = np.where(y == 1, 1.0, -1.0)
        bad = -good
        result = block_bootstrap_difference(y, good, bad, block=2, draws=100)
        entry = result["metrics"]["accuracy"]
        self.assertGreater(entry["difference"], 0.9)
        low, high = entry["difference_95pct_block_interval"]
        self.assertLessEqual(low, high)

    def test_invalid_arguments_are_refused(self):
        y, call = np.array([1., 0.]), np.array([1., -1.])
        for kwargs in [{"block": 0}, {"block": 2, "draws": 0}]:
            with self.assertRaises(ValueError):
                block_bootstrap_difference(y, call, call, **kwargs)
        with self.assertRaises(ValueError):
            block_bootstrap_difference(y, call, np.array([1.]), block=2)