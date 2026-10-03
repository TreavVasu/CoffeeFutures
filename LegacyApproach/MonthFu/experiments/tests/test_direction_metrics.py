"""Checks for class balance, selective recall, and frozen temporal decisions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from direction_metrics import (binary_labels_from_returns, classification_metrics,
                               nonoverlap_positions, paired_block_bootstrap,
                               select_abstention_band, select_threshold,
                               selective_metrics)
from MonthFu.experiments.scripts import train_direction
from MonthFu.experiments.scripts.train_direction import confirm_frozen, run_holdout
from experiment_models import Spec


class DirectionScoresTests(unittest.TestCase):
    def test_confusion_and_both_class_scores_match_hand_calculation(self):
        y = [1, 1, 1, 0, 0, 0, 0]
        p = [.9, .7, .3, .8, .6, .4, .1]
        score = classification_metrics(y, p)
        self.assertEqual([score[k] for k in ["tp", "tn", "fp", "fn"]], [2, 2, 2, 1])
        self.assertAlmostEqual(score["accuracy"], 4 / 7)
        self.assertAlmostEqual(score["up_precision"], .5)
        self.assertAlmostEqual(score["up_recall"], 2 / 3)
        self.assertAlmostEqual(score["down_precision"], 2 / 3)
        self.assertAlmostEqual(score["down_recall"], .5)
        self.assertAlmostEqual(score["balanced_accuracy"], 7 / 12)
        self.assertAlmostEqual(score["macro_f1"], 4 / 7)
        self.assertAlmostEqual(score["pr_auc_up"], (1 + 2 / 3 + 3 / 6) / 3)
        self.assertAlmostEqual(score["pr_auc_down"], (1 + 2 / 3 + 3 / 4 + 4 / 6) / 4)
        self.assertAlmostEqual(score["roc_auc"], 8 / 12)
        self.assertAlmostEqual(score["brier"], np.square(np.array(y) - p).mean())

    def test_class_swap_preserves_macro_scores_and_swaps_precision_recall(self):
        y, p = np.array([1, 1, 0, 1, 0, 0]), np.array([.85, .25, .75, .70, .30, .20])
        score, swap = classification_metrics(y, p, .6), classification_metrics(1 - y, 1 - p, .4)
        for key in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "brier", "log_loss"]:
            self.assertAlmostEqual(score[key], swap[key])
        for metric in ["precision", "recall", "f1"]:
            self.assertAlmostEqual(score[f"up_{metric}"], swap[f"down_{metric}"])
            self.assertAlmostEqual(score[f"down_{metric}"], swap[f"up_{metric}"])
        self.assertAlmostEqual(score["pr_auc_up"], swap["pr_auc_down"])

    def test_neutral_realized_labels_are_excluded_before_all_comparisons(self):
        labels = binary_labels_from_returns([.1, -.2, 0., np.nan, -0., .001])
        np.testing.assert_equal(labels, [1., 0., np.nan, np.nan, np.nan, 1.])
        with self.assertRaises(ValueError):
            classification_metrics(labels, [.7, .2, .5, .5, .5, .8])
        keep = np.isfinite(labels)
        score = classification_metrics(labels[keep], np.array([.7, .2, .5, .5, .5, .8])[keep])
        self.assertEqual(score["rows"], 3)
        self.assertEqual(score["accuracy"], 1.)

    def test_invalid_data_are_rejected_without_independent_row_dropping(self):
        for y, p in [([], []), ([1], [.1, .2]), ([np.nan], [.3]),
                     ([.5], [.5]), ([1], [np.inf]), ([1], [1.01]), ([0], [-.01]),
                     ([[1]], [[.5]])]:
            with self.subTest(y=y, p=p), self.assertRaises(ValueError):
                classification_metrics(y, p)
        for threshold in [np.nan, np.inf, -.1, 1.1]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                classification_metrics([0, 1], [.2, .8], threshold)

    def test_threshold_equality_is_up_and_hard_probability_losses_are_finite(self):
        score = classification_metrics([1, 0], [.5, .5], .5)
        self.assertEqual(score["predicted_up"], 2)
        score = classification_metrics([0, 1], [1., 0.])
        self.assertEqual(score["brier"], 1.)
        self.assertTrue(np.isfinite(score["log_loss"]))
        self.assertGreater(score["log_loss"], 30)

    def test_single_class_slices_do_not_invent_roc_discrimination(self):
        score = classification_metrics([1, 1], [.7, .9])
        self.assertIsNone(score["roc_auc"])
        self.assertIsNone(score["pr_auc_down"])
        self.assertEqual(score["down_recall"], 0.)
        self.assertEqual(score["macro_f1"], .5)
        self.assertEqual(score["balanced_accuracy"], .5)


class DevelopmentSelectionTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2021-01-01", periods=4)
        self.cutoff = pd.Timestamp("2022-01-01")
        self.y, self.p = np.array([0, 0, 1, 1]), np.array([.45, .55, .60, .80])

    def test_threshold_is_frozen_from_pre_holdout_predictions(self):
        selected = select_threshold(self.y, self.p, dates=self.dates,
                                    holdout_start=self.cutoff,
                                    target_end_dates=self.dates + pd.Timedelta(days=30),
                                    thresholds=[.5, .6, .7])
        self.assertEqual(selected["threshold"], .6)
        self.assertEqual(selected["selection_source"], "pre_holdout_oof")
        self.assertEqual(selected["selection_metrics"]["macro_f1"], 1.)
        holdout_score = classification_metrics([0, 1], [.59, .61], selected["threshold"])
        self.assertEqual(holdout_score["accuracy"], 1.)
        self.assertEqual(selected["threshold"], .6)

    def test_selection_rejects_holdout_origin_and_unmatured_target(self):
        invalid = self.dates.to_list()
        invalid[-1] = self.cutoff
        with self.assertRaisesRegex(ValueError, "pre-holdout"):
            select_threshold(self.y, self.p, dates=invalid, holdout_start=self.cutoff)
        ends = (self.dates + pd.Timedelta(days=30)).to_list()
        ends[-1] = self.cutoff
        with self.assertRaisesRegex(ValueError, "mature"):
            select_threshold(self.y, self.p, dates=self.dates,
                             holdout_start=self.cutoff, target_end_dates=ends)
        with self.assertRaises(ValueError):
            select_abstention_band(self.y, self.p, dates=invalid, holdout_start=self.cutoff)

    def test_selection_requires_unique_chronological_origin_inventory(self):
        for dates in [self.dates[::-1], self.dates[:3], [self.dates[0]] * 4]:
            with self.subTest(dates=dates), self.assertRaises(ValueError):
                select_threshold(self.y, self.p, dates=dates, holdout_start=self.cutoff)

    def test_band_cannot_obtain_perfect_precision_by_discarding_most_labels(self):
        y = [1, 1, 1, 0, 0, 0]
        p = [.95, .52, .48, .05, .48, .52]
        selection = select_abstention_band(y, p, .5,
                                          dates=pd.bdate_range("2021-01-01", periods=6),
                                          holdout_start=self.cutoff, widths=[0., .05],
                                          min_coverage=.6)
        self.assertTrue(selection["gates_passed"])
        self.assertEqual(selection["width"], 0.)
        impressive = selection["search"][1]
        self.assertEqual(impressive["macro_f1"], 1.)
        self.assertFalse(impressive["gates_passed"])
        self.assertAlmostEqual(impressive["coverage"], 1 / 3)

    def test_failed_class_gates_are_reported_instead_of_claiming_valid_selection(self):
        selection = select_abstention_band([1, 0, 1, 0], [.9, .8, .9, .8],
                                          dates=self.dates, holdout_start=self.cutoff)
        self.assertFalse(selection["gates_passed"])
        self.assertEqual(selection["width"], 0.)


class SelectiveAndDependentTests(unittest.TestCase):
    def test_abstention_reports_conditional_and_unconditional_recall_separately(self):
        score = selective_metrics([1, 1, 0, 0, 1, 0], [.9, .55, .1, .45, .2, .8], .4, .6)
        self.assertEqual(score["rows"], 6)
        self.assertEqual(score["accepted_rows"], 4)
        self.assertEqual([score[k] for k in ["tp", "tn", "fp", "fn"]], [1, 1, 1, 1])
        self.assertAlmostEqual(score["coverage"], 2 / 3)
        for label in ["up", "down"]:
            self.assertAlmostEqual(score[f"{label}_recall"], .5)
            self.assertAlmostEqual(score[f"{label}_unconditional_recall"], 1 / 3)
            self.assertAlmostEqual(score[f"{label}_class_coverage"], 2 / 3)
        self.assertEqual(score["abstained_up"], 1)
        self.assertEqual(score["abstained_down"], 1)

    def test_zero_width_band_matches_full_coverage_classification(self):
        y, p = [1, 0, 1, 0], [.5, .49, .7, .8]
        full, selective = classification_metrics(y, p, .5), selective_metrics(y, p, .5, .5)
        for key in ["tp", "tn", "fp", "fn", "accuracy", "macro_f1", "up_recall", "down_recall"]:
            self.assertEqual(full[key], selective[key])
        self.assertEqual(selective["coverage"], 1.)
        self.assertEqual(selective["up_unconditional_recall"], full["up_recall"])

    def test_empty_accepted_policy_has_zero_coverage_and_no_accuracy(self):
        score = selective_metrics([0, 1], [.5, .5], .4, .6)
        self.assertEqual(score["accepted_rows"], 0)
        self.assertEqual(score["coverage"], 0.)
        self.assertIsNone(score["accuracy"])
        self.assertEqual(score["up_unconditional_recall"], 0.)
        self.assertEqual(score["down_unconditional_recall"], 0.)

    def test_nonoverlap_sampling_uses_real_endpoints_on_irregular_calendar(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01", "2024-01-04", "2024-01-11", "2024-01-15", "2024-01-23", "2024-01-29"]),
            "target_end_date": pd.to_datetime(["2024-01-11", "2024-01-15", "2024-01-23", "2024-01-29", "2024-02-02", "2024-02-06"]),
        })
        self.assertEqual(nonoverlap_positions(frame), [0, 2, 4])
        self.assertEqual(nonoverlap_positions(frame, offset=1), [1, 3, 5])
        for offset in [0, 1]:
            sample = frame.iloc[nonoverlap_positions(frame, offset)]
            self.assertTrue((sample.Date.iloc[1:].to_numpy() >= sample.target_end_date.iloc[:-1].to_numpy()).all())

    def test_identical_paired_models_have_exactly_zero_uncertainty_interval(self):
        y = (np.sin(np.arange(150) / 8) > 0).astype(int)
        p = (np.cos(np.arange(150) / 9) + 1) / 2
        result = paired_block_bootstrap(y, p, p, block=40, draws=80)
        self.assertEqual(result["block_sessions"], 40)
        for metric in result["metrics"].values():
            self.assertEqual(metric["difference"], 0.)
            np.testing.assert_equal(metric["difference_95pct_block_interval"], [0., 0.])

    def test_positive_bootstrap_difference_favors_selected_policy(self):
        y = np.tile([0, 1], 80)
        result = paired_block_bootstrap(y, y, 1 - y, block=40, draws=80)
        self.assertEqual(result["difference_sign"], "positive_favors_selected")
        for metric in result["metrics"].values():
            self.assertEqual(metric["difference"], 1.)
            np.testing.assert_equal(metric["difference_95pct_block_interval"], [1., 1.])

    def test_invalid_bootstrap_configuration_cannot_silently_change_pairs(self):
        for block, draws in [(0, 10), (10, 0), (2.5, 10)]:
            with self.subTest(block=block, draws=draws), self.assertRaises(ValueError):
                paired_block_bootstrap([0, 1], [.2, .8], [.3, .7], block=block, draws=draws)
        with self.assertRaises(ValueError):
            paired_block_bootstrap([0, 1], [.2, .8], [np.nan, .7], draws=10)


class FrozenSelectionResumeTests(unittest.TestCase):
    """A resumed run must reuse completed work and never re-pick the model."""

    def frozen(self, folder, **overrides):
        decision = {"recipes": {"selected_direction": {"weights": {"c": 1.}, "threshold": .4}},
                    "threshold_decisions": {"c": {"threshold": .4}},
                    "selected_abstention": {"lower": .45, "upper": .55}}
        decision.update(overrides)
        (folder / "selection.json").write_text(json.dumps(decision))
        return decision

    def test_missing_frozen_selection_is_written_by_a_first_run(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            confirm_frozen(folder, {"a": 1}, {}, {})
            self.assertFalse((folder / "selection.json").exists())

    def test_resume_must_reproduce_the_frozen_pre_holdout_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frozen = self.frozen(folder)
            # An identical re-derivation is the only acceptable resume.
            confirm_frozen(folder, frozen["recipes"], frozen["threshold_decisions"],
                           frozen["selected_abstention"])
            # Re-selecting after inspecting the holdout is refused loudly.
            for name, recipes, decisions, band in [
                    ("recipes", {"selected_direction": {"weights": {"c": 2.}, "threshold": .6}},
                     frozen["threshold_decisions"], frozen["selected_abstention"]),
                    ("threshold", frozen["recipes"], {"c": {"threshold": .7}}, frozen["selected_abstention"]),
                    ("abstention", frozen["recipes"], frozen["threshold_decisions"],
                     {"lower": .3, "upper": .7})]:
                with self.subTest(changed=name), self.assertRaisesRegex(AssertionError, "Refusing to re-select"):
                    confirm_frozen(folder, recipes, decisions, band)

    def test_resume_tolerates_float32_noise_but_not_a_changed_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            self.frozen(folder, threshold_decisions={"c": {"threshold": .4, "log_loss": 1.3542274895765183}})
            # Cached float32 candidate columns can shift a reported score's last
            # digit without changing which candidate or threshold was chosen.
            confirm_frozen(folder, {"selected_direction": {"weights": {"c": 1.}, "threshold": .4}},
                           {"c": {"threshold": .4, "log_loss": 1.3542274895765172}},
                           {"lower": .45, "upper": .55})
            # A genuinely different score, or a different winner, is refused.
            with self.assertRaisesRegex(AssertionError, "Refusing to re-select"):
                confirm_frozen(folder, {"selected_direction": {"weights": {"c": 1.}, "threshold": .4}},
                               {"c": {"threshold": .4, "log_loss": 1.4542274895765183}},
                               {"lower": .45, "upper": .55})
            with self.assertRaisesRegex(AssertionError, "Refusing to re-select"):
                confirm_frozen(folder, {"selected_direction": {"weights": {"c": 1.}, "threshold": .41}},
                               {"c": {"threshold": .4}}, {"lower": .45, "upper": .55})
            # Structure changes are not smoothed over by the tolerance.
            for changed in [{"c": {}}, {"c": {"threshold": .4}, "d": {"threshold": .4}}]:
                with self.subTest(changed=changed), self.assertRaisesRegex(AssertionError, "Refusing to re-select"):
                    confirm_frozen(folder, {"selected_direction": {"weights": {"c": 1.}, "threshold": .4}},
                                   changed, {"lower": .45, "upper": .55})

    def test_holdout_resume_reuses_audited_quarters_and_refits_only_the_rest(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            dates = pd.to_datetime(["2022-01-03", "2022-01-04", "2022-04-01", "2022-04-04"])
            hold = pd.DataFrame({"Date": dates, "Close": 1., "target_end_date": dates + pd.Timedelta(days=30),
                                 "target_return": .01, "target_sessions": 21})
            quarter = dates.to_period("Q").astype(str)
            recipes = {"selected_direction": {"task": "class", "group": "basic",
                                              "weights": {"m": 1.}, "threshold": .4}}
            fitted = [0]
            def fake_fit(spec, rows, columns, quick=False):
                fitted[0] += 1
                return {"spec": {"task": "class", "require_all_cot": False}, "features": ["x"],
                        "required_cot_features": [], "cot_inclusion_audit": [],
                        "train_rows": len(rows), "train_last_date": "2021-12-31",
                        "train_last_target_date": "2021-12-30", "warnings": [],
                        "estimator": None}
            def fake_predict(recipe, members, frame):
                return np.full(len(frame), .7)
            def fake_add(part, old, train):
                part["baseline_monthly_return"] = 0.
            def fake_mature(frame, cutoff):
                return pd.DataFrame({"Date": [pd.Timestamp("2021-12-31")],
                                     "target_end_date": [pd.Timestamp("2021-12-30")]})
            specs = {"m": Spec("m", "basic", "class", "xgb", 0)}
            with patch.object(train_direction, "fit_member", fake_fit), \
                 patch.object(train_direction, "predict_recipe", fake_predict), \
                 patch.object(train_direction, "add_baselines", fake_add), \
                 patch.object(train_direction, "mature_training_rows", fake_mature):
                first = run_holdout(folder, "30calendar", pd.DataFrame(), {"basic": ["x"]}, specs, recipes,
                                    ["m"], hold, pd.DataFrame(), False, False, [])
                self.assertEqual(fitted[0], 2)
                self.assertEqual(first.quarter.nunique(), 2)
                # A resumed run must not refit the two quarters already audited.
                resumed = run_holdout(folder, "30calendar", pd.DataFrame(), {"basic": ["x"]}, specs, recipes,
                                      ["m"], hold, pd.DataFrame(), False, True, [])
                self.assertEqual(fitted[0], 2)
                self.assertEqual(len(resumed), len(first))
                self.assertEqual(resumed.quarter.tolist(), first.quarter.tolist())
                # Dropping a quarter's audit re-admits that single refit.
                audit = json.loads((folder / "fit_audit_holdout.json").read_text())
                kept = [row for row in audit if row["quarter"] != "2022Q1"]
                (folder / "fit_audit_holdout.json").write_text(json.dumps(kept))
                (folder / "holdout_predictions_partial.csv").unlink()
                run_holdout(folder, "30calendar", pd.DataFrame(), {"basic": ["x"]}, specs, recipes,
                            ["m"], hold, pd.DataFrame(), False, True, [])
                self.assertEqual(fitted[0], 4)


if __name__ == "__main__":
    unittest.main()
