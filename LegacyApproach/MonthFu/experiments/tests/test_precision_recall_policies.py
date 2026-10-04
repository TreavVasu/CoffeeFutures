"""Development-only objective, gates, alias and export-order checks."""
from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from MonthFu.experiments.scripts.precision_recall_policies import (
    THRESHOLDS, evaluate_policies, policy_block_uncertainty, run_horizon, select_precision_recall_policies)


class PrecisionRecallPolicyTests(unittest.TestCase):
    def setUp(self):
        returns = np.array([-0.1, 0.1, -0.2, 0.2, -0.1, 0.1, -0.2, 0.2, 0.0])
        first = np.array([0.15, 0.70, 0.45, 0.60, 0.65, 0.80, 0.30, 0.55, 0.9])
        second = np.array([0.30, 0.70, 0.50, 0.55, 0.58, 0.70, 0.35, 0.65, 0.1])
        dates = pd.bdate_range("2020-01-02", periods=len(returns))
        self.oof = pd.DataFrame({"Date": dates, "target_end_date": dates+pd.Timedelta(days=30),
                                 "target_return": returns, "class__basic__linear": first,
                                 "class__basic__xgb": second,
                                 "class__basic__equal_ensemble": (first+second)/2})
        recipes = {
            "class__basic__linear": {"task": "class", "group": "basic", "threshold": .5,
                                      "weights": {"linear_variant": 1.}},
            "class__basic__xgb": {"task": "class", "group": "basic", "threshold": .525,
                                   "weights": {"xgb_variant": 1.}},
            "class__basic__equal_ensemble": {"task": "class", "group": "basic", "threshold": .5,
                                              "weights": {"linear_variant": .5, "xgb_variant": .5}},
            "reg__basic__linear": {"task": "reg", "group": "basic", "threshold": .5,
                                    "weights": {"reg_variant": 1.}}}
        recipes["selected_direction"] = dict(recipes["class__basic__linear"])
        self.selection = {"holdout_start": "2022-01-01", "recipes": recipes}

    def test_exact_grid_zero_exclusion_and_alias_deduplication(self):
        selected, grid = select_precision_recall_policies(self.oof, self.selection)
        self.assertEqual(len(THRESHOLDS), 13)
        self.assertEqual(selected["selection_rows"], 8)
        self.assertEqual(selected["excluded_neutral_rows"], 1)
        self.assertEqual(selected["candidate_recipe_count"], 3)
        self.assertEqual(len(grid), 39)
        self.assertEqual(selected["deduplicated_aliases"]["selected_direction"], "class__basic__linear")
        self.assertNotIn("selected_direction", grid.recipe.values)
        self.assertNotIn("reg__basic__linear", grid.recipe.values)

    def test_two_objectives_match_independent_grid_maxima(self):
        selected, grid = select_precision_recall_policies(self.oof, self.selection)
        precision = selected["policies"]["precision_first"]
        passing = grid.loc[(grid.up_recall >= .4) & (grid.down_recall >= .4)]
        self.assertGreaterEqual(precision["selection_metrics"]["minimum_recall"], .4)
        self.assertEqual(precision["selection_metrics"]["minimum_precision"], passing.minimum_precision.max())
        top = passing.loc[passing.minimum_precision == passing.minimum_precision.max()]
        self.assertEqual(precision["selection_metrics"]["macro_f1"], top.macro_f1.max())
        recall = selected["policies"]["recall_balanced"]
        self.assertEqual(recall["selection_metrics"]["minimum_recall"], grid.minimum_recall.max())
        top = grid.loc[grid.minimum_recall == grid.minimum_recall.max()]
        self.assertEqual(recall["selection_metrics"]["macro_f1"], top.macro_f1.max())

    def test_gate_failure_does_not_loosen_gate_or_claim_success(self):
        failed = self.oof.copy()
        for name in [column for column in failed if column.startswith("class__")]:
            failed[name] = 0.9
        selected, _ = select_precision_recall_policies(failed, self.selection)
        policy = selected["policies"]["precision_first"]
        self.assertFalse(policy["gates_passed"])
        self.assertEqual(policy["recipe"], "class__basic__linear")
        self.assertEqual(policy["threshold"], .5)
        self.assertIn("No model/threshold", policy["fallback_reason"])
        self.assertEqual(policy["selection_metrics"]["down_recall"], 0.)

    def test_future_origins_immature_labels_and_invalid_probabilities_rejected(self):
        future = self.oof.copy()
        future.loc[len(future)-1, "Date"] = pd.Timestamp("2022-01-03")
        with self.assertRaisesRegex(ValueError, "strictly before holdout"):
            select_precision_recall_policies(future, self.selection)
        immature = self.oof.copy()
        immature.loc[0, "target_end_date"] = pd.Timestamp("2022-01-01")
        with self.assertRaisesRegex(ValueError, "strictly before holdout"):
            select_precision_recall_policies(immature, self.selection)
        invalid = self.oof.copy()
        invalid.loc[0, "class__basic__linear"] = 1.01
        with self.assertRaisesRegex(ValueError, "Invalid OOF"):
            select_precision_recall_policies(invalid, self.selection)

    def test_policy_selection_is_saved_before_any_holdout_read(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"selection.json").write_text(json.dumps(self.selection))
            self.oof.to_csv(path/"cv_predictions.csv", index=False)
            holdout = self.oof.copy()
            holdout.Date = holdout.Date+pd.DateOffset(years=3)
            holdout.target_end_date = holdout.target_end_date+pd.DateOffset(years=3)
            holdout.to_csv(path/"holdout_predictions.csv", index=False)
            original_read = pd.read_csv
            observed = []
            def inspect_read(file, *args, **kwargs):
                if Path(file).name == "holdout_predictions.csv":
                    self.assertTrue((path/"precision_recall_policy_selection.json").exists())
                    frozen = json.loads((path/"precision_recall_policy_selection.json").read_text())
                    self.assertEqual(frozen["selection_source"], "mature_pre_holdout_oof_only")
                    observed.append(True)
                return original_read(file, *args, **kwargs)
            with patch("MonthFu.experiments.scripts.precision_recall_policies.pd.read_csv", side_effect=inspect_read):
                run_horizon(path)
            self.assertEqual(observed, [True])
            result = pd.read_csv(path/"precision_recall_policy_metrics.csv")
            self.assertEqual(len(result), 5)
            self.assertTrue((result.coverage == 1).all())

    def test_probability_losses_do_not_change_with_threshold(self):
        selected, _ = select_precision_recall_policies(self.oof, self.selection)
        holdout = self.oof.copy()
        holdout.Date += pd.DateOffset(years=3)
        holdout.target_end_date += pd.DateOffset(years=3)
        result = evaluate_policies(holdout, selected).set_index("policy")
        for policy in ["precision_first", "recall_balanced"]:
            for metric in ["brier", "log_loss", "pr_auc_up", "pr_auc_down", "roc_auc"]:
                self.assertEqual(result.loc[policy, metric], result.loc[f"{policy}__original_macro_f1_threshold", metric])

    def test_blocks_compare_frozen_policy_to_original_without_reselection(self):
        selected, _ = select_precision_recall_policies(self.oof, self.selection)
        selected["policies"]["precision_first"]["recipe"] = selected["original_direction_policy"]["recipe"]
        selected["policies"]["precision_first"]["threshold"] = selected["original_direction_policy"]["threshold"]
        result = policy_block_uncertainty(self.oof, selected, draws=20)
        self.assertEqual(result["primary_block_sessions"], 60)
        for estimate in result["policies"]["precision_first"].values():
            for metric in estimate["metrics"].values():
                self.assertEqual(metric["difference"], 0.)
                self.assertEqual(metric["difference_95pct_block_interval"], [0., 0.])


if __name__ == "__main__":
    unittest.main()
