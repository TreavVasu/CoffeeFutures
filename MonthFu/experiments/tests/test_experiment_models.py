from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MonthFu/experiments/src"))

from experiment_models import (  # noqa: E402
    Spec, candidates, estimator, fit_member, predict_member, predict_recipe,
    regression_direction_score,
)
from MonthFu.experiments.scripts.train_direction import (  # noqa: E402
    direction_row, predict_oof, select_recipes,
)
from MonthFu.src.modeling import mature_training_rows  # noqa: E402


def synthetic_train(rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(732)
    x = rng.normal(size=rows)
    dates = pd.bdate_range("2018-01-02", periods=rows)
    frame = pd.DataFrame({"Date": dates, "target_end_date": dates + pd.Timedelta(days=30),
                          "target_return": .04 * np.where(x >= 0, 1., -1.) + .004 * x,
                          "signal": x, "noise": rng.normal(size=rows),
                          "constant": 17., "sparse": np.nan})
    frame.loc[:19, "sparse"] = rng.normal(size=20)
    frame.loc[::13, "noise"] = np.nan
    return frame


def synthetic_oof():
    dates = pd.bdate_range("2020-01-02", periods=96)
    returns = .04 * np.where(np.arange(len(dates)) % 3 == 0, -1., 1.)
    specs = candidates({"basic": ["signal", "noise"]})
    frame = pd.DataFrame({"Date": dates, "target_end_date": dates + pd.Timedelta(days=30),
                          "target_return": returns})
    for spec in specs:
        if spec.task == "class":
            good = np.where(returns > 0, .8, .2)
            frame[spec.name] = good if spec.variant == 0 else 1 - good
        else:
            frame[spec.name] = returns if spec.variant == 0 else -returns
    return frame, specs


class ExperimentModelTests(unittest.TestCase):
    def test_candidate_cross_product_and_parameter_variants(self):
        specs = candidates({"basic": [], "direction_core": []})
        self.assertEqual(len(specs), 24)
        self.assertEqual(len({spec.name for spec in specs}), len(specs))
        for task in ("class", "reg"):
            for family in ("linear", "xgb", "forest"):
                first = estimator(Spec("a", "basic", task, family, 0), quick=True)
                second = estimator(Spec("b", "basic", task, family, 1), quick=True)
                self.assertNotEqual(first.named_steps["model"].get_params(), second.named_steps["model"].get_params())
                self.assertEqual(first.named_steps["impute"].strategy, "median")
                self.assertTrue(first.named_steps["impute"].add_indicator)
        self.assertEqual(estimator(Spec("a", "basic", "class", "linear", 0)).named_steps["model"].class_weight, "balanced")
        self.assertEqual(estimator(Spec("a", "basic", "class", "forest", 0)).named_steps["model"].class_weight, "balanced_subsample")

    def test_all_six_family_tasks_fit_and_return_finite_predictions(self):
        train = synthetic_train()
        latest = train.iloc[-9:].copy()
        for task in ("class", "reg"):
            for family in ("linear", "xgb", "forest"):
                with self.subTest(task=task, family=family):
                    model = fit_member(Spec(f"{task}_{family}", "basic", task, family, 0),
                                       train, ["signal", "noise", "constant", "sparse"], quick=True)
                    self.assertEqual(model["features"], ["signal", "noise"])
                    values = predict_member(model, latest)
                    self.assertEqual(values.shape, (len(latest),))
                    self.assertTrue(np.isfinite(values).all())
                    if task == "class":
                        self.assertTrue(((values >= 0) & (values <= 1)).all())

    def test_transform_statistics_use_training_observations_only(self):
        train = synthetic_train()
        member = fit_member(Spec("linear", "basic", "reg", "linear", 0),
                            train, ["signal", "noise", "constant", "sparse"], quick=True)
        expected_medians = train[member["features"]].median().to_numpy()
        imputer = member["estimator"].named_steps["impute"]
        np.testing.assert_allclose(imputer.statistics_, expected_medians)
        before = imputer.statistics_.copy()
        future = train.iloc[-4:].copy()
        future.loc[:, ["signal", "noise"]] = 1e12
        predict_member(member, future)
        np.testing.assert_array_equal(imputer.statistics_, before)
        self.assertEqual(member["train_last_target_date"], str(train.target_end_date.max().date()))

    def test_zero_labels_excluded_for_class_and_retained_for_regression(self):
        train = synthetic_train()
        train.loc[:9, "target_return"] = 0
        classifier = fit_member(Spec("class", "basic", "class", "linear", 0), train, ["signal", "noise"], quick=True)
        regression = fit_member(Spec("reg", "basic", "reg", "linear", 0), train, ["signal", "noise"], quick=True)
        self.assertEqual(classifier["train_rows"], len(train) - 10)
        self.assertEqual(regression["train_rows"], len(train))

    def test_missing_or_nonfinite_training_labels_are_rejected_before_binary_cast(self):
        for task in ("class", "reg"):
            for invalid in (np.nan, np.inf, -np.inf):
                train = synthetic_train()
                train.loc[0, "target_return"] = invalid
                with self.subTest(task=task, invalid=invalid), self.assertRaisesRegex(ValueError, "finite"):
                    fit_member(Spec("invalid", "basic", task, "linear", 0), train, ["signal"], quick=True)
            with self.subTest(task=task, invalid="empty"), self.assertRaisesRegex(ValueError, "nonempty"):
                fit_member(Spec("invalid", "basic", task, "linear", 0), synthetic_train().iloc[:0], ["signal"], quick=True)

    def test_all_classifier_families_require_both_nonzero_return_classes(self):
        train = synthetic_train()
        train["target_return"] = .04
        for family in ("linear", "xgb", "forest"):
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, "Both classes"):
                fit_member(Spec("oneclass", "basic", "class", family, 0), train, ["signal"], quick=True)

    def test_xgboost_sample_weights_are_balanced_from_its_training_labels(self):
        train = synthetic_train()
        train.loc[:130, "target_return"] = .04
        spec = Spec("xgb", "basic", "class", "xgb", 0)
        pipeline = estimator(spec, quick=True)
        with patch("experiment_models.estimator", return_value=pipeline), patch.object(
            pipeline, "fit", wraps=pipeline.fit) as fitted:
            fit_member(spec, train, ["signal", "noise"], quick=True)
        weights = fitted.call_args.kwargs["model__sample_weight"]
        positive = train.target_return.gt(0).to_numpy()
        self.assertAlmostEqual(float(weights[positive].sum()), len(train) / 2)
        self.assertAlmostEqual(float(weights[~positive].sum()), len(train) / 2)
        self.assertAlmostEqual(float(weights.mean()), 1)

    def test_maturity_purges_targets_ending_at_refit_cutoff(self):
        frame = synthetic_train()
        cutoff = frame.Date.iloc[120]
        frame.loc[20, "target_end_date"] = cutoff
        frame.loc[21, "target_return"] = np.nan
        frame.loc[22, "target_end_date"] = pd.NaT
        mature = mature_training_rows(frame, cutoff)
        self.assertNotIn(20, mature.index)
        self.assertNotIn(21, mature.index)
        self.assertNotIn(22, mature.index)
        self.assertTrue(mature.Date.lt(cutoff).all())
        self.assertTrue(mature.target_end_date.lt(cutoff).all())
        self.assertTrue(mature.target_return.notna().all())

    def test_serialized_bundle_preserves_equal_ensemble_predictions(self):
        train = synthetic_train()
        specs = [Spec(f"class_{family}", "basic", "class", family, 0)
                 for family in ("linear", "xgb", "forest")]
        members = {spec.name: fit_member(spec, train, ["signal", "noise"], quick=True) for spec in specs}
        recipe = {"task": "class", "weights": {spec.name: 1 / 3 for spec in specs}, "threshold": .5}
        expected = np.mean([predict_member(members[spec.name], train.iloc[-5:]) for spec in specs], axis=0)
        np.testing.assert_allclose(predict_recipe(recipe, members, train.iloc[-5:]), expected)
        bundle = {"recipes": {"basic": recipe}, "members": members, "horizon": 30, "unit": "calendar"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.joblib"
            joblib.dump(bundle, path)
            restored = joblib.load(path)
        np.testing.assert_allclose(predict_recipe(restored["recipes"]["basic"], restored["members"], train.iloc[-5:]), expected)

    def test_regression_score_preserves_return_sign_without_probability_losses(self):
        values = np.array([-1e5, -.01, 0, .01, 1e5])
        np.testing.assert_array_equal(regression_direction_score(values) >= .5, values >= 0)
        frame = pd.DataFrame({"target_return": [-.04, -.03, 0, .02, .03]})
        result = direction_row(frame, values, classifier=False)
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["excluded_neutral_rows"], 1)
        self.assertEqual(result["accuracy"], 1)
        self.assertIsNone(result["brier"])
        self.assertIsNone(result["log_loss"])
        self.assertEqual(result["score_type"], "uncalibrated_return_rank_score")

    def test_recipe_variants_and_equal_weights_selected_from_pre_holdout_oof(self):
        oof, specs = synthetic_oof()
        recipes, ranking, decisions, abstention = select_recipes(oof, specs, {"basic": []}, pd.Timestamp("2022-01-01"))
        self.assertEqual(len(ranking), len(specs) + 2)
        for task in ("class", "reg"):
            for family in ("linear", "xgb", "forest"):
                self.assertEqual(recipes[f"{task}__basic__{family}"]["weights"], {f"{task}__basic__{family}_0": 1.})
            recipe = recipes[f"{task}__basic__equal_ensemble"]
            self.assertEqual(len(recipe["weights"]), 3)
            self.assertAlmostEqual(sum(recipe["weights"].values()), 1)
            self.assertTrue(all(name.endswith("_0") for name in recipe["weights"]))
            expected = np.mean(oof[list(recipe["weights"])].to_numpy(), axis=1)
            np.testing.assert_allclose(predict_oof(recipe, oof), expected)
        self.assertEqual(recipes["selected_direction"]["task"], "class")
        self.assertEqual(recipes["selected_return"]["task"], "reg")
        for decision in [*decisions.values(), abstention]:
            self.assertEqual(decision["selection_source"], "pre_holdout_oof")
            self.assertLess(pd.Timestamp(decision["selection_end"]), pd.Timestamp("2022-01-01"))

    def test_recipe_selection_rejects_holdout_origins_and_immature_oof_targets(self):
        for column in ("Date", "target_end_date"):
            oof, specs = synthetic_oof()
            oof.loc[oof.index[-1], column] = pd.Timestamp("2022-01-01")
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "pre-holdout|mature strictly before"):
                select_recipes(oof, specs, {"basic": []}, pd.Timestamp("2022-01-01"))

    def test_neutral_oof_rows_excluded_consistently_from_class_decisions(self):
        oof, specs = synthetic_oof()
        oof.loc[:4, "target_return"] = 0
        _, _, decisions, abstention = select_recipes(oof, specs, {"basic": []}, pd.Timestamp("2022-01-01"))
        self.assertTrue(all(decision["selection_rows"] == len(oof) - 5 for decision in decisions.values()))
        self.assertEqual(abstention["selection_rows"], len(oof) - 5)

    def test_all_cot_retains_sparse_constant_empty_fields_for_all_model_families(self):
        train = synthetic_train()
        train["cot_legacy_constant"] = 100.
        train["cot_disagg_sparse"] = np.nan
        train.loc[:3, "cot_disagg_sparse"] = [1., 2., 3., 4.]
        train["cot_disagg_empty"] = np.nan
        fields = ["cot_legacy_constant", "cot_disagg_sparse", "cot_disagg_empty"]
        for task in ("class", "reg"):
            for family in ("linear", "xgb", "forest"):
                spec = Spec(f"all_{task}_{family}", "basic", task, family, 0, require_all_cot=True)
                with self.subTest(task=task, family=family):
                    member = fit_member(spec, train, ["signal", "constant", "sparse", *fields], quick=True)
                    self.assertTrue(set(fields).issubset(member["features"]))
                    self.assertNotIn("constant", member["features"])
                    self.assertNotIn("sparse", member["features"])
                    self.assertEqual(member["required_cot_features"], fields)
                    self.assertTrue(all(item["included"] and item["required"] for item in member["cot_inclusion_audit"]))
                    self.assertTrue(member["cot_inclusion_audit"][-1]["empty_training_column"])
                    np.testing.assert_allclose(member["estimator"].named_steps["impute"].statistics_,
                                               [train.signal.median(), 100., 2.5, 0.])
                    latest = train.iloc[-3:].copy()
                    latest.loc[:, fields] = [200., 20., 9.]
                    self.assertTrue(np.isfinite(predict_member(member, latest)).all())
                    self.assertTrue(train.cot_disagg_empty.isna().all())

    def test_all_cot_group_omission_and_inference_schema_change_fail(self):
        train = synthetic_train()
        train["cot_legacy_signal"] = train.signal
        spec = Spec("all", "basic", "reg", "linear", 0, require_all_cot=True)
        with self.assertRaisesRegex(ValueError, "omits required"):
            fit_member(spec, train, ["signal"], quick=True)
        member = fit_member(spec, train, ["signal", "cot_legacy_signal"], quick=True)
        for changed in (train.drop(columns="cot_legacy_signal"), train.assign(cot_new_source_field=1.)):
            with self.assertRaisesRegex(ValueError, "inference schema"):
                predict_member(member, changed)

    def test_candidate_all_cot_requirement_is_explicit_for_both_tasks(self):
        specs = candidates({"basic": [], "direction_core": [], "engineered_direction": []}, require_all_cot=True)
        self.assertEqual(len(specs), 36)
        self.assertTrue(all(spec.require_all_cot for spec in specs))
        self.assertEqual({spec.task for spec in specs}, {"class", "reg"})


if __name__ == "__main__":
    unittest.main()
