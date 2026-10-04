from __future__ import annotations

import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MonthFu/experiments/src"))

from direction_features import cot_schema_hash, cot_source_policy  # noqa: E402
from experiment_models import Spec, fit_member  # noqa: E402
from MonthFu.experiments.scripts.predict import forecast, validate_source_schema  # noqa: E402
from MonthFu.experiments.scripts.train_direction import (  # noqa: E402
    artifact_directory, export_training_data,
)
from MonthFu.experiments.tests.test_experiment_models import synthetic_train  # noqa: E402


def all_cot_bundle():
    frame = synthetic_train()
    frame["cot_legacy_signal"] = frame.signal
    frame["cot_disagg_empty"] = np.nan
    columns = ["cot_legacy_signal", "cot_disagg_empty"]
    members, recipes = {}, {}
    for task, name in (("class", "selected_direction"), ("reg", "selected_return")):
        spec = Spec(name, "basic", task, "linear", 0, require_all_cot=True)
        members[name] = fit_member(spec, frame, ["signal", *columns], quick=True)
        recipes[name] = {"task": task, "weights": {name: 1.}, "threshold": .5}
    metadata = {"cot_policy": "all", "cot": {"alignment": "next observed session",
        "feature_schema_version": 2, "backcast_policy": "published history only"}}
    bundle = {"cot_policy": "all", "bundle_version": 2,
        "required_cot_features": columns, "cot_feature_schema_sha256": cot_schema_hash(columns),
        "cot_source_policy": cot_source_policy(metadata), "members": members, "recipes": recipes,
        "abstention": {"lower": .45, "upper": .55}, "horizon": 30,
        "unit": "calendar", "fitted_as_of": str(frame.Date.max().date())}
    frame["Close"] = 100.
    return bundle, frame, metadata


class AllCotInferenceTests(unittest.TestCase):
    def test_all_cot_forecast_validates_both_tasks_and_reports_requirement(self):
        bundle, frame, metadata = all_cot_bundle()
        rows = forecast(bundle, frame, metadata)
        self.assertEqual({row["model"] for row in rows}, {"selected_direction", "selected_return"})
        self.assertTrue(all(row["cot_policy"] == "all" and row["required_cot_feature_count"] == 2 for row in rows))
        self.assertTrue(np.isfinite(rows[0]["probability_up"]))
        self.assertTrue(np.isfinite(rows[1]["predicted_return"]))

    def test_schema_and_source_policy_changes_fail_before_inference(self):
        bundle, frame, metadata = all_cot_bundle()
        with self.assertRaisesRegex(ValueError, "source schema differs"):
            validate_source_schema(bundle, frame.assign(cot_new_field=1.), metadata)
        with self.assertRaisesRegex(ValueError, "source-policy metadata"):
            validate_source_schema(bundle, frame)
        altered = deepcopy(metadata)
        altered["cot"]["backcast_policy"] = "unsafe historical fill"
        with self.assertRaisesRegex(ValueError, "availability/source policy"):
            validate_source_schema(bundle, frame, altered)
        for name in bundle["members"]:
            broken = deepcopy(bundle)
            broken["members"][name]["features"].remove("cot_disagg_empty")
            with self.subTest(member=name), self.assertRaisesRegex(ValueError, "member omits"):
                validate_source_schema(broken, frame, metadata)

    def test_source_policy_ignores_report_counts_that_change_when_cache_refreshes(self):
        bundle, frame, metadata = all_cot_bundle()
        metadata["cot"]["reports"] = 100000
        metadata["cot"]["last_report_date"] = "2030-01-01"
        validate_source_schema(bundle, frame, metadata)

    def test_explicit_original_benchmark_bundle_still_has_compatible_schema_policy(self):
        validate_source_schema({"members": {}}, pd.DataFrame({"price_return": [1.]}))

    def test_training_exports_preserve_unknown_cells_and_only_mature_labels(self):
        frame = synthetic_train()
        frame["Date"] = pd.bdate_range("2008-01-02", periods=len(frame))
        frame["target_end_date"] = frame.Date + pd.Timedelta(days=30)
        frame["Close"] = 100.
        frame["target_close"] = 100 * (1 + frame.target_return)
        frame["target_sessions"] = 21.
        frame["forecast_steps"] = 21.
        frame["cot_legacy_signal"] = frame.signal
        frame.loc[0, "cot_legacy_signal"] = np.nan
        frame["cot_disagg_empty"] = np.nan
        groups = {"basic": ["signal", "cot_legacy_signal", "cot_disagg_empty"]}
        metadata = {"cot_policy": "all", "feature_manifest": [
            {"feature": name, "groups": ["basic"]} for name in groups["basic"]]}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            audit = export_training_data(out, frame, groups, metadata)
            saved = pd.read_csv(out/"training_dataset.csv.gz", parse_dates=["Date", "target_end_date"])
            origins = pd.read_csv(out/"features_and_targets.csv.gz")
            self.assertEqual(len(origins), len(frame))
            self.assertLess(len(saved), len(frame))
            self.assertTrue(saved.target_end_date.lt(frame.Date.max() + pd.Timedelta(days=1)).all())
            self.assertTrue(saved.cot_disagg_empty.isna().all())
            self.assertTrue(pd.isna(saved.cot_legacy_signal.iloc[0]))
            self.assertGreater(audit["pre_2009_training_rows"], 0)
            self.assertGreater(audit["pre_2009_observed_legacy_rows"], 0)
            self.assertTrue(audit["groups_include_entire_cot_bank"]["basic"])

    def test_new_runs_use_separate_paths_without_overwriting_prior_results(self):
        all_path = artifact_directory(30)
        benchmark = artifact_directory(30, cot_policy="benchmark")
        quick = artifact_directory(30, quick=True)
        self.assertEqual(str(all_path.relative_to(ROOT)), "MonthFu/experiments/artifacts/all_cot/direction/30calendar")
        self.assertEqual(str(benchmark.relative_to(ROOT)), "MonthFu/experiments/artifacts/benchmark_retrained/direction/30calendar")
        self.assertIn("all_cot/quick/30calendar", str(quick))
        self.assertNotEqual(all_path, ROOT / "MonthFu/experiments/artifacts/direction/30calendar")


if __name__ == "__main__":
    unittest.main()
