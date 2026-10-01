from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MonthFu/experiments/src"))

from direction_features import cot_schema_hash, cot_source_policy  # noqa: E402
from experiment_models import Spec, fit_member, predict_member  # noqa: E402
from MonthFu.experiments.scripts.audit_direction_experiments import (  # noqa: E402
    _audit_all_cot_inclusion, _audit_final_replay,
)
from MonthFu.experiments.scripts.train_direction import export_training_data  # noqa: E402
from MonthFu.experiments.tests.test_experiment_models import synthetic_train  # noqa: E402


def fixture():
    frame = synthetic_train()
    frame["Close"] = 100.
    frame["target_close"] = 100 * (1 + frame.target_return)
    frame["target_sessions"] = 21.
    frame["forecast_steps"] = 21.
    frame["cot_legacy_signal"] = frame.signal
    frame["cot_disagg_empty"] = np.nan
    required = ["cot_legacy_signal", "cot_disagg_empty"]
    groups = {"basic": ["signal", *required]}
    metadata = {"cot_policy": "all", "cot": {"reports": 2, "alignment": "next session", "feature_schema_version": 2},
        "feature_manifest": [{"feature": c, "groups": ["basic"]} for c in groups["basic"]]}
    cutoff = frame.Date.max() + pd.Timedelta(days=1)
    train = frame.loc[frame.target_end_date.lt(cutoff)]
    members, recipes, fits = {}, {}, []
    for task in ("class", "reg"):
        name = f"{task}__basic__linear_0"
        model = fit_member(Spec(name, "basic", task, "linear", 0, require_all_cot=True), train, groups["basic"])
        members[name] = model
        alias = "selected_direction" if task == "class" else "selected_return"
        recipes[alias] = {"task": task, "weights": {name: 1.}, "threshold": .5}
        fits.append({"phase": "final", "candidate": name, "cutoff": str(cutoff.date()),
            "features": model["features"], "required_cot_features": model["required_cot_features"],
            "cot_inclusion_audit": model["cot_inclusion_audit"]})
    bundle = {"cot_policy": "all", "horizon": 30, "unit": "calendar", "fitted_as_of": str(frame.Date.max().date()),
        "required_cot_features": required, "cot_feature_schema_sha256": cot_schema_hash(required),
        "cot_source_policy": cot_source_policy(metadata), "members": members, "recipes": recipes}
    return frame, groups, metadata, bundle, fits, cutoff


class AllCotAuditTests(unittest.TestCase):
    def test_independent_all_cot_audit_checks_dataset_and_empty_input_inclusion(self):
        frame, groups, metadata, bundle, fits, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root/"data/COT"
            source_dir.mkdir(parents=True)
            pd.DataFrame({"source_dataset": ["legacy_futures_only", "disaggregated_futures_only"],
                "Report_Date_as_MM_DD_YYYY": ["2018-01-02", "2018-01-02"]}).to_csv(source_dir/"coffee_c_all_cot_data.csv", index=False)
            export_training_data(root, frame, groups, metadata)
            (root/"cot_inclusion_audit.json").write_text(json.dumps({"required_policy": True,
                "all_required_fields_retained_every_fit": True, "fit_count": len(fits)}))
            with patch("direction_features.build_experiment_frame", return_value=(frame, groups, metadata)), patch(
                "MonthFu.experiments.scripts.audit_direction_experiments.ROOT", root):
                checked = _audit_all_cot_inclusion(root, bundle, fits)
                self.assertEqual(checked["all_cot_source_reports_checked"], 2)
                self.assertEqual(checked["all_cot_required_features_per_fit"], 2)
                self.assertEqual(checked["all_cot_empty_training_fields"], ["cot_disagg_empty"])
                fits[0]["cot_inclusion_audit"][0]["included"] = False
                with self.assertRaises(AssertionError):
                    _audit_all_cot_inclusion(root, bundle, fits)
                fits[0]["cot_inclusion_audit"][0]["included"] = True
                stored = pd.read_csv(root/"training_dataset.csv.gz")
                stored.loc[0, "cot_disagg_empty"] = 0.
                stored.to_csv(root/"training_dataset.csv.gz", index=False, compression="gzip")
                with self.assertRaises(AssertionError):
                    _audit_all_cot_inclusion(root, bundle, fits)

    def test_independent_replay_understands_empty_imputer_placeholders_and_all_schema(self):
        frame, groups, metadata, bundle, _, cutoff = fixture()
        latest = []
        for alias, recipe in bundle["recipes"].items():
            name = next(iter(recipe["weights"]))
            value = float(predict_member(bundle["members"][name], frame.tail(1))[0])
            threshold = recipe["threshold"] if recipe["task"] == "class" else 0.
            latest.append({"model": alias, "value": value, "threshold": threshold,
                "direction": "Up" if value >= threshold else "Down"})
        with patch("direction_features.build_experiment_frame", return_value=(frame, groups, metadata)):
            checked = _audit_final_replay(Path("unused"), bundle, pd.DataFrame(latest), cutoff)
        self.assertEqual(checked["latest_forecasts_replayed"], 2)
        self.assertEqual(checked["train_only_preprocessors_checked"], 2)


if __name__ == "__main__":
    unittest.main()
