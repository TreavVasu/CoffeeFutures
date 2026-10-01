from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("monthly_artifact_audit", ROOT / "MonthFu/scripts/audit_artifacts.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class AllCOTArtifactAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.folder = self.root / "MonthFu/artifacts/all_cot/30calendar"
        self.folder.mkdir(parents=True)
        (self.root / "data/COT").mkdir(parents=True)
        self.prices = pd.DataFrame({"Date": pd.bdate_range("2009-10-21", "2010-04-09")})
        self.prices["Close"] = 100 + np.arange(len(self.prices))
        self.labels = AUDIT.independent_labels(self.prices, 1, "sessions")
        self.required = ["cot_legacy_source_open_interest_all", "cot_disagg_source_open_interest_all"]
        inputs = self.prices.copy()
        for feature in self.required:
            inputs[feature] = np.log1p(100)
        inputs.to_csv(self.folder / "prediction_inputs.csv.gz", index=False)
        mature = AUDIT.mature_labels(self.labels, self.prices.Date.max()+pd.Timedelta(days=1))
        training = mature.merge(inputs[["Date", *self.required]], on="Date", validate="one_to_one")
        training.to_csv(self.folder / "training_dataset.csv.gz", index=False)
        pd.DataFrame({"source_dataset": ["legacy_futures_only", "disaggregated_futures_only"],
                      "Report_Date_as_MM_DD_YYYY": ["2000-01-04", "2009-08-25"],
                      "Open_Interest_All": [100, 100]}).to_csv(self.root / "data/COT/coffee_c_all_cot_data.csv", index=False)
        coverage = {
            family: {"included_source_fields": {"Open_Interest_All": {"feature": feature}},
                     "excluded_source_fields": {}, "source_fields_with_observations": ["Open_Interest_All"]}
            for family, feature in zip(["legacy_futures_only", "disaggregated_futures_only"], self.required)
        }
        source_manifest = {"feature_schema_version": 2, "features": 2, "reports": 2,
                           "excluded_prelaunch_disaggregated_backcasts": 0,
                           "retained_prelaunch_disaggregated_backcasts": 1, "source_field_coverage": coverage}
        self.schema = {"cot_policy": "all", "required_cot_features": self.required,
                       "source_field_manifest": source_manifest}
        (self.folder / "feature_schema.json").write_text(json.dumps(self.schema))
        self.selection = {"cot_policy": "all", "required_cot_features": self.required,
                          "eligible_candidates": ["complete"], "recipe": {"weights": {"complete": .5}}}
        self.specs = {"complete": {"require_all_cot": True}, "price": {"require_all_cot": False}}
        self.summary = {"cv_folds": 2}
        rows = []
        for stage, cutoffs in [("cv", ["2010-01-04", "2010-02-01"]),
                               ("holdout", ["2010-01-04", "2010-04-01"]),
                               ("final", ["2010-04-10"])]:
            for cutoff in cutoffs:
                for feature in self.required:
                    rows.append({"stage": stage, "cutoff": cutoff, "member": "complete",
                                 "comparison": "selected" if stage == "holdout" else None,
                                 "feature": feature, "observed_rows": 25, "training_rows": 50,
                                 "required": True, "retained": True})
        self.inclusion = pd.DataFrame(rows)
        self.inclusion.to_csv(self.folder / "cot_feature_inclusion.csv.gz", index=False)
        pd.DataFrame({"Date": pd.to_datetime(["2010-01-04", "2010-04-01"])}).to_csv(self.folder / "holdout_predictions.csv", index=False)

    def tearDown(self):
        self.temp.cleanup()

    def verify(self):
        with patch.object(AUDIT, "ROOT", self.root):
            return AUDIT.verify_all_cot_dataset(self.folder, self.prices, self.labels,
                                                self.summary, self.selection, self.specs)

    def test_complete_archive_matrices_and_fit_schema_pass(self):
        result = self.verify()
        self.assertEqual(result["reports"], 2)
        self.assertEqual(result["retained_backcasts"], 1)
        self.assertEqual(result["verified_feature_fit_rows"], 10)

    def test_a_dropped_cot_predictor_in_one_quarter_is_rejected(self):
        missing = self.inclusion.loc[~(self.inclusion.stage.eq("holdout") & self.inclusion.cutoff.eq("2010-04-01") & self.inclusion.feature.eq(self.required[0]))]
        missing.to_csv(self.folder / "cot_feature_inclusion.csv.gz", index=False)
        with self.assertRaisesRegex(AssertionError, "lacks required"):
            self.verify()

    def test_immature_training_rows_are_rejected(self):
        training = pd.read_csv(self.folder / "training_dataset.csv.gz")
        training.loc[len(training)] = training.iloc[-1]
        training.loc[len(training)-1, "Date"] = str(self.prices.Date.max().date())
        training.to_csv(self.folder / "training_dataset.csv.gz", index=False)
        with self.assertRaisesRegex(AssertionError, "label maturity"):
            self.verify()

    def test_unrecognized_omitted_source_fields_are_rejected(self):
        self.schema["source_field_manifest"]["source_field_coverage"]["legacy_futures_only"]["excluded_source_fields"]["New_measurement"] = "unrecognized_nonmeasurement_field"
        (self.folder / "feature_schema.json").write_text(json.dumps(self.schema))
        with self.assertRaisesRegex(AssertionError, "need review"):
            self.verify()

    def test_custom_artifact_root_preserves_existing_audit_document(self):
        monthfu = self.root / "MonthFu"
        docs = monthfu / "docs"
        docs.mkdir()
        previous = docs / "INDEPENDENT_AUDIT.md"
        previous.write_text("saved baseline audit")
        custom = monthfu / "artifacts/custom"
        custom.mkdir()
        with patch.object(AUDIT, "ROOT", self.root), patch.object(AUDIT, "MONTHFU", monthfu), \
                patch.object(AUDIT, "read_prices", return_value=self.prices), \
                patch.object(sys, "argv", ["audit", "--artifacts-dir", str(custom)]):
            AUDIT.main()
        self.assertEqual(previous.read_text(), "saved baseline audit")
        self.assertTrue((custom / "independent_audit.md").exists())


if __name__ == "__main__":
    unittest.main()
