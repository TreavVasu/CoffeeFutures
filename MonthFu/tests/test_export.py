"""Regression check for interrupted export of neutral-forecast metrics."""
from pathlib import Path
import json
import sys
import unittest

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "scripts"))
from train import json_ready
from modeling import metrics


class ExportTests(unittest.TestCase):
    def test_undefined_correlations_roundtrip_as_json_null(self):
        records = pd.DataFrame([
            {"model":"zero",**metrics([.1,-.1],[0.,0.])},
            {"model":"signal",**metrics([.1,-.1],[.02,-.03])},
        ]).to_dict(orient="records")
        # DataFrame coercion replaces the first row's None with NaN.
        payload = json.loads(json.dumps(json_ready({"holdout_metrics":records}),allow_nan=False))
        self.assertIsNone(payload["holdout_metrics"][0]["correlation"])
        self.assertAlmostEqual(payload["holdout_metrics"][1]["correlation"],1.)
        self.assertEqual(payload["holdout_metrics"][0]["r2_vs_zero"],0.)


if __name__=="__main__":
    unittest.main()
