from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Scripts/project/src"))
from cot_release_features import build_cot_features


class COTAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        # Keep DATA inside the sandbox; sibling tests would otherwise share one
        # DATA directory and leak fixtures between cases.
        self.root = Path(self.temp.name) / "project"
        self.root.mkdir()
        (self.root / "../DATA/COT").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def write_reports(self, dates, values=None, family="legacy_futures_only"):
        values = list(values) if values is not None else list(range(1, len(dates) + 1))
        frame = pd.DataFrame({
            "Report_Date_as_MM_DD_YYYY": dates,
            "source_dataset": family,
            "Open_Interest_All": 100,
            "Comm_Positions_Long_All": values,
            "Comm_Positions_Short_All": 0,
            "M_Money_Positions_Long_ALL": values,
            "M_Money_Positions_Short_ALL": 0,
        })
        frame.to_csv(self.root / "../DATA/COT/coffee_c_all_cot_data.csv", index=False)
        return frame

    def test_friday_release_cannot_enter_friday_close_or_backfill(self):
        self.write_reports(["2024-01-09", "2024-01-16"], [10, 30])
        prices = pd.DataFrame({"Date": pd.bdate_range("2024-01-08", "2024-01-24")})
        features, audit, _ = build_cot_features(prices, self.root)
        series = features.set_index("Date")["cot_legacy_commercial_net_oi"]
        self.assertTrue(series.loc[:"2024-01-12"].isna().all())
        self.assertEqual(series.loc["2024-01-15"], 0.1)
        self.assertEqual(series.loc["2024-01-19"], 0.1)
        self.assertEqual(series.loc["2024-01-22"], 0.3)
        self.assertTrue((audit.first_usable_session > audit.publication_date).all())
        change = features.set_index("Date")["cot_legacy_commercial_net_change_1w"]
        self.assertAlmostEqual(change.loc["2024-01-22"], 0.2)
        self.assertAlmostEqual(change.loc["2024-01-24"], 0.2)

    def test_holiday_and_missing_session_delay(self):
        self.write_reports(["2024-11-26"])
        prices = pd.DataFrame({"Date": pd.to_datetime(["2024-11-29", "2024-12-02", "2024-12-04"])})
        features, audit, _ = build_cot_features(prices, self.root)
        self.assertEqual(audit.publication_date.iloc[0], pd.Timestamp("2024-12-02"))
        self.assertEqual(audit.first_usable_session.iloc[0], pd.Timestamp("2024-12-04"))
        self.assertTrue(features.cot_legacy_commercial_net_oi.iloc[:2].isna().all())

    def test_shutdown_and_ion_release_delays(self):
        for report, publication in [("2013-10-08", "2013-11-08"), ("2018-12-24", "2019-02-01"), ("2023-01-31", "2023-02-24"), ("2025-09-30", "2025-11-19"), ("2025-11-10", "2025-12-10")]:
            with self.subTest(report=report):
                self.write_reports([report])
                prices = pd.DataFrame({"Date": pd.bdate_range(report, pd.Timestamp(publication) + pd.Timedelta(days=6))})
                features, audit, _ = build_cot_features(prices, self.root)
                self.assertEqual(audit.publication_date.iloc[0], pd.Timestamp(publication))
                self.assertTrue(features.loc[features.Date.le(publication), "cot_legacy_commercial_net_oi"].isna().all())

    def test_backcast_history_cannot_seed_prelaunch_windows(self):
        frame = self.write_reports(["2009-08-25", "2009-09-01", "2009-09-08"], [99, 10, 30], "disaggregated_futures_only")
        legacy = frame.iloc[[0]].copy()
        legacy["source_dataset"] = "legacy_futures_only"
        pd.concat([frame, legacy]).to_csv(self.root / "../DATA/COT/coffee_c_all_cot_data.csv", index=False)
        prices = pd.DataFrame({"Date": pd.bdate_range("2009-08-24", "2009-09-15")})
        features, _, metadata = build_cot_features(prices, self.root)
        self.assertEqual(metadata["excluded_prelaunch_disaggregated_backcasts"], 1)
        self.assertTrue(features.loc[features.Date.le("2009-09-04"), "cot_disagg_managed_money_net_oi"].isna().all())
        self.assertTrue(features.loc[features.Date.eq("2009-09-07"), "cot_disagg_managed_money_net_change_1w"].isna().all())

    def test_future_values_cannot_change_previous_features(self):
        dates = pd.date_range("2021-01-05", periods=110, freq="W-TUE")
        frame = self.write_reports(dates, np.arange(110))
        prices = pd.DataFrame({"Date": pd.bdate_range("2021-01-01", "2023-04-01")})
        first, _, _ = build_cot_features(prices, self.root)
        frame.loc[100:, "Comm_Positions_Long_All"] = 100000
        frame.to_csv(self.root / "../DATA/COT/coffee_c_all_cot_data.csv", index=False)
        second, _, _ = build_cot_features(prices, self.root)
        mask = first.Date.lt(dates[100])
        assert_frame_equal(first.loc[mask], second.loc[mask])

    def test_out_of_order_override_does_not_leak_weekly_dependencies(self):
        self.write_reports(["2024-01-09", "2024-01-16"], [10, 30])
        pd.DataFrame({"report_date": ["2024-01-09"], "publication_date": ["2024-01-26"], "source": ["test confirmed delay"]}).to_csv(self.root / "../DATA/COT/cot_release_overrides.csv", index=False)
        prices = pd.DataFrame({"Date": pd.bdate_range("2024-01-08", "2024-01-30")})
        features, _, _ = build_cot_features(prices, self.root)
        self.assertTrue(features.loc[features.Date.le("2024-01-26"), "cot_legacy_commercial_net_change_1w"].isna().all())
        self.assertAlmostEqual(features.loc[features.Date.eq("2024-01-29"), "cot_legacy_commercial_net_change_1w"].iloc[0], 0.2)


if __name__ == "__main__":
    unittest.main()
