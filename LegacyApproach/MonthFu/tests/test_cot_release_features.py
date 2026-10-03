from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from MonthFu.src.cot_release_features import build_cot_features


class COTAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        # The loaders read DATA from the project root's parent, so the project
        # root must be a child of the sandbox. Sibling tests would otherwise
        # share one DATA directory and leak fixtures between cases.
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
        features, audit, metadata = build_cot_features(prices, self.root)
        self.assertEqual(metadata["excluded_prelaunch_disaggregated_backcasts"], 0)
        self.assertEqual(metadata["retained_prelaunch_disaggregated_backcasts"], 1)
        archive = audit.loc[audit.historical_backcast]
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive.publication_date.iloc[0], pd.Timestamp("2009-10-20"))
        self.assertTrue(features.loc[features.Date.le("2009-09-04"), "cot_disagg_managed_money_net_oi"].isna().all())
        self.assertTrue(features.loc[features.Date.eq("2009-09-07"), "cot_disagg_managed_money_net_change_1w"].isna().all())

    def test_pre_2009_legacy_reports_are_retained_and_used(self):
        self.write_reports(["2000-01-04", "2006-06-13", "2009-08-25"], [10, 20, 30])
        prices = pd.DataFrame({"Date": pd.bdate_range("2000-01-03", "2009-09-04")})
        features, audit, metadata = build_cot_features(prices, self.root)
        self.assertEqual(len(audit), 3)
        self.assertEqual(metadata["reports"], 3)
        self.assertFalse(audit.historical_backcast.any())
        series = features.set_index("Date").cot_legacy_commercial_net_oi
        self.assertEqual(series.loc["2000-01-10"], 0.1)
        self.assertEqual(series.loc["2006-06-19"], 0.2)
        self.assertEqual(series.loc["2009-08-31"], 0.3)

    def test_archive_activates_rolling_history_without_replacing_live_positions(self):
        dates = pd.date_range("2008-01-01", "2009-10-27", freq="W-TUE")
        frame = self.write_reports(dates, np.linspace(5, 90, len(dates)), "disaggregated_futures_only")
        prices = pd.DataFrame({"Date": pd.bdate_range("2008-01-01", "2009-11-03")})
        features, audit, metadata = build_cot_features(prices, self.root)
        series = features.set_index("Date")
        self.assertEqual(len(audit), len(frame))
        self.assertGreater(metadata["retained_prelaunch_disaggregated_backcasts"], 0)
        self.assertTrue(series.loc[:"2009-09-04", "cot_disagg_managed_money_net_oi"].isna().all())
        # Sep/early-Oct live reports continue to arrive before archive release.
        first = frame.loc[pd.to_datetime(frame.Report_Date_as_MM_DD_YYYY).eq("2009-09-01"), "M_Money_Positions_Long_ALL"].iloc[0] / 100
        self.assertAlmostEqual(series.loc["2009-09-07", "cot_disagg_managed_money_net_oi"], first)
        self.assertTrue(series.loc[:"2009-10-20", "cot_disagg_managed_money_net_z_52w"].isna().all())
        # The release enriches the latest snapshot's historical window on Oct21.
        self.assertTrue(np.isfinite(series.loc["2009-10-21", "cot_disagg_managed_money_net_z_52w"]))
        self.assertEqual(series.loc["2009-10-20", "cot_disagg_backcast_history_available"], 0)
        self.assertEqual(series.loc["2009-10-21", "cot_disagg_backcast_history_available"], 1)
        self.assertEqual(series.cot_disagg_backcast_history_new_session.sum(), 1)
        self.assertEqual(series.loc["2009-10-21", "cot_disagg_snapshot_age_days"], 8)
        self.assertEqual(series.loc["2009-10-20", "cot_disagg_managed_money_net_oi"], series.loc["2009-10-21", "cot_disagg_managed_money_net_oi"])
        # Altering backcasts cannot alter any prediction features before release.
        frame.loc[pd.to_datetime(frame.Report_Date_as_MM_DD_YYYY).lt("2009-09-01"), "M_Money_Positions_Long_ALL"] *= 10
        frame.to_csv(self.root / "../DATA/COT/coffee_c_all_cot_data.csv", index=False)
        changed, _, _ = build_cot_features(prices, self.root)
        assert_frame_equal(features.loc[features.Date.le("2009-10-20")], changed.loc[changed.Date.le("2009-10-20")])
        self.assertNotEqual(series.loc["2009-10-21", "cot_disagg_managed_money_net_z_52w"], changed.set_index("Date").loc["2009-10-21", "cot_disagg_managed_money_net_z_52w"])

    def test_all_numeric_measurements_have_explicit_source_features(self):
        frame = self.write_reports(["2024-01-09"], [10])
        measurements = {
            "NonComm_Postions_Spread_All": 5,
            "Comm_Positions_Long_Old": 40,
            "NonRept_Positions_Short_Other": 2,
            "Pct_of_OI_Comm_Long_Old": 35,
            "Traders_NonComm_Spead_Old": 4,
            "Conc_Gross_LE_8_TDR_Long_Other": 65,
            "Change_in_Comm_Long_All": -10,
            "Open_Interest_Other": 25,
        }
        for name, value in measurements.items():
            frame[name] = value
        frame["CFTC_Contract_Market_Code"] = 83731
        frame["As_of_Date_In_Form_YYMMDD"] = 240109
        frame.to_csv(self.root / "../DATA/COT/coffee_c_all_cot_data.csv", index=False)
        features, _, metadata = build_cot_features(pd.DataFrame({"Date": pd.bdate_range("2024-01-08", "2024-01-18")}), self.root)
        coverage = metadata["source_field_coverage"]["legacy_futures_only"]
        for name in measurements:
            with self.subTest(name=name):
                predictor = coverage["included_source_fields"][name]["feature"]
                self.assertTrue(features[predictor].notna().any())
        self.assertEqual(coverage["excluded_source_fields"]["CFTC_Contract_Market_Code"], "identifier_or_text_metadata")
        self.assertNotIn("M_Money_Positions_Long_ALL", coverage["included_source_fields"])
        self.assertAlmostEqual(features.cot_legacy_source_change_in_comm_long_all.dropna().iloc[0], -0.1)
        self.assertAlmostEqual(features.cot_legacy_source_pct_of_oi_comm_long_old.dropna().iloc[0], 0.35)

    def test_backcast_override_delays_expanded_history_but_preserves_live_reports(self):
        self.write_reports(["2009-08-25", "2009-09-01", "2009-09-08"], [90, 10, 30], "disaggregated_futures_only")
        pd.DataFrame({"report_date": ["2009-08-25"], "publication_date": ["2009-10-23"], "source": ["verified archive correction"]}).to_csv(self.root / "../DATA/COT/cot_release_overrides.csv", index=False)
        prices = pd.DataFrame({"Date": pd.bdate_range("2009-08-24", "2009-10-27")})
        features, audit, _ = build_cot_features(prices, self.root)
        series = features.set_index("Date")
        self.assertEqual(series.loc["2009-09-07", "cot_disagg_managed_money_net_oi"], 0.1)
        self.assertEqual(series.loc["2009-10-23", "cot_disagg_backcast_history_available"], 0)
        self.assertEqual(series.loc["2009-10-26", "cot_disagg_backcast_history_available"], 1)
        self.assertEqual(series.loc["2009-10-26", "cot_disagg_managed_money_net_oi"], 0.3)
        self.assertEqual(audit.loc[audit.historical_backcast, "raw_value_available_date"].iloc[0], pd.Timestamp("2009-10-23"))

    def test_future_report_prefix_is_invariant_with_activated_archive_history(self):
        dates = pd.date_range("2008-01-01", "2010-06-29", freq="W-TUE")
        frame = self.write_reports(dates, np.arange(len(dates)), "disaggregated_futures_only")
        prices = pd.DataFrame({"Date": pd.bdate_range("2008-01-01", "2010-07-09")})
        full, audit, _ = build_cot_features(prices, self.root)
        future_index = int(np.flatnonzero(dates >= pd.Timestamp("2010-04-06"))[0])
        cutoff = audit.loc[audit.report_date.eq(dates[future_index]), "first_usable_session"].iloc[0]
        frame.iloc[:future_index].to_csv(self.root / "../DATA/COT/coffee_c_all_cot_data.csv", index=False)
        truncated, _, _ = build_cot_features(prices, self.root)
        assert_frame_equal(full.loc[full.Date.lt(cutoff)], truncated.loc[truncated.Date.lt(cutoff)])
        frame.loc[future_index:, "M_Money_Positions_Long_ALL"] = 999999
        frame.to_csv(self.root / "../DATA/COT/coffee_c_all_cot_data.csv", index=False)
        changed, _, _ = build_cot_features(prices, self.root)
        assert_frame_equal(full.loc[full.Date.lt(cutoff)], changed.loc[changed.Date.lt(cutoff)])

    def test_future_values_cannot_change_previous_features(self):
        dates = pd.date_range("2021-01-05", periods=110, freq="W-TUE")
        frame = self.write_reports(dates, np.arange(110))
        prices = pd.DataFrame({"Date": pd.bdate_range("2021-01-01", "2023-04-01")})
        first, audit, _ = build_cot_features(prices, self.root)
        frame.loc[100:, "Comm_Positions_Long_All"] = 100000
        frame.to_csv(self.root / "../DATA/COT/coffee_c_all_cot_data.csv", index=False)
        second, _, _ = build_cot_features(prices, self.root)
        cutoff = audit.loc[audit.report_date.eq(dates[100]), "first_usable_session"].iloc[0]
        mask = first.Date.lt(cutoff)
        assert_frame_equal(first.loc[mask], second.loc[mask])

    def test_removing_future_reports_preserves_all_features_before_availability(self):
        dates = pd.date_range("2021-01-05", periods=110, freq="W-TUE")
        frame = self.write_reports(dates, np.arange(110))
        prices = pd.DataFrame({"Date": pd.bdate_range("2021-01-01", "2023-04-01")})
        full, audit, _ = build_cot_features(prices, self.root)
        frame.iloc[:100].to_csv(self.root / "../DATA/COT/coffee_c_all_cot_data.csv", index=False)
        truncated, _, _ = build_cot_features(prices, self.root)
        cutoff = audit.loc[audit.report_date.eq(dates[100]), "first_usable_session"].iloc[0]
        mask = full.Date.lt(cutoff)
        assert_frame_equal(full.loc[mask], truncated.loc[mask])

    def test_out_of_order_override_does_not_leak_weekly_dependencies(self):
        self.write_reports(["2024-01-09", "2024-01-16"], [10, 30])
        pd.DataFrame({"report_date": ["2024-01-09"], "publication_date": ["2024-01-26"], "source": ["test confirmed delay"]}).to_csv(self.root / "../DATA/COT/cot_release_overrides.csv", index=False)
        prices = pd.DataFrame({"Date": pd.bdate_range("2024-01-08", "2024-01-30")})
        features, _, _ = build_cot_features(prices, self.root)
        self.assertTrue(features.loc[features.Date.le("2024-01-26"), "cot_legacy_commercial_net_change_1w"].isna().all())
        self.assertAlmostEqual(features.loc[features.Date.eq("2024-01-29"), "cot_legacy_commercial_net_change_1w"].iloc[0], 0.2)

    def test_unusual_monday_snapshot_is_never_released_before_friday(self):
        self.write_reports(["2024-04-08"])
        prices = pd.DataFrame({"Date": pd.bdate_range("2024-04-08", "2024-04-16")})
        features, audit, _ = build_cot_features(prices, self.root)
        self.assertEqual(audit.publication_date.iloc[0], pd.Timestamp("2024-04-12"))
        self.assertTrue(features.loc[features.Date.le("2024-04-12"), "cot_legacy_commercial_net_oi"].isna().all())
        self.assertEqual(audit.first_usable_session.iloc[0], pd.Timestamp("2024-04-15"))

    def test_known_corrected_coffee_archive_waits_for_correction(self):
        self.write_reports(["2019-03-19", "2019-03-26", "2019-04-02"], [10, 30, 60])
        prices = pd.DataFrame({"Date": pd.bdate_range("2019-03-18", "2019-04-09")})
        features, audit, metadata = build_cot_features(prices, self.root)
        corrected = audit.loc[audit.report_date.eq("2019-03-26")].iloc[0]
        self.assertEqual(corrected.publication_date, pd.Timestamp("2019-03-29"))
        self.assertEqual(corrected.raw_value_available_date, pd.Timestamp("2019-04-03"))
        self.assertEqual(corrected.first_usable_session, pd.Timestamp("2019-04-04"))
        net = features.set_index("Date").cot_legacy_commercial_net_oi
        self.assertEqual(net.loc["2019-04-03"], 0.1)
        self.assertEqual(net.loc["2019-04-04"], 0.3)
        self.assertEqual(metadata["known_corrected_archive_reports"], 1)

    def test_announced_schedule_is_not_labelled_confirmed(self):
        self.write_reports(["2026-06-16"])
        prices = pd.DataFrame({"Date": pd.bdate_range("2026-06-15", "2026-06-24")})
        _, audit, _ = build_cot_features(prices, self.root)
        self.assertEqual(audit.publication_date.iloc[0], pd.Timestamp("2026-06-22"))
        self.assertEqual(audit.release_date_status.iloc[0], "announced_schedule")
        self.assertTrue(audit.release_date_estimated.iloc[0])

    def test_extra_federal_closures_match_official_exceptions(self):
        for report, publication in [("2008-12-22", "2008-12-29"), ("2014-12-23", "2014-12-30"), ("2020-12-21", "2020-12-28"), ("2021-06-15", "2021-06-21")]:
            with self.subTest(report=report):
                self.write_reports([report])
                prices = pd.DataFrame({"Date": pd.bdate_range(report, pd.Timestamp(publication) + pd.Timedelta(days=3))})
                features, audit, _ = build_cot_features(prices, self.root)
                self.assertEqual(audit.publication_date.iloc[0], pd.Timestamp(publication))
                self.assertTrue(features.loc[features.Date.le(publication), "cot_legacy_commercial_net_oi"].isna().all())

    def test_monthfu_override_has_precedence_and_validates_dates(self):
        self.write_reports(["2024-01-09"])
        (self.root / "MonthFu/data").mkdir(parents=True)
        override = self.root / "MonthFu/data/cot_release_overrides.csv"
        pd.DataFrame({"report_date": ["2024-01-09"], "publication_date": ["2024-01-17"], "source": ["verified date"]}).to_csv(override, index=False)
        prices = pd.DataFrame({"Date": pd.bdate_range("2024-01-08", "2024-01-22")})
        _, audit, _ = build_cot_features(prices, self.root)
        self.assertEqual(audit.publication_date.iloc[0], pd.Timestamp("2024-01-17"))
        self.assertEqual(audit.first_usable_session.iloc[0], pd.Timestamp("2024-01-18"))
        pd.DataFrame({"report_date": ["2024-01-09"], "publication_date": ["2024-01-08"], "source": ["invalid date"]}).to_csv(override, index=False)
        with self.assertRaisesRegex(ValueError, "cannot precede"):
            build_cot_features(prices, self.root)

    def test_reports_beyond_last_price_session_cannot_enter_daily_rows(self):
        self.write_reports(["2024-01-09", "2024-01-16", "2024-01-23"], [10, 30, 999999])
        prices = pd.DataFrame({"Date": pd.bdate_range("2024-01-08", "2024-01-19")})
        features, audit, _ = build_cot_features(prices, self.root)
        self.assertTrue(audit.loc[audit.report_date.ge("2024-01-16"), "first_usable_session"].isna().all())
        self.assertEqual(features.cot_legacy_commercial_net_oi.dropna().max(), 0.1)

    def test_unavailable_disaggregated_family_is_not_filled_from_legacy(self):
        legacy = self.write_reports(["2024-01-09"], [10])
        disagg = legacy.copy()
        disagg["source_dataset"] = "disaggregated_futures_only"
        disagg["Report_Date_as_MM_DD_YYYY"] = "2024-01-23"
        pd.concat([legacy, disagg]).to_csv(self.root / "../DATA/COT/coffee_c_all_cot_data.csv", index=False)
        prices = pd.DataFrame({"Date": pd.bdate_range("2024-01-08", "2024-01-22")})
        features, _, _ = build_cot_features(prices, self.root)
        self.assertTrue(features.cot_disagg_managed_money_net_oi.isna().all())
        self.assertEqual(features.cot_disagg_available.sum(), 0)
        self.assertGreater(features.cot_legacy_available.sum(), 0)


if __name__ == "__main__":
    unittest.main()
