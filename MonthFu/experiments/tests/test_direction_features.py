from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from MonthFu.experiments.src.direction_features import (  # noqa: E402
    COT_GROUPS, WEATHER_REGIONS, _groups, _rank, _signed_streak,
    add_direction_features, build_experiment_frame,
)
from MonthFu.src.features import price_features  # noqa: E402


def synthetic_frame(rows: int = 420) -> pd.DataFrame:
    rng = np.random.default_rng(732)
    dates = pd.bdate_range("2014-01-02", periods=rows)
    close = 100 * np.exp(np.cumsum(rng.normal(.0002, .014, rows)))
    opening = close * np.exp(rng.normal(0, .002, rows))
    raw = pd.DataFrame({"Date": dates, "Close": close, "Open": opening,
                        "High": np.maximum(close, opening) * 1.012,
                        "Low": np.minimum(close, opening) * .99,
                        "Volume": rng.integers(1000, 40000, rows).astype(float)})
    frame = raw.merge(price_features(raw), on="Date")
    event_number = np.floor(np.arange(rows) / 5)
    cot_features = {}
    for i, group in enumerate(COT_GROUPS):
        net = .12 * np.sin(event_number / 8 + i)
        for kind, value in {"net_oi": net, "net_change_1w": .01 * np.cos(event_number / 8 + i),
                            "net_change_4w": .04 * np.cos(event_number / 8 + i),
                            "net_z_52w": net / .1, "index_52w": .5 + net * 3}.items():
            cot_features[f"cot_{group}_{kind}"] = value
    for family in ("legacy", "disagg"):
        age = np.arange(rows) % 5 + 3
        for kind, value in {"release_age_days": age, "snapshot_age_days": age + 3,
                            "new_release_session": (np.arange(rows) % 5 == 0).astype(float),
                            "dependency_delay_days": np.zeros(rows), "stale_14d": np.zeros(rows),
                            "oi_change_4w": np.zeros(rows)}.items():
            cot_features[f"cot_{family}_{kind}"] = value
    weather_features = {}
    for i, region in enumerate(WEATHER_REGIONS):
        for kind, value in {"rain_seasonal_z_30d": np.sin(np.arange(rows) / 20 + i),
                            "temp_seasonal_z_30d": np.cos(np.arange(rows) / 15 + i),
                            "soil_seasonal_z_30d": np.sin(np.arange(rows) / 40 + i),
                            "water_deficit_30d": np.arange(rows) % 60,
                            "dry_heat_stress_30d": np.arange(rows) % 20,
                            "rain_acceleration": np.sin(np.arange(rows) / 20),
                            "soil_change_30d": np.cos(np.arange(rows) / 40),
                            "min_temperature_7d": 8 + np.sin(np.arange(rows) / 10),
                            "source_age_days": np.full(rows, 5), "stale": np.zeros(rows)}.items():
            weather_features[f"weather_{region}_{kind}"] = value
    frame = pd.concat([frame, pd.DataFrame(cot_features), pd.DataFrame(weather_features)], axis=1)
    frame["news_daily_event_count_log"] = rng.normal(size=rows)
    frame["target_return_injected"] = rng.normal(size=rows)
    frame["future_return_injected"] = 1e9
    return frame


class DirectionFeatureTests(unittest.TestCase):
    def test_full_history_equals_every_prefix_predictor(self):
        frame = synthetic_frame()
        full, full_families = add_direction_features(frame)
        for cutoff in (150, 301):
            prefix, families = add_direction_features(frame.iloc[:cutoff])
            assert_frame_equal(full.iloc[:cutoff], prefix, check_exact=True)
            self.assertEqual(full_families, families)

    def test_future_price_cot_weather_mutation_preserves_earlier_rows(self):
        frame = synthetic_frame()
        changed = frame.copy()
        columns = changed.select_dtypes(include="number").columns
        changed.loc[301:, columns] = changed.loc[301:, columns] * 23 + 70
        baseline, _ = add_direction_features(frame)
        mutation, _ = add_direction_features(changed)
        assert_frame_equal(baseline.iloc[:301], mutation.iloc[:301], check_exact=True)
        self.assertNotEqual(baseline.iloc[301].direction_price_signed_streak_log,
                            mutation.iloc[301].direction_price_signed_streak_log)

    def test_all_groups_are_fixed_target_and_news_free(self):
        frame, families = add_direction_features(synthetic_frame())
        self.assertFalse(any(c.startswith(("target", "news_")) or "future_return" in c for c in frame))
        groups = _groups(frame, families)
        self.assertEqual(set(groups), {"basic", "direction_core", "price_direction", "cot_direction",
                                       "weather_direction", "engineered_direction"})
        self.assertEqual(len(groups["basic"]), 20)
        self.assertGreaterEqual(len(groups["direction_core"]), 40)
        self.assertLessEqual(len(groups["direction_core"]), 80)
        self.assertLessEqual(len(groups["engineered_direction"]), 300)
        for name, columns in groups.items():
            self.assertEqual(len(columns), len(set(columns)))
            self.assertTrue(set(columns).issubset(frame.columns))
            self.assertFalse(any(c.startswith(("target", "news_")) or "future_return" in c for c in columns))
            if name != "basic":
                self.assertTrue(set(groups["basic"]).issubset(columns))
        partial, partial_families = add_direction_features(synthetic_frame().iloc[:100])
        self.assertEqual(groups, _groups(partial, partial_families))

    def test_unknown_values_are_not_forward_or_backward_imputed(self):
        frame = synthetic_frame()
        frame.loc[200, "price_gap_return"] = np.nan
        frame.loc[210, "weather_brazil_minas_gerais_rain_seasonal_z_30d"] = np.nan
        result, _ = add_direction_features(frame)
        self.assertTrue(pd.isna(result.loc[200, "direction_price_gap_intraday_balance"]))
        self.assertTrue(pd.isna(result.loc[200, "direction_price_gap_intraday_disagreement"]))
        self.assertTrue(pd.isna(result.loc[210, "direction_weather_brazil_minas_gerais_drought_heat_severity"]))
        self.assertTrue(pd.isna(result.loc[210, "direction_weather_brazil_minas_gerais_stress_rank_126"]))
        self.assertTrue(result.loc[:4, "direction_price_prior_high_distance_atr_5"].isna().all())

    def test_weather_stale_current_values_and_interactions_remain_unknown(self):
        frame = synthetic_frame()
        frame.loc[200:220, "weather_brazil_minas_gerais_stale"] = 1
        result, families = add_direction_features(frame)
        brazil = [c for c in families["weather"] if c.startswith("direction_weather_brazil_")]
        cross_region = [c for c in families["weather"] if c.startswith("direction_weather_arabica_")]
        self.assertTrue(result.loc[200:220, brazil + cross_region].isna().all(axis=None))
        self.assertTrue(result.loc[199, brazil].notna().any())

    def test_cot_release_events_do_not_count_the_carried_daily_rows(self):
        frame = synthetic_frame()
        frame.loc[:4, "cot_legacy_new_release_session"] = 0
        result, _ = add_direction_features(frame)
        name = "direction_cot_legacy_noncommercial_released_net_rank_13"
        # Seven actual released states are required. The first is session 5.
        self.assertTrue(result.loc[:34, name].isna().all())
        self.assertTrue(result.loc[35:39, name].notna().all())
        self.assertEqual(result.loc[35, name], result.loc[39, name])
        self.assertTrue(result.loc[:4, "direction_cot_legacy_noncommercial_released_flow_streak"].isna().all())

    def test_missing_new_report_does_not_carry_previous_derived_report(self):
        frame = synthetic_frame()
        for kind in ("net_oi", "net_change_1w"):
            frame.loc[200:204, f"cot_legacy_noncommercial_{kind}"] = np.nan
        result, _ = add_direction_features(frame)
        columns = [c for c in result if c.startswith("direction_cot_legacy_noncommercial_released_")]
        self.assertTrue(result.loc[199, columns].notna().all())
        self.assertTrue(result.loc[200:204, columns].isna().all(axis=None))
        self.assertTrue(result.loc[205, columns].notna().all())

    def test_streak_unknown_or_zero_interruption_and_rank_ties(self):
        values = pd.Series([np.nan, .1, .2, 0, -.1, -.2, np.nan, -.1])
        expected = [np.nan, 1, 2, 0, -1, -2, np.nan, -1]
        np.testing.assert_allclose(_signed_streak(values), expected, equal_nan=True)
        rank = _rank(pd.Series([1., 1., 1., np.nan]), 3, 2)
        self.assertEqual(rank.iloc[2], .5)
        self.assertTrue(pd.isna(rank.iloc[3]))

    def test_build_api_attaches_existing_targets_only_after_features(self):
        base = synthetic_frame()
        audits = {"metadata": {"cot": {"alignment": "publication-aware"}},
                  "cot_release_audit": pd.DataFrame()}
        with patch("MonthFu.experiments.src.direction_features.build_feature_frame",
                   return_value=(base, {}, audits)):
            frame, groups, metadata = build_experiment_frame(ROOT, horizon=30, unit="calendar")
            sessions, _, _ = build_experiment_frame(ROOT, horizon=21, unit="sessions")
        self.assertIn("target_return", frame)
        self.assertNotIn("target_return_injected", frame)
        self.assertFalse(any(c.startswith("news_") for c in frame))
        self.assertTrue(frame.iloc[-20:].target_return.isna().all())
        self.assertEqual(frame.target_end_date.iloc[0], base.Date.loc[base.Date.ge(base.Date.iloc[0] + pd.Timedelta(days=30))].iloc[0])
        self.assertEqual(sessions.target_end_date.iloc[0], base.Date.iloc[21])
        self.assertEqual(metadata["feature_counts"], {name: len(columns) for name, columns in groups.items()})
        self.assertEqual(metadata["cot"]["alignment"], "publication-aware")

    def test_invalid_date_order_is_rejected(self):
        frame = synthetic_frame().iloc[:50].copy()
        frame.loc[10, "Date"] = frame.Date.iloc[9]
        with self.assertRaisesRegex(ValueError, "unique increasing"):
            add_direction_features(frame)

    def test_all_cot_policy_includes_every_source_field_in_each_bounded_group(self):
        source = synthetic_frame()
        source["cot_legacy_raw_sparse_trader_count"] = np.nan
        source.loc[20, "cot_legacy_raw_sparse_trader_count"] = 7.
        source["cot_disagg_raw_constant_percent"] = 100.
        source["cot_release_source_text"] = "metadata"
        frame, families = add_direction_features(source)
        benchmark = _groups(frame, families)
        groups = _groups(frame, families, cot_policy="all")
        self.assertEqual(set(groups), {"basic", "direction_core", "engineered_direction"})
        required = {c for c in frame if c.startswith("cot_") and pd.api.types.is_numeric_dtype(frame[c])}
        for name, columns in groups.items():
            self.assertTrue(required.issubset(columns))
            self.assertTrue(set(benchmark[name]).issubset(columns))
            self.assertNotIn("cot_release_source_text", columns)
        self.assertNotIn("cot_legacy_raw_sparse_trader_count", benchmark["basic"])

    def test_all_cot_build_preserves_missing_cells_and_records_schema(self):
        base = synthetic_frame()
        base["cot_disagg_raw_empty"] = np.nan
        audits = {"metadata": {"cot": {"alignment": "publication-aware"}}}
        with patch("MonthFu.experiments.src.direction_features.build_feature_frame",
                   return_value=(base, {}, audits)):
            frame, groups, metadata = build_experiment_frame(ROOT, cot_policy="all")
        self.assertEqual(metadata["cot_policy"], "all")
        self.assertIn("cot_disagg_raw_empty", metadata["cot_feature_names"])
        self.assertTrue(frame.cot_disagg_raw_empty.isna().all())
        self.assertEqual(len(metadata["cot_feature_schema_sha256"]), 64)
        self.assertTrue(all(set(metadata["cot_feature_names"]).issubset(c) for c in groups.values()))


if __name__ == "__main__":
    unittest.main()
