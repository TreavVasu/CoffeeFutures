"""Checks for window-grid expansion and the separate-direction machinery."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "MonthFu/experiments/src"))

from polarity_metrics import (  # noqa: E402
    decide, evaluate_calls, margin_baseline, polarity_report, select_side_threshold,
)
from polarity_models import (  # noqa: E402
    DirectionSpec, candidates, polarity_banks, training_rows, usable_features,
)
from window_dataset import (  # noqa: E402
    DEFAULT_FORWARDS, DEFAULT_LOOKBACKS, Cell, build_grid, cell_frame, feature_columns_for,
)


def price_frame(rows: int = 400, start: str = "2016-01-04") -> pd.DataFrame:
    rng = np.random.default_rng(5150)
    close = 100 * np.exp(np.cumsum(rng.normal(.0004, .018, rows)))
    opening = close * np.exp(rng.normal(0, .003, rows))
    return pd.DataFrame({
        "Date": pd.bdate_range(start, periods=rows), "Close": close, "Open": opening,
        "High": np.maximum(close, opening) * 1.01, "Low": np.minimum(close, opening) * .99,
        "Volume": rng.integers(900, 40000, rows).astype(float),
    })


class GridTests(unittest.TestCase):
    def test_default_grid_is_the_full_expansion(self):
        self.assertEqual(DEFAULT_LOOKBACKS, (5, 10, 15, 21))
        self.assertEqual(DEFAULT_FORWARDS, (3, 5, 8, 10, 15, 18, 20))
        self.assertEqual(len(build_grid()), 28)

    def test_grid_is_the_cartesian_product_and_names_are_unique(self):
        cells = build_grid((5, 15), (3, 5))
        self.assertEqual([c.name for c in cells], ["L5_k3", "L5_k5", "L15_k3", "L15_k5"])
        self.assertEqual(len({c.name for c in build_grid()}), 28)

    def test_invalid_cells_are_rejected(self):
        for cell in [(1, 5), (15, 0)]:
            with self.assertRaises(ValueError):
                Cell(*cell)
        for args in [((5, 5), (3,)), ((), (3,))]:
            with self.assertRaises(ValueError):
                build_grid(*args)


class CellFrameTests(unittest.TestCase):
    def setUp(self):
        self.base = price_frame()

    def test_requested_examples_map_to_the_eighteenth_and_twentieth_sessions(self):
        # The request: 15 days of data predicting the 18th day, and the 20th.
        for offset, expected_session in ((3, 18), (5, 20)):
            frame = cell_frame(self.base, Cell(15, offset))
            origin = 14  # the 15th observed session
            self.assertEqual(frame.target_end_date.iloc[origin],
                             self.base.Date.iloc[origin + offset])
            self.assertEqual(origin + 1 + offset, expected_session)

    def test_endpoint_and_return_use_the_observed_calendar(self):
        for lookback, offset in [(5, 8), (15, 3), (21, 20), (10, 18)]:
            frame = cell_frame(self.base, Cell(lookback, offset))
            for index in (14, 60, 250):
                if index + offset >= len(self.base):
                    continue
                self.assertEqual(frame.target_end_date.iloc[index],
                                 self.base.Date.iloc[index + offset])
                expected = self.base.Close.iloc[index + offset] / self.base.Close.iloc[index] - 1
                self.assertAlmostEqual(float(frame.target_return.iloc[index]), float(expected), places=12)

    def test_rows_without_a_mature_endpoint_keep_a_nan_label(self):
        frame = cell_frame(self.base, Cell(15, 10))
        self.assertEqual(len(frame), len(self.base))
        self.assertTrue(frame.target_return.iloc[-10:].isna().all())
        self.assertTrue(frame.target_end_date.iloc[:-10].notna().all())
        # The tail is excluded by label, never silently dropped from the frame.
        self.assertEqual(int(frame.direction_up.notna().sum()), len(self.base) - 10)

    def test_directions_are_binary_complements_where_observed(self):
        frame = cell_frame(self.base, Cell(15, 5))
        observed = frame.direction_up.notna()
        self.assertEqual(int(frame.direction_down.notna().sum()), int(observed.sum()))
        both = observed & frame.direction_down.notna()
        np.testing.assert_allclose(frame.direction_up[both] + frame.direction_down[both], 1.0)
        self.assertTrue(np.isin(frame.direction_up.dropna().unique(), [0.0, 1.0]).all())

    def test_exactly_zero_forward_return_has_no_direction(self):
        base = price_frame(60)
        base.loc[30 + 5, "Close"] = base.loc[30, "Close"]  # force a zero forward return
        frame = cell_frame(base, Cell(15, 5))
        self.assertEqual(float(frame.target_return.iloc[30]), 0.0)
        self.assertTrue(pd.isna(frame.direction_up.iloc[30]))
        self.assertTrue(pd.isna(frame.direction_down.iloc[30]))
        self.assertFalse(pd.isna(frame.direction_up.iloc[31]))  # exclusion is row-local

    def test_lookback_window_bounds_are_recorded(self):
        frame = cell_frame(self.base, Cell(15, 5))
        self.assertEqual(frame.lookback_start_date.iloc[14], self.base.Date.iloc[0])
        self.assertEqual(frame.lookback_start_date.iloc[20], self.base.Date.iloc[6])
        self.assertTrue(frame.lookback_complete.iloc[:14].eq(False).all())
        self.assertTrue(frame.lookback_complete.iloc[14:].all())

    def test_unsorted_or_duplicate_dates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique increasing"):
            cell_frame(self.base.iloc[::-1], Cell(15, 5))


class FeatureColumnTests(unittest.TestCase):
    def test_a_cell_reads_only_its_own_aggregate_window(self):
        candles = ["candle_hammer", "candle_hammer_count@5", "candle_hammer_recency@5",
                   "candle_net_pressure@5", "candle_hammer_count@15",
                   "candle_net_pressure@15", "candle_net_pressure@21"]
        columns = feature_columns_for(Cell(15, 5), candles, ["price_return_5"])
        self.assertIn("candle_hammer", columns)          # current bar, untagged
        self.assertIn("candle_hammer_count@15", columns)
        self.assertNotIn("candle_hammer_count@5", columns)
        self.assertNotIn("candle_net_pressure@5", columns)
        self.assertNotIn("candle_net_pressure@21", columns)
        self.assertIn("price_return_5", columns)

    def test_targets_are_refused_in_a_feature_set(self):
        with self.assertRaises(AssertionError):
            feature_columns_for(Cell(15, 5), [], ["target_return", "news_x"])