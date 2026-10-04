"""Checks for candlestick conditions: causality, trend context, unknown bars."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "MonthFu/experiments/src"))

from candlestick_patterns import (  # noqa: E402
    BEARISH_PATTERNS, BULLISH_PATTERNS, KNOWN_NAME_FORMULA_MISMATCH,
    PATTERN_BUILDERS, PATTERN_NAMES, CandleGeometry, PatternConfig,
    add_candlestick_features, candles_for_polarity, pattern_occurrences,
)


def bar(open_: float, high: float, low: float, close: float) -> dict:
    return {"Open": open_, "High": high, "Low": low, "Close": close}


def trend_frame(closes: list[float], *tail: dict) -> pd.DataFrame:
    """A history of small-range bars, then any explicit trailing bars.

    ``closes`` become doji-like bars (Open == Close); pass explicit ``tail`` bars
    when a pattern needs a directional bar such as an engulfing prior candle.
    """
    rows = [bar(c, c + .4, c - .4, c) for c in closes] + list(tail)
    frame = pd.DataFrame(rows)
    frame["Date"] = pd.bdate_range("2018-01-01", periods=len(frame))
    frame["Volume"] = 1000.0
    return frame


def hammer_frame(rising: bool) -> pd.DataFrame:
    """Same hammer-shaped final bar, opposite preceding trend.

    ``rising=True`` makes the prior closes ascend so the identical shape is a
    Hanging Man; ``rising=False`` makes them descend so it is a Hammer.  The
    only difference between the two frames is the direction of the trend.
    """
    closes = list(np.linspace(80, 99.5, 24)) if rising else list(np.linspace(110, 100.5, 24))
    # body 0.3, upper 0.1, lower 0.8 -> long lower shadow, small body
    return trend_frame(closes, bar(100.0, 100.4, 99.2, 100.3))


class PatternConfigTests(unittest.TestCase):
    def test_threshold_ordering_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "0 < doji <= small <= long"):
            PatternConfig(long_body=0.3, small_body=0.5, doji_body=0.1)

    def test_windows_and_ratios_are_validated(self):
        for kwargs in [{"atr_window": 1}, {"sma_window": 1}, {"trend_lag": 0},
                       {"small_shadow_ratio": 1.5}, {"equal_atr": -0.1}]:
            with self.assertRaises(ValueError):
                PatternConfig(**kwargs)

    def test_every_reference_pattern_is_registered_with_one_polarity(self):
        self.assertEqual(len(PATTERN_BUILDERS), 38)
        self.assertEqual(len(BULLISH_PATTERNS), 21)
        self.assertEqual(len(BEARISH_PATTERNS), 17)
        self.assertEqual(set(PATTERN_NAMES), set(PATTERN_BUILDERS))
        self.assertEqual(set(BULLISH_PATTERNS) & set(BEARISH_PATTERNS), set())


class TrendContextTests(unittest.TestCase):
    """Hammer and Hanging Man share a shape; only the trend separates them."""

    def test_identical_shape_is_hammer_in_downtrend_and_hanging_man_in_uptrend(self):
        down = pattern_occurrences(hammer_frame(rising=False))
        up = pattern_occurrences(hammer_frame(rising=True))
        self.assertEqual(down.candle_hammer.iloc[-1], 1.0)
        self.assertEqual(down.candle_hanging_man.iloc[-1], 0.0)
        self.assertEqual(up.candle_hanging_man.iloc[-1], 1.0)
        self.assertEqual(up.candle_hammer.iloc[-1], 0.0)

    def test_inverted_hammer_and_shooting_star_separate_only_by_trend(self):
        # open 100.0, close 100.3 -> body 0.3; high 101.0 -> upper 0.7; low 99.9 -> lower 0.1
        last = bar(100.0, 101.0, 99.9, 100.3)
        down = pattern_occurrences(trend_frame(list(np.linspace(110, 100.5, 24)), last))
        up = pattern_occurrences(trend_frame(list(np.linspace(80, 99.5, 24)), last))
        self.assertEqual(down.candle_inverted_hammer.iloc[-1], 1.0)
        self.assertEqual(down.candle_shooting_star.iloc[-1], 0.0)
        self.assertEqual(up.candle_shooting_star.iloc[-1], 1.0)
        self.assertEqual(up.candle_inverted_hammer.iloc[-1], 0.0)


class GeometryTests(unittest.TestCase):
    def test_unusable_bars_are_unknown_rather_than_absent(self):
        frame = trend_frame(list(np.linspace(100, 110, 24)), bar(120.0, 119.0, 118.0, 121.0))
        self.assertFalse(bool(CandleGeometry(frame).valid.iloc[-1]))  # Low below a body above High
        occurrences = pattern_occurrences(frame)
        self.assertTrue(pd.isna(occurrences.iloc[-1]).all())
        self.assertTrue(occurrences.iloc[:-1].notna().all().all())

    def test_non_positive_prices_are_unusable(self):
        frame = trend_frame(list(np.linspace(100, 110, 24)), bar(-5.0, 0.0, -6.0, -4.0))
        self.assertTrue(pd.isna(pattern_occurrences(frame).iloc[-1]).all())

    def test_engulfing_requires_the_prior_body_to_be_covered(self):
        # Prior long bear 100 -> 98; current bull opens at 97.5 and closes at 100.5.
        history = list(np.linspace(130, 100.5, 22))
        prior_bear = bar(100.0, 100.1, 97.9, 98.0)
        engulfing = bar(97.5, 100.6, 97.4, 100.5)
        self.assertEqual(pattern_occurrences(trend_frame(history, prior_bear, engulfing))
                         .candle_bullish_engulfing.iloc[-1], 1.0)
        # A current bar that fails to reclaim the prior open is not an engulfing.
        partial = bar(97.5, 99.6, 97.4, 99.0)
        self.assertEqual(pattern_occurrences(trend_frame(history, prior_bear, partial))
                         .candle_bullish_engulfing.iloc[-1], 0.0)

    def test_duplicate_or_unsorted_dates_are_rejected(self):
        frame = trend_frame(list(np.linspace(100, 110, 24)), bar(100.0, 100.4, 99.7, 100.1))
        with self.assertRaisesRegex(ValueError, "unique increasing"):
            pattern_occurrences(frame.iloc[::-1].reset_index(drop=True))

    def test_name_formula_mismatches_are_declared(self):
        self.assertIn("mat_hold", KNOWN_NAME_FORMULA_MISMATCH)
        self.assertIn("three_line_strike_bull", KNOWN_NAME_FORMULA_MISMATCH)
        self.assertTrue(set(KNOWN_NAME_FORMULA_MISMATCH) <= set(PATTERN_BUILDERS))


class CausalityTests(unittest.TestCase):
    """Appending or mutating the future must not move an earlier value."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20260904)
        rows = 520
        close = 100 * np.exp(np.cumsum(rng.normal(0, .015, rows)))
        opening = close * np.exp(rng.normal(0, .003, rows))
        cls.base = pd.DataFrame({
            "Date": pd.bdate_range("2015-01-01", periods=rows), "Close": close,
            "Open": opening, "High": np.maximum(close, opening) * 1.012,
            "Low": np.minimum(close, opening) * .989,
            "Volume": rng.integers(900, 40000, rows).astype(float),
        })

    def test_prefix_rebuild_matches_the_full_frame(self):
        full, _ = add_candlestick_features(self.base, windows=(5, 10, 15, 21))
        for cut in (200, 371):
            prefix, _ = add_candlestick_features(self.base.iloc[:cut], windows=(5, 10, 15, 21))
            assert_frame_equal(prefix, full.iloc[:cut].reset_index(drop=True))

    def test_mutating_future_rows_leaves_earlier_values_equal(self):
        full, _ = add_candlestick_features(self.base, windows=(5, 10, 15, 21))
        mutated = self.base.copy()
        mutated.loc[mutated.index >= 300, ["Open", "High", "Low", "Close"]] *= 1.4
        changed, _ = add_candlestick_features(mutated, windows=(5, 10, 15, 21))
        assert_frame_equal(full[full.index < 300].reset_index(drop=True),
                           changed[changed.index < 300].reset_index(drop=True))

    def test_no_feature_carries_a_target_or_news_column(self):
        frame = self.base.assign(target_return=1.0, news_daily_count=2.0,
                                 some_future_return=3.0)
        _, names = add_candlestick_features(frame, windows=(5,))
        self.assertFalse(any(n.startswith(("target", "news_")) for n in names))
        self.assertFalse(any("future_return" in n for n in names))


class WindowTagTests(unittest.TestCase):
    def setUp(self):
        self.frame = hammer_frame(rising=False)

    def test_counts_sum_pattern_hits_inside_the_window(self):
        tagged_frame, names = add_candlestick_features(self.frame, windows=(5, 21))
        occurrences = pattern_occurrences(self.frame)
        for window in (5, 21):
            for name in PATTERN_NAMES:
                column = occurrences[f"candle_{name}"].fillna(0.0)
                expected = column.rolling(window, min_periods=1).sum().iloc[-1]
                self.assertAlmostEqual(float(expected),
                                       float(tagged_frame.iloc[-1][f"candle_{name}_count@{window}"]),
                                       places=10, msg=f"{name}@{window}")
        self.assertTrue(any(t.endswith("@5") for t in names))
        self.assertTrue(any(t.endswith("@21") for t in names))

    def test_net_pressure_is_bull_minus_bear_count(self):
        tagged, _ = add_candlestick_features(self.frame, windows=(10,))
        last = tagged.iloc[-1]
        self.assertAlmostEqual(float(last["candle_net_pressure@10"]),
                               float(last["candle_bull_count@10"] - last["candle_bear_count@10"]),
                               places=10)

    def test_recency_is_sessions_since_last_hit_and_respects_the_window(self):
        occurrences = pattern_occurrences(self.frame)
        fired = np.flatnonzero(occurrences.candle_hammer.fillna(0).to_numpy() > 0)
        tagged, _ = add_candlestick_features(self.frame, windows=(5, 21))
        for window in (5, 21):
            recent = fired[fired >= len(self.frame) - window]
            expected = float(len(self.frame) - 1 - recent.max()) if len(recent) else np.nan
            got = float(tagged.iloc[-1][f"candle_hammer_recency@{window}"])
            if np.isnan(expected):
                self.assertTrue(np.isnan(got))
            else:
                self.assertAlmostEqual(got, expected, places=10, msg=f"recency@{window}")

    def test_bad_windows_are_rejected(self):
        for windows in [(1,), (5, 5)]:
            with self.assertRaises(ValueError):
                add_candlestick_features(self.frame, windows=windows)


class PolarityBankTests(unittest.TestCase):
    def test_each_side_receives_only_its_own_patterns(self):
        _, names = add_candlestick_features(hammer_frame(rising=False), windows=(5,))
        bull, bear = candles_for_polarity(names, "bull"), candles_for_polarity(names, "bear")
        self.assertEqual(set(bull) & set(bear), set())
        # Every returned column must name a pattern registered to that side.
        bull_stems = {f"candle_{n}" for n in BULLISH_PATTERNS}
        bear_stems = {f"candle_{n}" for n in BEARISH_PATTERNS}
        for column in bull:
            self.assertTrue(any(stem in column for stem in bull_stems), column)
        for column in bear:
            self.assertTrue(any(stem in column for stem in bear_stems), column)
        # Hammer is a bullish pattern even though its name carries no "bull".
        self.assertIn("candle_hammer", bull)
        self.assertIn("candle_hanging_man", bear)
        self.assertNotIn("candle_hanging_man", bull)
        # Shared pressure summaries belong to neither side's exclusive bank.
        self.assertNotIn("candle_net_pressure@5", bull + bear)
        with self.assertRaises(ValueError):
            candles_for_polarity(names, "sideways")


if __name__ == "__main__":
    unittest.main()