"""Causal candlestick pattern conditions for the observed Coffee C session frame.

Candlestick patterns are *boolean conditions* on Open/High/Low/Close, optionally
gated by a trend filter.  They are not a score and not a universal formula, so
every threshold used here is frozen in :class:`PatternConfig` before any model
sees a row.

Causality
---------
Forecast origins sit after the observed session close.  Every condition reads
only bars at or before its own index, so appending or mutating future rows can
never change an earlier value.  :func:`add_candlestick_features` is required to
satisfy that prefix invariance; ``tests/test_candlestick_patterns.py`` checks it
on truncated prefixes.

Unknown is not False
--------------------
``MonthFu.src.features.load_prices`` masks inconsistent OHLC (High below the
body, Low above the body, non-positive or non-finite values).  A masked bar
cannot support any shape claim, so conditions return ``NaN`` for that bar rather
than ``False``.  Roughly 11% of this dataset has an unusable bar; silently
scoring those as "no pattern" would understate pattern incidence and would tell
the model a measurement was made when it was not.

Naming
------
The supplied reference table describes *Bullish Three-Line Strike* with the
bearish three-line-strike condition and vice versa, and *Mat Hold* with the
identical condition to *Rising Three Methods*.  This module implements the
**formulas as written** and keeps the reference names so the feature manifest
maps one-to-one onto the supplied tables.  :data:`KNOWN_NAME_FORMULA_MISMATCH`
records every case for the report rather than silently renaming a column.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PatternConfig:
    """Frozen thresholds for every shape test.

    Defaults are the reference values: a long body is at least 60% of the
    range, a small body at most 30%, and a doji at most 10%.  ``equal`` compares
    two prices within the larger of 0.1% of the reference price, 0.1 ATR, and
    one tick.  Trend filters compare the previous close against the close five
    sessions earlier and against a 20-session simple moving average.
    """

    long_body: float = 0.6
    small_body: float = 0.3
    doji_body: float = 0.1
    equal_pct: float = 0.001
    equal_atr: float = 0.1
    tick_size: float = 0.05
    atr_window: int = 14
    sma_window: int = 20
    trend_lag: int = 5
    small_shadow_ratio: float = 0.3

    def __post_init__(self) -> None:
        if not 0 < self.doji_body <= self.small_body <= self.long_body <= 1:
            raise ValueError("Body thresholds must satisfy 0 < doji <= small <= long <= 1.")
        if self.atr_window < 2 or self.sma_window < 2 or self.trend_lag < 1:
            raise ValueError("Windows must be at least two sessions and the trend lag at least one.")
        if self.equal_pct < 0 or self.equal_atr < 0 or self.tick_size < 0:
            raise ValueError("Equality tolerances must be nonnegative.")
        if not 0 <= self.small_shadow_ratio <= 1:
            raise ValueError("The small-shadow ratio must lie in [0, 1].")


# The reference tables reuse one formula under two names and swap the polarity of
# the three-line strikes.  Implemented as written; recorded here for the report.
KNOWN_NAME_FORMULA_MISMATCH: dict[str, str] = {
    "three_line_strike_bull": "Implemented with the condition printed under 'Bullish Three-Line Strike' "
                               "(three up closes then a long bear that closes below the first open). "
                               "That condition is the standard BEARISH three-line strike; the reference "
                               "table's label and formula disagree.",
    "three_line_strike_bear": "Implemented with the condition printed under 'Bearish Three-Line Strike' "
                               "(three down closes then a bull that closes above the first open). "
                               "That condition is the standard BULLISH three-line strike.",
    "mat_hold": "Identical condition to rising_three_methods in the reference table.",
}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"Candlestick patterns require an {column} column.")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def FMAX(left: pd.Series, right: pd.Series) -> pd.Series:
    """Elementwise maximum that propagates NaN, unlike ``Series.combine_max``.

    Body highs and lows must stay unknown when either input is unknown; silently
    treating a missing bar as a bound would fabricate a shape.
    """
    return pd.Series(np.fmax(left.to_numpy(dtype=float), right.to_numpy(dtype=float)),
                     index=left.index, dtype=float)


def FMIN(left: pd.Series, right: pd.Series) -> pd.Series:
    """Elementwise minimum that propagates NaN."""
    return pd.Series(np.fmin(left.to_numpy(dtype=float), right.to_numpy(dtype=float)),
                     index=left.index, dtype=float)


def _valid_ohlc(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """High above the body, Low below it, and every price finite and positive."""
    return (high >= FMAX(open_, close)) & (low <= FMIN(open_, close)) & (
        open_ > 0) & (high > 0) & (low > 0) & (close > 0)


def _average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    previous = close.shift(1)
    true_range = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
    return true_range.rolling(window, min_periods=max(2, window // 2)).mean()


class CandleGeometry:
    """Per-bar shape quantities, all derived from OHLC alone.

    ``valid`` marks bars whose geometry is measurable.  Boolean conditions are
    masked with it, so an unusable bar reports ``NaN`` instead of ``False``.
    """

    def __init__(self, frame: pd.DataFrame, config: PatternConfig | None = None) -> None:
        self.config = config or PatternConfig()
        self.open = _numeric(frame, "Open")
        self.high = _numeric(frame, "High")
        self.low = _numeric(frame, "Low")
        self.close = _numeric(frame, "Close")
        self.valid = _valid_ohlc(self.open, self.high, self.low, self.close)

        self.body = (self.close - self.open).abs()
        self.upper = self.high - FMAX(self.open, self.close)
        self.lower = FMIN(self.open, self.close) - self.low
        self.spread = self.high - self.low
        self.mid = (self.open + self.close) / 2
        self.bull = self.close > self.open
        self.bear = self.close < self.open
        # A zero-range bar cannot support a ratio test, so shape conditions that
        # need one stay unavailable rather than becoming trivially true.
        self.scalable = self.spread > 0

        self.long_body = self.body >= self.config.long_body * self.spread
        self.small_body = self.body <= self.config.small_body * self.spread
        self.doji = self.body <= self.config.doji_body * self.spread

        self.atr = _average_true_range(self.high, self.low, self.close, self.config.atr_window)
        sma = self.close.rolling(self.config.sma_window, min_periods=self.config.sma_window).mean()
        self.previous_close = self.close.shift(1)
        self.gap_up = self.low > self.high.shift(1)
        self.gap_down = self.high < self.low.shift(1)
        self.uptrend = ((self.previous_close > self.close.shift(self.config.trend_lag))
                        & (self.previous_close > sma.shift(1)))
        self.downtrend = ((self.previous_close < self.close.shift(self.config.trend_lag))
                          & (self.previous_close < sma.shift(1)))

    def equal(self, left: pd.Series, right: pd.Series) -> pd.Series:
        """Reference equality: within 0.1% of price, 0.1 ATR, or one tick."""
        tolerance = np.maximum.reduce([
            self.config.equal_pct * right.abs(),
            self.config.equal_atr * self.atr,
            pd.Series(self.config.tick_size, index=left.index, dtype=float),
        ])
        return (left - right).abs() <= tolerance

    def inside_body(self, current: int = 0, previous: int = 1) -> pd.Series:
        """Both the current open and close strictly inside the prior body."""
        outer_low = FMIN(self.open.shift(previous), self.close.shift(previous))
        outer_high = FMAX(self.open.shift(previous), self.close.shift(previous))
        inner_low = FMIN(self.open.shift(current), self.close.shift(current))
        inner_high = FMAX(self.open.shift(current), self.close.shift(current))
        return (inner_low > outer_low) & (inner_high < outer_high)

    def bull_engulf(self, current: int = 0, previous: int = 1) -> pd.Series:
        return ((self.open.shift(current) <= self.close.shift(previous))
                & (self.close.shift(current) >= self.open.shift(previous)))

    def bear_engulf(self, current: int = 0, previous: int = 1) -> pd.Series:
        return ((self.open.shift(current) >= self.close.shift(previous))
                & (self.close.shift(current) <= self.open.shift(previous)))

    def shifted(self, name: str, offset: int) -> pd.Series:
        return getattr(self, name).shift(offset)


def _small_upper_shadows(geometry: CandleGeometry, offsets: tuple[int, ...]) -> pd.Series:
    """Upper shadows at each offset are a small fraction of that bar's range.

    Three White Soldiers specify small *upper* shadows only; requiring both
    shadows to be small would be stricter than the reference formula.
    """
    result = pd.Series(True, index=geometry.spread.index)
    ratio = geometry.config.small_shadow_ratio
    for offset in offsets:
        result &= geometry.shifted("upper", offset) <= ratio * geometry.shifted("spread", offset)
    return result


def _small_lower_shadows(geometry: CandleGeometry, offsets: tuple[int, ...]) -> pd.Series:
    """Lower shadows at each offset are a small fraction of that bar's range.

    Three Black Crows specify small *lower* shadows only.
    """
    result = pd.Series(True, index=geometry.spread.index)
    ratio = geometry.config.small_shadow_ratio
    for offset in offsets:
        result &= geometry.shifted("lower", offset) <= ratio * geometry.shifted("spread", offset)
    return result


def _open_inside_previous_body(geometry: CandleGeometry, offset: int) -> pd.Series:
    """The open at ``offset`` sits strictly inside the preceding bar's body."""
    body_low = FMIN(geometry.shifted("open", offset + 1), geometry.shifted("close", offset + 1))
    body_high = FMAX(geometry.shifted("open", offset + 1), geometry.shifted("close", offset + 1))
    opening = geometry.shifted("open", offset)
    return (opening > body_low) & (opening < body_high)


def _closes_advance(geometry: CandleGeometry, offsets: tuple[int, ...], up: bool) -> pd.Series:
    """Closes form a strictly monotonic run across ``offsets``.

    ``offsets`` are counted backwards in time (``(2, 1, 0)`` is oldest first), so
    an advancing sequence requires the *older* close to be *lower*.  For ``up``
    that means ``C[i-2] < C[i-1] < C[i]``; for a bear run it is reversed.
    """
    result = pd.Series(True, index=geometry.spread.index)
    for earlier, later in zip(offsets[:-1], offsets[1:]):
        result &= (geometry.shifted("close", earlier) < geometry.shifted("close", later) if up
                   else geometry.shifted("close", earlier) > geometry.shifted("close", later))
    return result


def _three_soldiers(geometry: CandleGeometry, up: bool) -> pd.Series:
    """Three White Soldiers / Three Black Crows.

    Three consecutive same-direction bars whose closes advance, each open inside
    the prior body, with the shadows on the exhausted side held small.  The
    reference formula requires only *direction* on the three bars, not long
    bodies, so no body-size gate is applied here.
    """
    offsets = (2, 1, 0)
    same_direction = (geometry.shifted("bull", 2) & geometry.shifted("bull", 1) & geometry.bull if up
                      else geometry.shifted("bear", 2) & geometry.shifted("bear", 1) & geometry.bear)
    shadows = _small_upper_shadows if up else _small_lower_shadows
    return (same_direction & _closes_advance(geometry, offsets, up)
            & _open_inside_previous_body(geometry, 1) & _open_inside_previous_body(geometry, 0)
            & shadows(geometry, offsets))


def _three_inside_body(geometry: CandleGeometry, up: bool) -> pd.Series:
    """Three Inside Up / Down: long first bar, inside second bar, then a break.

    The second bar must open in the direction of the third, which is what
    separates this from a harami.
    """
    outer_low = FMIN(geometry.shifted("open", 2), geometry.shifted("close", 2))
    outer_high = FMAX(geometry.shifted("open", 2), geometry.shifted("close", 2))
    second_inside = (geometry.shifted("open", 1) > outer_low) & (geometry.shifted("close", 1) > outer_low) & (
        geometry.shifted("open", 1) < outer_high) & (geometry.shifted("close", 1) < outer_high)
    second = geometry.bull if up else geometry.bear
    third = geometry.bull if up else geometry.bear
    clears = (geometry.close > geometry.shifted("open", 2) if up
              else geometry.close < geometry.shifted("open", 2))
    return geometry.shifted("long_body", 2) & second_inside & second & third & clears


def _three_outside(geometry: CandleGeometry, up: bool) -> pd.Series:
    """Three Outside Up / Down: reversal bar engulfed by an outside second bar."""
    first = geometry.shifted("bear", 2) if up else geometry.shifted("bull", 2)
    engulf = geometry.bull_engulf(1, 2) if up else geometry.bear_engulf(1, 2)
    third = geometry.bull if up else geometry.bear
    extends = (geometry.close > geometry.shifted("close", 1) if up
               else geometry.close < geometry.shifted("close", 1))
    return first & engulf & third & extends


def _three_method(geometry: CandleGeometry, up: bool) -> pd.Series:
    """Rising Three Methods / Mat Hold: long bar, three counter bars, long bar.

    The counter-trend bars stay inside the first bar's high/low range and the
    final long bar closes beyond the first bar's close.
    """
    inside = pd.Series(True, index=geometry.spread.index)
    counter = pd.Series(True, index=geometry.spread.index)
    for offset in (3, 2, 1):
        inside &= geometry.shifted("low", offset) >= geometry.shifted("low", 4)
        inside &= geometry.shifted("high", offset) <= geometry.shifted("high", 4)
        inside &= geometry.shifted("small_body", offset)
        counter &= geometry.shifted("bear", offset) if up else geometry.shifted("bull", offset)
    final = geometry.long_body & (geometry.bull if up else geometry.bear)
    extends = (geometry.close > geometry.shifted("close", 4) if up
               else geometry.close < geometry.shifted("close", 4))
    return geometry.shifted("long_body", 4) & inside & counter & final & extends


def _three_line_strike(geometry: CandleGeometry, up: bool) -> pd.Series:
    """Four-bar strike, implemented exactly as the reference tables print it.

    The reference labels these opposite to their polarity; see
    :data:`KNOWN_NAME_FORMULA_MISMATCH`.
    """
    run = (geometry.shifted("bull", 3) & geometry.shifted("bull", 2) & geometry.shifted("bull", 1) if up
           else geometry.shifted("bear", 3) & geometry.shifted("bear", 2) & geometry.shifted("bear", 1))
    reversal = geometry.bear if up else geometry.bull
    opens_beyond = (geometry.open > geometry.shifted("close", 1) if up
                    else geometry.open < geometry.shifted("close", 1))
    closes_beyond = (geometry.close < geometry.shifted("open", 3) if up
                     else geometry.close > geometry.shifted("open", 3))
    return run & reversal & geometry.long_body & opens_beyond & closes_beyond


def _ladder_bottom(geometry: CandleGeometry) -> pd.Series:
    """Three long bears, a small pause, then a long bull that clears the pause."""
    three_bears = (geometry.shifted("long_body", 4) & geometry.shifted("long_body", 3)
                   & geometry.shifted("long_body", 2) & geometry.shifted("bear", 4)
                   & geometry.shifted("bear", 3) & geometry.shifted("bear", 2))
    return (three_bears & geometry.shifted("small_body", 1) & geometry.long_body & geometry.bull
            & (geometry.close > geometry.shifted("open", 1)))


def _concealing_baby_swallow(geometry: CandleGeometry) -> pd.Series:
    """Two long bears, a gap-down pause, then a bear that engulfs the pause."""
    prior = (geometry.shifted("long_body", 3) & geometry.shifted("long_body", 2)
             & geometry.shifted("bear", 3) & geometry.shifted("bear", 2))
    paused = geometry.shifted("small_body", 1) & (geometry.high.shift(1) < geometry.low.shift(2))
    return prior & paused & geometry.bear_engulf(0, 1) & geometry.bear


def _near_high(geometry: CandleGeometry) -> pd.Series:
    ratio = geometry.config.small_shadow_ratio
    return geometry.close >= geometry.high - ratio * geometry.spread


def _near_low(geometry: CandleGeometry) -> pd.Series:
    ratio = geometry.config.small_shadow_ratio
    return geometry.close <= geometry.low + ratio * geometry.spread


# name -> (polarity, builder).  Every builder reads only bars at or before its
# own index, so the registry as a whole is causal.
PATTERN_BUILDERS: dict[str, tuple[str, Callable[[CandleGeometry], pd.Series]]] = {
    # --- Bullish -------------------------------------------------------------
    "bullish_engulfing": ("bull", lambda g: g.downtrend & g.shifted("bear", 1) & g.bull
                          & (g.open <= g.shifted("close", 1)) & (g.close >= g.shifted("open", 1))),
    "hammer": ("bull", lambda g: g.downtrend & (g.lower >= 2 * g.body) & (g.upper <= g.body) & g.small_body),
    "morning_star": ("bull", lambda g: g.shifted("bear", 2) & g.shifted("long_body", 2) & g.shifted("small_body", 1)
                     & g.bull & g.long_body & (g.close > g.shifted("mid", 2))),
    "piercing_line": ("bull", lambda g: g.downtrend & g.shifted("bear", 1) & g.shifted("long_body", 1) & g.bull
                      & (g.open < g.shifted("close", 1)) & (g.close > g.shifted("mid", 1))
                      & (g.close < g.shifted("open", 1))),
    "bullish_harami": ("bull", lambda g: g.downtrend & g.shifted("bear", 1) & g.shifted("long_body", 1) & g.bull
                       & g.small_body & g.inside_body(0, 1)),
    "three_white_soldiers": ("bull", lambda g: _three_soldiers(g, up=True)),
    "inverted_hammer": ("bull", lambda g: g.downtrend & (g.upper >= 2 * g.body) & (g.lower <= g.body) & g.small_body),
    "dragonfly_doji_bullish": ("bull", lambda g: g.downtrend & g.doji & (g.lower >= 2 * g.body) & (g.upper <= g.body)),
    "bullish_abandoned_baby": ("bull", lambda g: g.shifted("bear", 2) & g.shifted("long_body", 2) & g.shifted("doji", 1)
                               & g.gap_down.shift(1) & g.gap_up & g.bull & g.long_body
                               & (g.close > g.shifted("mid", 2))),
    "three_inside_up": ("bull", lambda g: _three_inside_body(g, up=True)),
    "three_outside_up": ("bull", lambda g: _three_outside(g, up=True)),
    "bullish_kicker": ("bull", lambda g: g.shifted("bear", 1) & g.shifted("long_body", 1) & g.bull & g.long_body
                       & (g.open > g.shifted("close", 1))),
    "tweezer_bottom": ("bull", lambda g: g.downtrend & g.shifted("bear", 1) & g.bull & g.equal(g.low, g.low.shift(1))),
    "rising_three_methods": ("bull", lambda g: _three_method(g, up=True)),
    "concealing_baby_swallow": ("bull", lambda g: _concealing_baby_swallow(g)),
    "mat_hold": ("bull", lambda g: _three_method(g, up=True)),
    "bullish_separating_lines": ("bull", lambda g: g.shifted("bear", 1) & g.bull
                                 & g.equal(g.open, g.shifted("open", 1)) & (g.close > g.shifted("close", 1))),
    "bullish_belt_hold": ("bull", lambda g: g.downtrend & g.bull & g.long_body & _near_high(g)
                          & ((g.lower <= g.config.small_shadow_ratio * g.spread)
                             | (g.open <= g.low + g.config.small_shadow_ratio * g.spread))),
    "three_line_strike_bull": ("bull", lambda g: _three_line_strike(g, up=True)),
    "ladder_bottom": ("bull", lambda g: _ladder_bottom(g)),
    "meeting_lines": ("bull", lambda g: g.shifted("bear", 1) & g.shifted("long_body", 1) & g.bull & g.long_body
                      & (g.open < g.shifted("close", 1)) & g.equal(g.close, g.shifted("close", 1))),
    # --- Bearish -------------------------------------------------------------
    "bearish_engulfing": ("bear", lambda g: g.uptrend & g.shifted("bull", 1) & g.bear
                          & (g.open >= g.shifted("close", 1)) & (g.close <= g.shifted("open", 1))),
    "bearish_belt_hold": ("bear", lambda g: g.uptrend & g.bear & g.long_body & _near_low(g)
                          & ((g.upper <= g.config.small_shadow_ratio * g.spread)
                             | (g.open >= g.high - g.config.small_shadow_ratio * g.spread))),
    "three_black_crows": ("bear", lambda g: _three_soldiers(g, up=False)),
    "three_line_strike_bear": ("bear", lambda g: _three_line_strike(g, up=False)),
    "hanging_man": ("bear", lambda g: g.uptrend & (g.lower >= 2 * g.body) & (g.upper <= g.body) & g.small_body),
    "upside_gap_two_crows": ("bear", lambda g: g.shifted("bull", 2) & g.shifted("long_body", 2)
                            & g.shifted("small_body", 1) & g.small_body & g.shifted("bear", 1) & g.bear
                            & (g.shifted("open", 1) > g.shifted("close", 2)) & (g.open > g.shifted("open", 1))
                            & (g.close < g.shifted("close", 1))),
    "bearish_evening_star": ("bear", lambda g: g.shifted("bull", 2) & g.shifted("long_body", 2)
                             & g.shifted("small_body", 1) & g.bear & g.long_body & (g.close < g.shifted("mid", 2))),
    "shooting_star": ("bear", lambda g: g.uptrend & (g.upper >= 2 * g.body) & (g.lower <= g.body) & g.small_body),
    "bearish_harami": ("bear", lambda g: g.uptrend & g.shifted("bull", 1) & g.shifted("long_body", 1) & g.bear
                       & g.small_body & g.inside_body(0, 1)),
    "bearish_doji_star": ("bear", lambda g: g.shifted("bull", 2) & g.shifted("long_body", 2) & g.shifted("doji", 1)
                          & g.bear & ((g.close < g.shifted("mid", 2)) | (g.close < g.shifted("close", 2)))),
    "bearish_abandoned_baby": ("bear", lambda g: g.shifted("bull", 2) & g.shifted("long_body", 2)
                               & g.shifted("doji", 1) & g.gap_up.shift(1) & g.gap_down & g.bear
                               & g.long_body & (g.close < g.shifted("mid", 2))),
    "bearish_tweezer_top": ("bear", lambda g: g.uptrend & g.shifted("bull", 1) & g.bear
                            & g.equal(g.high, g.high.shift(1))),
    "bearish_kicker": ("bear", lambda g: g.shifted("bull", 1) & g.shifted("long_body", 1) & g.bear & g.long_body
                       & (g.open < g.shifted("open", 1)) & (g.close < g.shifted("close", 1))),
    "bearish_three_inside_down": ("bear", lambda g: _three_inside_body(g, up=False)),
    "bearish_three_outside_down": ("bear", lambda g: _three_outside(g, up=False)),
    "bearish_mat_hold": ("bear", lambda g: _three_method(g, up=False)),
    "dark_cloud_cover": ("bear", lambda g: g.uptrend & g.shifted("bull", 1) & g.shifted("long_body", 1) & g.bear
                         & (g.open > g.shifted("high", 1)) & (g.close < g.shifted("mid", 1))
                         & (g.close > g.shifted("open", 1))),
}

BULLISH_PATTERNS: tuple[str, ...] = tuple(n for n, (side, _) in PATTERN_BUILDERS.items() if side == "bull")
BEARISH_PATTERNS: tuple[str, ...] = tuple(n for n, (side, _) in PATTERN_BUILDERS.items() if side == "bear")
PATTERN_NAMES: tuple[str, ...] = tuple(PATTERN_BUILDERS)


def pattern_occurrences(frame: pd.DataFrame, config: PatternConfig | None = None) -> pd.DataFrame:
    """Boolean pattern hits per bar; NaN marks a bar whose shape is unknown."""
    dates = pd.DatetimeIndex(pd.to_datetime(frame.Date))
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Candlestick patterns require unique increasing session dates.")
    geometry = CandleGeometry(frame, config)
    result = {}
    for name, (_, builder) in PATTERN_BUILDERS.items():
        # A shape test on an unusable bar is unknown, not absent.
        result[f"candle_{name}"] = builder(geometry).where(geometry.valid).astype(float)
    return pd.DataFrame(result, index=frame.index)


def window_tagged_features(frame: pd.DataFrame, windows: tuple[int, ...],
                          config: PatternConfig | None = None) -> pd.DataFrame:
    """Pattern hits plus rolling aggregates, tagged by trailing window.

    A column ending ``@w`` aggregates the last ``w`` observed sessions ending at
    that bar.  Windows are counted in sessions rather than calendar days so the
    grid in ``window_dataset`` keeps its meaning across holidays and weekends.
    Recency is likewise in sessions.
    """
    if any(w < 2 for w in windows):
        raise ValueError("Candlestick aggregate windows must be at least two sessions.")
    if len(set(windows)) != len(windows):
        raise ValueError("Candlestick aggregate windows must be distinct.")
    occurrences = pattern_occurrences(frame, config)
    # Position index, not distance from the end of the frame.  Recency must be
    # measured from the current bar so that rebuilding on a truncated prefix
    # reproduces the same value; a distance-from-end counter would not.
    positions = pd.Series(np.arange(len(occurrences), dtype=float), index=occurrences.index)
    parts = [occurrences]
    for window in windows:
        counts, recency = {}, {}
        for name in PATTERN_NAMES:
            column = occurrences[f"candle_{name}"]
            # An unknown bar must not be scored as a confirmed miss.
            counts[f"candle_{name}_count@{window}"] = column.fillna(0.0).rolling(window, min_periods=1).sum()
            # Sessions since the most recent confirmed hit, NaN when the pattern
            # has not fired within the window.
            last_hit = positions.where(column.eq(1)).ffill()
            age = positions - last_hit
            recency[f"candle_{name}_recency@{window}"] = age.where(age < window)
        bull_count = pd.concat(
            [counts[f"candle_{name}_count@{window}"] for name in BULLISH_PATTERNS], axis=1
        ).sum(axis=1, min_count=1)
        bear_count = pd.concat(
            [counts[f"candle_{name}_count@{window}"] for name in BEARISH_PATTERNS], axis=1
        ).sum(axis=1, min_count=1)
        summary = pd.DataFrame({
            f"candle_bull_count@{window}": bull_count,
            f"candle_bear_count@{window}": bear_count,
            f"candle_net_pressure@{window}": bull_count - bear_count,
            f"candle_bull_active@{window}": (bull_count > 0).astype(float),
            f"candle_bear_active@{window}": (bear_count > 0).astype(float),
        })
        parts.extend([pd.DataFrame(counts, index=frame.index),
                      pd.DataFrame(recency, index=frame.index),
                      summary])
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


def add_candlestick_features(frame: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 15, 21),
                             config: PatternConfig | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Attach window-tagged candle columns; return the frame and their names.

    Prefix invariance is part of the contract: running this on a truncated
    prefix must reproduce the same values on the rows both frames share.
    """
    tagged = window_tagged_features(frame, windows, config)
    forbidden = [c for c in tagged if c.startswith(("target", "news_")) or "future_return" in c]
    if forbidden:
        raise AssertionError(f"Candlestick features cannot carry target or news columns: {forbidden}")
    result = pd.concat([frame, tagged], axis=1)
    return result, list(tagged.columns)


def candles_for_polarity(columns: list[str], polarity: str) -> list[str]:
    """Candle columns belonging to one side, used to give each model its own bank."""
    if polarity not in {"bull", "bear"}:
        raise ValueError("Polarity must be bull or bear.")
    names = BULLISH_PATTERNS if polarity == "bull" else BEARISH_PATTERNS
    own = {f"candle_{name}" for name in names}
    return [c for c in columns if any(part in c for part in own)]