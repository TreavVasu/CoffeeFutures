"""Dataset expansion over (lookback, forward) origin pairs.

The request was to expand the dataset so each origin contributes several
labelled rows: "15 days of data predicting the 18th day, 15 days to the 20th
day, and similar".  That is a grid of cells

    cell = (lookback L, forward offset k)

where the feature row is the trailing window ending at origin ``t`` and the
label is the direction of ``Close[t+k] / Close[t] - 1`` measured in observed
sessions.  In the example ``L=15`` with ``k=3`` predicts the 18th session and
``k=5`` predicts the 20th.

Two rules make the expansion safe:

* **Sessions, not calendar days.**  Offsets index the observed trading
  calendar, so a holiday does not silently shorten a horizon.
* **One cell is one experiment.**  Rows are never pooled across different
  ``k`` inside a single fitted model.  Mixing horizons would let a 3-session
  label and a 20-session label share a decision boundary and would make the
  purge/embargo logic ambiguous.  Each cell is trained, selected and evaluated
  on its own, and the grid is summarised across cells afterwards.

Overlap is the cost of expansion.  Neighbouring origins inside a cell share
future observations, and neighbouring cells share origins, so the raw row count
overstates the independent evidence.  Every consumer must keep using the
dependence-aware block bootstrap rather than an independent-sample interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_LOOKBACKS: tuple[int, ...] = (5, 10, 15, 21)
DEFAULT_FORWARDS: tuple[int, ...] = (3, 5, 8, 10, 15, 18, 20)


@dataclass(frozen=True)
class Cell:
    """One (lookback, forward) origin pair from the expansion grid."""

    lookback: int
    forward: int

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("A lookback needs at least two sessions of history.")
        if self.forward < 1:
            raise ValueError("A forward offset needs at least one session.")

    @property
    def name(self) -> str:
        return f"L{self.lookback}_k{self.forward}"

    def __str__(self) -> str:
        return self.name


def build_grid(lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS,
               forwards: tuple[int, ...] = DEFAULT_FORWARDS) -> list[Cell]:
    """Cartesian product of the requested lookbacks and forward offsets."""
    if not lookbacks or not forwards:
        raise ValueError("The expansion grid needs at least one lookback and one forward offset.")
    if len(set(lookbacks)) != len(lookbacks) or len(set(forwards)) != len(forwards):
        raise ValueError("Lookbacks and forward offsets must each be distinct.")
    return [Cell(int(l), int(k)) for l in lookbacks for k in forwards]


def cell_frame(frame: pd.DataFrame, cell: Cell) -> pd.DataFrame:
    """Attach this cell's forward return, endpoint and direction labels.

    ``frame`` must be the observed-session frame with unique increasing ``Date``
    and a finite positive ``Close``.  Rows whose endpoint runs past the last
    observed session, or whose forward return is not finite, are kept with a
    ``NaN`` label rather than dropped, so the caller sees the true row count and
    applies one consistent label filter.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(frame.Date))
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Window expansion requires unique increasing session dates.")
    close = pd.to_numeric(frame.Close, errors="coerce").to_numpy(dtype=float)
    count = len(frame)
    endpoint = np.arange(count) + cell.forward
    valid = endpoint < count
    end_dates = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    end_close = np.full(count, np.nan)
    safe_endpoint = np.where(valid, endpoint, 0)
    end_dates.iloc[:] = dates.take(safe_endpoint)
    end_close[valid] = close[safe_endpoint[valid]]

    result = frame.copy()
    result["cell"] = cell.name
    result["lookback"] = cell.lookback
    result["forward"] = cell.forward
    result["target_end_date"] = end_dates
    result["target_close"] = end_close
    result["target_return"] = end_close / close - 1.0
    result["target_sessions"] = np.where(valid, cell.forward, np.nan)
    # The lookback window must actually exist behind the origin.
    result["lookback_start_date"] = dates.take(np.maximum(np.arange(count) - cell.lookback + 1, 0))
    result["lookback_complete"] = np.arange(count) >= cell.lookback - 1
    # Both sides are binary 0/1 on the SAME rows: 1 for this side's outcome,
    # 0 for the opposite one, NaN when the return is exactly zero or undefined.
    # They are therefore exact complements wherever both are observed, which is
    # what makes a fair head-to-head comparison between the two models possible.
    returns = result.target_return.to_numpy(dtype=float)
    observed = np.isfinite(returns) & (returns != 0)
    result["direction_up"] = np.where(observed, (returns > 0).astype(float), np.nan)
    result["direction_down"] = np.where(observed, (returns < 0).astype(float), np.nan)
    return result


def feature_columns_for(cell: Cell, candle_columns: list[str],
                        shared: list[str]) -> list[str]:
    """Columns visible to one cell: shared inputs plus this cell's candle window.

    The candle bank carries every trailing window, so a cell reads the
    aggregates tagged with its own ``@lookback`` and deliberately ignores the
    other windows in the bank.  ``candle_<pattern>`` occurrence columns are
    untagged because they describe the current bar rather than an aggregate.
    Shared inputs are non-candle predictors that are not horizon-specific.
    """
    tag = f"@{cell.lookback}"
    current_bar = [c for c in candle_columns if "@" not in c]
    this_window = [c for c in candle_columns if c.endswith(tag)]
    columns = list(dict.fromkeys(current_bar + this_window + list(shared)))
    forbidden = [c for c in columns if c.startswith(("target", "news_")) or "future_return" in c]
    if forbidden:
        raise AssertionError(f"Targets and retrospective news cannot enter a feature set: {forbidden}")
    return columns