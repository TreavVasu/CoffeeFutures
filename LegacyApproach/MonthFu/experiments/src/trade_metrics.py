"""Decision-level evaluation: does a forecast earn money after costs?

Classification accuracy answers "how often is the sign right?". A tradeable rule
needs a different question: what is the expected return per position, and does it
survive trading costs, turnover and drawdown? This module computes that, and
reports the break-even cost so a rule's viability can be judged against a cost
assumption rather than against a bare accuracy number.

Conventions match the existing project backtest:

* Positions are held on a **non-overlapping** schedule. Monthly targets overlap
  on consecutive daily origins, so charging every origin would multiply one
  economic bet and inflate turnover. ``nonoverlap_positions`` selects origins
  whose target window has closed before the next position opens.
* Turnover is ``abs(position - previous_position)``, so moving from short to long
  is 2 and pays two one-way costs.
* Costs are charged on turnover at a stated basis-point rate.
* Sharpe annualizes by ``252 / horizon`` periods per year.

A forecast with no edge must fail here even if its accuracy looks respectable,
because a near-chance sign rule with turnover loses to cash once costs apply.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252.0


def cost_curve(signal: np.ndarray, actual: np.ndarray, bps: float,
               periods_per_year: float) -> dict:
    """Performance of a fixed position path at one cost rate.

    ``signal`` must already be a non-overlapping position path in {-1, 0, 1}.
    Returns are simple returns on the forecast horizon, not daily marks, so
    volatility is scaled by periods per year rather than by sqrt(252).
    """
    signal = np.asarray(signal, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if signal.shape != actual.shape:
        raise ValueError("Signal and realized returns must align.")
    if not len(signal):
        raise ValueError("At least one period is required.")
    previous = np.r_[0.0, signal[:-1]]
    turnover = np.abs(signal - previous)
    cost = turnover * bps / 10_000.0
    net = signal * actual - cost
    gross = signal * actual
    standard_deviation = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    wealth = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    sharpe = (float(np.mean(net) / standard_deviation * np.sqrt(periods_per_year))
              if standard_deviation > 0 else float("nan"))
    annualized = (float(wealth[-1] ** (periods_per_year / len(net)) - 1.0)
                  if len(net) and wealth[-1] > 0 else float("nan"))
    return {
        "periods": int(len(net)),
        "cost_bps": float(bps),
        "gross_mean_return": float(np.mean(gross)),
        "net_mean_return": float(np.mean(net)),
        "total_cost_drag": float(np.mean(cost)),
        "average_turnover": float(np.mean(turnover)),
        "cumulative_return": float(wealth[-1] - 1.0),
        "annualized_return": annualized,
        "annualized_volatility": float(standard_deviation * np.sqrt(periods_per_year)),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((net > 0).mean()),
        "exposure": float(np.mean(np.abs(signal))),
    }


def breakeven_cost(signal: np.ndarray, actual: np.ndarray, periods_per_year: float,
                   tolerance: float = 1e-12) -> float:
    """Cost rate at which mean net return reaches zero, in basis points.

    Mean net return is linear in the cost rate, so one evaluation is enough:
    ``mean(signal * actual) - mean(turnover) * bps / 10000 = 0``. A negative
    result means the rule cannot break even at any non-negative cost, which is
    the decisive statement for tradeability.
    """
    signal = np.asarray(signal, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if signal.shape != actual.shape or not len(signal):
        raise ValueError("Signal and realized returns must align and be non-empty.")
    turnover = np.abs(signal - np.r_[0.0, signal[:-1]])
    edge = float(np.mean(signal * actual))
    traded = float(np.mean(turnover))
    if traded <= 0:
        return float("inf") if edge > 0 else float("nan")
    return edge / traded * 10_000.0


def market_neutral_edge(signal: np.ndarray, actual: np.ndarray,
                        periods_per_year: float) -> dict:
    """Separate a timing edge from a directional market bet.

    A rule that is simply long while the market rises shows positive Sharpe
    without forecasting anything. Subtracting the unconditional mean return
    leaves only the part attributable to the signal, so a rule whose neutralised
    Sharpe is near zero is a market bet, not a timing model.
    """
    signal = np.asarray(signal, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if signal.shape != actual.shape or not len(signal):
        raise ValueError("Signal and realized returns must align and be non-empty.")
    market = float(actual.mean())
    excess = actual - market
    long_mask, short_mask = signal > 0, signal < 0
    neutral = signal * excess
    standard_deviation = float(np.std(neutral, ddof=1)) if len(neutral) > 1 else 0.0
    return {
        "market_mean_return": market,
        "signal_correlation_with_return": float(np.corrcoef(signal, actual)[0, 1]),
        "signal_correlation_with_excess": float(np.corrcoef(signal, excess)[0, 1]),
        "excess_return_when_long": (float(excess[long_mask].mean())
                                    if long_mask.any() else float("nan")),
        "excess_return_when_short": (float(excess[short_mask].mean())
                                     if short_mask.any() else float("nan")),
        "long_minus_short_excess": (float(excess[long_mask].mean() - excess[short_mask].mean())
                                    if long_mask.any() and short_mask.any() else float("nan")),
        "neutralised_sharpe": (float(np.mean(neutral) / standard_deviation
                                      * np.sqrt(periods_per_year))
                               if standard_deviation > 0 else float("nan")),
        "position_up_share": float(signal.mean()),
    }


def build_signal(probability_up: np.ndarray, threshold: float,
                 abstain: tuple[float, float] | None = None) -> np.ndarray:
    """Map an up-probability to a position in {-1, 0, +1}.

    ``abstain`` is the frozen ``(lower, upper)`` band. Inside the band the rule
    holds no position, so a low-conviction forecast is not paid for as a trade.
    """
    probability_up = np.asarray(probability_up, dtype=float)
    signal = np.where(probability_up >= threshold, 1.0, -1.0)
    if abstain:
        lower, upper = abstain
        if upper > lower:
            signal = np.where((probability_up > lower) & (probability_up < upper),
                              0.0, signal)
    return signal


def nonoverlap_signal(frame: pd.DataFrame, probability_up: np.ndarray,
                      threshold: float, abstain: tuple[float, float] | None = None,
                      offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Positions and realized returns on a non-overlapping schedule.

    Returns ``(signal, actual)`` for the selected origins only, so cost and
    performance are measured per economic bet rather than per daily origin.
    """
    from direction_metrics import nonoverlap_positions
    positions = nonoverlap_positions(frame, offset)
    if not positions:
        raise ValueError("No non-overlapping origins are available.")
    rows = frame.iloc[positions]
    actual = rows.target_return.to_numpy(dtype=float)
    keep = np.isfinite(actual)
    signal = build_signal(np.asarray(probability_up, dtype=float)[positions][keep],
                          threshold, abstain)
    return signal, actual[keep]
