"""Cost-aware evaluation of a trading rule, independent of the forecast itself."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trade_metrics import (breakeven_cost, build_signal, cost_curve,  # noqa: E402
                           market_neutral_edge, nonoverlap_signal)


class TradeMetricsTests(unittest.TestCase):
    def test_zero_cost_leaves_gross_return_and_costs_reduce_it(self):
        signal = np.array([1.0, 1.0, -1.0, -1.0])
        actual = np.array([0.01, 0.02, -0.01, -0.03])
        free = cost_curve(signal, actual, 0.0, periods_per_year=12)
        paid = cost_curve(signal, actual, 10.0, periods_per_year=12)
        self.assertEqual(free["periods"], 4)
        self.assertAlmostEqual(free["net_mean_return"], free["gross_mean_return"])
        self.assertLess(paid["net_mean_return"], free["net_mean_return"])
        self.assertGreater(paid["total_cost_drag"], 0)

    def test_turnover_counts_a_reversal_as_two_units(self):
        signal = np.array([1.0, -1.0])
        actual = np.array([0.0, 0.0])
        curve = cost_curve(signal, actual, 10.0, periods_per_year=12)
        # One unit to open long, then two to flip to short.
        self.assertAlmostEqual(curve["average_turnover"], 1.5)

    def test_always_in_market_has_no_edge_after_costs(self):
        signal = np.ones(200)
        actual = np.zeros(200)
        self.assertLess(cost_curve(signal, actual, 2.0, 12)["net_mean_return"], 0)

    def test_breakeven_cost_is_the_rate_that_zeroes_mean_return(self):
        signal = np.array([1.0, 1.0, 1.0, -1.0])
        actual = np.array([0.02, 0.01, -0.01, -0.02])
        rate = breakeven_cost(signal, actual, periods_per_year=12)
        at_rate = cost_curve(signal, actual, rate, 12)
        self.assertAlmostEqual(at_rate["net_mean_return"], 0.0, places=12)

    def test_unprofitable_rule_reports_negative_breakeven(self):
        signal = np.ones(50)
        actual = np.full(50, -0.01)
        self.assertLess(breakeven_cost(signal, actual, 12), 0.0)

    def test_market_neutral_edge_separates_drift_from_forecast(self):
        actual = np.array([0.05, 0.04, 0.06, 0.03])
        # Constant long in a rising market: no forecast content. The spread needs
        # both sides to exist, so an all-long signal yields NaN rather than a
        # fabricated zero, and the signal's own variance is zero.
        drift = market_neutral_edge(np.ones(4), actual, 12)
        self.assertTrue(np.isnan(drift["long_minus_short_excess"]))
        self.assertAlmostEqual(drift["excess_return_when_long"], 0.0, places=12)
        # A signal that is long more often but takes both sides isolates drift.
        biased = market_neutral_edge(np.array([1.0, 1.0, 1.0, -1.0]), actual, 12)
        self.assertAlmostEqual(biased["long_minus_short_excess"],
                               biased["excess_return_when_long"]
                               - biased["excess_return_when_short"], places=12)
        # A genuinely alternating signal does carry excess content.
        mixed = market_neutral_edge(np.array([1.0, -1.0, 1.0, -1.0]), actual, 12)
        self.assertGreater(mixed["long_minus_short_excess"], 0)
        self.assertGreater(mixed["signal_correlation_with_excess"], 0)

    def test_abstention_withholds_a_position_only_inside_the_band(self):
        probability = np.array([0.10, 0.45, 0.80])
        plain = build_signal(probability, 0.4)
        self.assertTrue(np.all(plain != 0))
        gated = build_signal(probability, 0.4, abstain=(0.4, 0.6))
        self.assertEqual(gated[1], 0.0)
        self.assertEqual(gated[0], -1.0)
        self.assertEqual(gated[2], 1.0)

    def test_nonoverlap_signal_returns_one_row_per_economic_bet(self):
        frame = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01", "2024-01-04", "2024-01-11"]),
            "target_end_date": pd.to_datetime(["2024-01-11", "2024-01-11", "2024-02-01"]),
            "target_return": [0.01, 0.02, 0.03],
        })
        signal, actual = nonoverlap_signal(frame, np.array([0.9, 0.8, 0.7]), 0.5)
        self.assertEqual(len(signal), len(actual))
        self.assertEqual(len(signal), 2)
        np.testing.assert_allclose(actual, [0.01, 0.03])

    def test_mismatched_or_empty_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            cost_curve(np.ones(3), np.ones(4), 2.0, 12)
        with self.assertRaises(ValueError):
            cost_curve(np.array([]), np.array([]), 2.0, 12)


if __name__ == "__main__":
    unittest.main()
