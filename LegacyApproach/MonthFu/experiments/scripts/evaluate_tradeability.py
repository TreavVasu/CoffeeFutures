"""Does the monthly direction model support a buy/sell decision?

Accuracy alone does not answer that. This script converts the frozen holdout
forecasts into an explicit trading rule and reports what the rule would have
earned after costs, how much cost it can absorb before breaking even, and
whether abstaining on low conviction changes the answer.

It reads only saved artifacts. Nothing is refitted and no threshold is chosen
here: the classifier threshold and abstention band were frozen from pre-2022 data
before the holdout was scored, so re-optimizing either on these results would
invalidate the comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MonthFu/experiments/src"))

from direction_metrics import binary_labels_from_returns  # noqa: E402
from trade_metrics import (breakeven_cost, cost_curve,  # noqa: E402
                           market_neutral_edge, nonoverlap_signal)

COST_GRID = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]


def evaluate(folder: Path, horizon: int) -> dict:
    selection = json.loads((folder / "selection.json").read_text())
    holdout = pd.read_csv(folder / "holdout_predictions.csv",
                          parse_dates=["Date", "target_end_date"])
    holdout = holdout.sort_values("Date").reset_index(drop=True)
    threshold = selection["recipes"]["selected_direction"]["threshold"]
    band = selection["selected_abstention"]
    abstain = (float(band["lower"]), float(band["upper"])) if band["upper"] > band["lower"] else None
    periods_per_year = 252.0 / horizon

    labels = binary_labels_from_returns(holdout.target_return)
    keep = np.isfinite(labels)
    record: dict = {
        "horizon_days": horizon, "threshold": threshold,
        "abstention_band": list(abstain) if abstain else None,
        "holdout_rows": int(len(holdout)), "binary_rows": int(keep.sum()),
        "method": ("Non-overlapping positions, costs charged on turnover, "
                   "Sharpe annualized by 252/horizon. Thresholds frozen pre-holdout."),
    }

    models = ["selected_direction", "selected_return", "class__basic__equal_ensemble",
              "baseline_monthly_return", "always_up"]
    record["models"] = {}
    for name in models:
        if name not in holdout.columns:
            continue
        raw = holdout[name].to_numpy(dtype=float)
        is_probability = name.startswith(("selected_direction", "class__"))
        probability = np.clip(raw, 0, 1) if is_probability else None
        entry: dict = {}
        if is_probability:
            up = probability >= threshold
            entry["accuracy"] = float(np.mean(up[keep] == labels[keep].astype(int)))
            entry["predicted_up_share"] = float(np.mean(up))
        source = probability if is_probability else raw
        rule_threshold = threshold if is_probability else 0.0

        for label, band_used in (("always_in", None), ("with_abstention", abstain)):
            signal, actual = nonoverlap_signal(holdout, source, rule_threshold,
                                               band_used, offset=0)
            curve = {str(int(b)): cost_curve(signal, actual, b, periods_per_year)
                     for b in COST_GRID}
            edge = breakeven_cost(signal, actual, periods_per_year)
            entry[label] = {"net_at_2bps": curve["2"], "cost_curve": curve,
                            "breakeven_cost_bps": (float(edge) if np.isfinite(edge) else None),
                            "neutralised": market_neutral_edge(signal, actual, periods_per_year)}
        record["models"][name] = entry

    # Controls. Without these, a rule that is simply long in a rising market
    # looks like a forecast. Buy-and-hold is the honest comparator.
    _, market_actual = nonoverlap_signal(
        holdout, holdout.baseline_monthly_return.to_numpy(float), 0.0, None, offset=0)
    ones = np.ones_like(market_actual)
    record["controls"] = {
        "bets": int(len(market_actual)),
        "buy_and_hold": cost_curve(ones, market_actual, 2.0, periods_per_year),
        "always_short": cost_curve(-ones, market_actual, 2.0, periods_per_year),
        "note": ("Buy-and-hold is the relevant comparator; a long-biased rule "
                 "cannot be credited for the market's own drift."),
    }
    return record


def write_markdown(record: dict, path: Path) -> None:
    horizon = record["horizon_days"]
    control = record.get("controls", {})
    hold = control.get("buy_and_hold", {})
    lines = [
        f"# Tradeability of the {horizon}-day direction model", "",
        f"{record['holdout_rows']:,} holdout origins "
        f"({record['binary_rows']:,} with a nonzero return). "
        f"Frozen threshold {record['threshold']}. "
        f"Abstention band {record['abstention_band'] or 'degenerate (no trades withheld)'}.", "",
        "## Controls first", "",
        f"Buy-and-hold over the same {control.get('bets', 0)} non-overlapping bets: "
        f"net mean {100*hold.get('net_mean_return', 0):+.3f}%/period, "
        f"Sharpe {hold.get('sharpe', float('nan')):+.2f}, "
        f"win rate {100*hold.get('win_rate', 0):.1f}%.",
        "",
        "A long-biased rule earns the market's drift. It is only a forecast if it "
        "beats this line *and* keeps a positive market-neutral edge.", "",
        "| rule | Sharpe | net mean | win rate | market-neutral Sharpe | excess when long | excess when short |",
        "|---|---|---|---|---|---|---|",
        f"| buy-and-hold | {hold.get('sharpe', float('nan')):+.2f} | "
        f"{100*hold.get('net_mean_return', 0):+.3f}% | "
        f"{100*hold.get('win_rate', 0):.1f}% | 0.00 (by construction) | — | — |",
    ]
    for name, entry in record["models"].items():
        block = entry.get("always_in")
        if not block:
            continue
        net, neutral = block["net_at_2bps"], block.get("neutralised", {})
        lines.append(
            f"| {name} | {net['sharpe']:+.2f} | {100*net['net_mean_return']:+.3f}% | "
            f"{100*net['win_rate']:.1f}% | {neutral.get('neutralised_sharpe', float('nan')):+.2f} | "
            f"{100*neutral.get('excess_return_when_long', float('nan')):+.3f}% | "
            f"{100*neutral.get('excess_return_when_short', float('nan')):+.3f}% |")

    lines += ["", "## Cost and rule variants (2bps)", "",
              "| model | rule | bets | net mean | Sharpe | max DD | win rate | "
              "exposure | break-even |", "|---|---|---|---|---|---|---|---|---|"]
    for name, entry in record["models"].items():
        accuracy = f"{100*entry['accuracy']:.2f}%" if "accuracy" in entry else "—"
        for label, block in entry.items():
            if label in {"accuracy", "predicted_up_share"}:
                continue
            net = block["net_at_2bps"]
            edge = block["breakeven_cost_bps"]
            edge_text = (f"{edge:.1f} bps" if isinstance(edge, float)
                         else "cannot break even")
            lines.append(
                f"| {name} | {label} | {net['periods']} | "
                f"{100*net['net_mean_return']:+.3f}% | {net['sharpe']:.2f} | "
                f"{100*net['max_drawdown']:.1f}% | {100*net['win_rate']:.1f}% | "
                f"{net['exposure']:.2f} | {edge_text} |")
    lines += ["", f"Accuracy of the frozen classifier: " +
              ", ".join(f"{n} {100*e['accuracy']:.2f}%"
                        for n, e in record["models"].items() if "accuracy" in e) + ".",
              "", "## Cost sensitivity: net mean return per period", "",
              "| model | rule | " + " | ".join(f"{b:g}bps" for b in COST_GRID) + " |",
              "|---|---|" + "---|" * len(COST_GRID)]
    for name, entry in record["models"].items():
        for label, block in entry.items():
            if label in {"accuracy", "predicted_up_share"}:
                continue
            curve = block["cost_curve"]
            cells = " | ".join(f"{100*curve[str(int(b))]['net_mean_return']:+.3f}%"
                               for b in COST_GRID)
            lines.append(f"| {name} | {label} | {cells} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folders", nargs="+", type=Path, required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[30, 28])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for folder, horizon in zip(args.folders, args.horizons):
        record = evaluate(folder, horizon)
        records.append(record)
        write_markdown(record, folder / "tradeability.md")
        print(f"{horizon}-day: wrote {folder/'tradeability.md'}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, default=str) + "\n")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

