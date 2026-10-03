# Tradeability of the 28-day direction model

1,159 holdout origins (1,159 with a nonzero return). Frozen threshold 0.4. Abstention band degenerate (no trades withheld).

## Controls first

Buy-and-hold over the same 61 non-overlapping bets: net mean +1.008%/period, Sharpe +0.32, win rate 52.5%.

A long-biased rule earns the market's drift. It is only a forecast if it beats this line *and* keeps a positive market-neutral edge.

| rule | Sharpe | net mean | win rate | market-neutral Sharpe | excess when long | excess when short |
|---|---|---|---|---|---|---|
| buy-and-hold | +0.32 | +1.008% | 52.5% | 0.00 (by construction) | — | — |
| selected_direction | +0.56 | +1.722% | 60.7% | +0.37 | +0.733% | -2.707% |
| selected_return | -0.16 | -0.512% | 52.5% | -0.25 | -0.608% | +1.078% |
| class__basic__equal_ensemble | -0.11 | -0.356% | 55.7% | -0.25 | -0.539% | +1.287% |
| baseline_monthly_return | -0.26 | -0.800% | 52.5% | -0.34 | -0.831% | +1.473% |

## Cost and rule variants (2bps)

| model | rule | bets | net mean | Sharpe | max DD | win rate | exposure | break-even |
|---|---|---|---|---|---|---|---|---|
| selected_direction | always_in | 61 | +1.722% | 0.56 | -33.4% | 60.7% | 1.00 | 340.9 bps |
| selected_direction | with_abstention | 61 | +1.722% | 0.56 | -33.4% | 60.7% | 1.00 | 340.9 bps |
| selected_return | always_in | 61 | -0.512% | -0.16 | -56.2% | 52.5% | 1.00 | -64.4 bps |
| selected_return | with_abstention | 61 | -0.512% | -0.16 | -56.2% | 52.5% | 1.00 | -64.4 bps |
| class__basic__equal_ensemble | always_in | 61 | -0.356% | -0.11 | -55.3% | 55.7% | 1.00 | -72.9 bps |
| class__basic__equal_ensemble | with_abstention | 61 | -0.356% | -0.11 | -55.3% | 55.7% | 1.00 | -72.9 bps |
| baseline_monthly_return | always_in | 61 | -0.800% | -0.26 | -66.7% | 52.5% | 1.00 | -83.6 bps |
| baseline_monthly_return | with_abstention | 61 | -0.800% | -0.26 | -66.7% | 52.5% | 1.00 | -83.6 bps |

Accuracy of the frozen classifier: selected_direction 55.31%, class__basic__equal_ensemble 53.06%.

## Cost sensitivity: net mean return per period

| model | rule | 0bps | 1bps | 2bps | 5bps | 10bps | 20bps |
|---|---|---|---|---|---|---|---|
| selected_direction | always_in | +1.733% | +1.727% | +1.722% | +1.707% | +1.682% | +1.631% |
| selected_direction | with_abstention | +1.733% | +1.727% | +1.722% | +1.707% | +1.682% | +1.631% |
| selected_return | always_in | -0.496% | -0.504% | -0.512% | -0.535% | -0.573% | -0.650% |
| selected_return | with_abstention | -0.496% | -0.504% | -0.512% | -0.535% | -0.573% | -0.650% |
| class__basic__equal_ensemble | always_in | -0.346% | -0.351% | -0.356% | -0.370% | -0.394% | -0.442% |
| class__basic__equal_ensemble | with_abstention | -0.346% | -0.351% | -0.356% | -0.370% | -0.394% | -0.442% |
| baseline_monthly_return | always_in | -0.781% | -0.791% | -0.800% | -0.828% | -0.875% | -0.968% |
| baseline_monthly_return | with_abstention | -0.781% | -0.791% | -0.800% | -0.828% | -0.875% | -0.968% |
