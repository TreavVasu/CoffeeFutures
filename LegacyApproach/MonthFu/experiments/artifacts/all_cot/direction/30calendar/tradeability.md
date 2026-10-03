# Tradeability of the 30-day direction model

1,157 holdout origins (1,156 with a nonzero return). Frozen threshold 0.4. Abstention band degenerate (no trades withheld).

## Controls first

Buy-and-hold over the same 56 non-overlapping bets: net mean +1.383%/period, Sharpe +0.32, win rate 51.8%.

A long-biased rule earns the market's drift. It is only a forecast if it beats this line *and* keeps a positive market-neutral edge.

| rule | Sharpe | net mean | win rate | market-neutral Sharpe | excess when long | excess when short |
|---|---|---|---|---|---|---|
| buy-and-hold | +0.32 | +1.383% | 51.8% | 0.00 (by construction) | — | — |
| selected_direction | +0.53 | +2.263% | 55.4% | +0.34 | +0.945% | -3.464% |
| selected_return | +0.55 | +2.357% | 55.4% | +0.41 | +1.246% | -3.114% |
| class__basic__equal_ensemble | +0.65 | +2.785% | 53.6% | +0.55 | +1.784% | -3.474% |
| baseline_monthly_return | +0.54 | +2.329% | 55.4% | +0.43 | +1.363% | -2.876% |

## Cost and rule variants (2bps)

| model | rule | bets | net mean | Sharpe | max DD | win rate | exposure | break-even |
|---|---|---|---|---|---|---|---|---|
| selected_direction | always_in | 56 | +2.263% | 0.53 | -67.3% | 55.4% | 1.00 | 386.1 bps |
| selected_direction | with_abstention | 56 | +2.263% | 0.53 | -67.3% | 55.4% | 1.00 | 386.1 bps |
| selected_return | always_in | 56 | +2.357% | 0.55 | -40.0% | 55.4% | 1.00 | 309.0 bps |
| selected_return | with_abstention | 56 | +2.357% | 0.55 | -40.0% | 55.4% | 1.00 | 309.0 bps |
| class__basic__equal_ensemble | always_in | 56 | +2.785% | 0.65 | -40.4% | 53.6% | 1.00 | 333.9 bps |
| class__basic__equal_ensemble | with_abstention | 56 | +2.785% | 0.65 | -40.4% | 53.6% | 1.00 | 333.9 bps |
| baseline_monthly_return | always_in | 56 | +2.329% | 0.54 | -40.4% | 55.4% | 1.00 | 320.0 bps |
| baseline_monthly_return | with_abstention | 56 | +2.329% | 0.54 | -40.4% | 55.4% | 1.00 | 320.0 bps |

Accuracy of the frozen classifier: selected_direction 57.61%, class__basic__equal_ensemble 54.15%.

## Cost sensitivity: net mean return per period

| model | rule | 0bps | 1bps | 2bps | 5bps | 10bps | 20bps |
|---|---|---|---|---|---|---|---|
| selected_direction | always_in | +2.275% | +2.269% | +2.263% | +2.246% | +2.216% | +2.157% |
| selected_direction | with_abstention | +2.275% | +2.269% | +2.263% | +2.246% | +2.216% | +2.157% |
| selected_return | always_in | +2.372% | +2.365% | +2.357% | +2.334% | +2.296% | +2.219% |
| selected_return | with_abstention | +2.372% | +2.365% | +2.357% | +2.334% | +2.296% | +2.219% |
| class__basic__equal_ensemble | always_in | +2.802% | +2.794% | +2.785% | +2.760% | +2.718% | +2.634% |
| class__basic__equal_ensemble | with_abstention | +2.802% | +2.794% | +2.785% | +2.760% | +2.718% | +2.634% |
| baseline_monthly_return | always_in | +2.343% | +2.336% | +2.329% | +2.307% | +2.270% | +2.197% |
| baseline_monthly_return | with_abstention | +2.343% | +2.336% | +2.329% | +2.307% | +2.270% | +2.197% |
