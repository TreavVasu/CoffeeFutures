# 30-calendar-day experiment

The direction recipe chosen before the 2022 holdout worsened holdout macro F1 by -0.62 percentage points versus the saved monthly model. Its accuracy is 53.03% versus 55.02%. These are exploratory reused-holdout results.

Binary metrics use 1,156 common nonzero-return origins; return metrics retain 1,157 origins. Up and down scores receive equal weight in macro F1. Pre-2022 selection used 2,016 expanding OOF origins and 72 classifier/regressor configurations.

## Selected and simple model comparisons

| model | accuracy | macro_f1 | balanced_accuracy | up_precision | up_recall | down_precision | down_recall |
|---|---|---|---|---|---|---|---|
| class__basic__equal_ensemble | 55.80% | 55.22% | 55.46% | 56.40% | 64.77% | 54.91% | 46.14% |
| reg__basic__equal_ensemble | 55.45% | 54.48% | 54.99% | 55.79% | 67.61% | 54.88% | 42.37% |
| selected_direction | 53.03% | 53.01% | 53.03% | 54.84% | 52.92% | 51.21% | 53.14% |
| selected_return | 55.10% | 53.70% | 54.54% | 55.28% | 69.95% | 54.77% | 39.14% |
| baseline_monthly_return | 55.02% | 53.63% | 54.46% | 55.22% | 69.78% | 54.64% | 39.14% |
| baseline_paper_return | 50.78% | 50.78% | 50.85% | 52.70% | 48.91% | 49.00% | 52.78% |
| baseline_majority | 48.18% | 32.52% | 50.00% | 0.00% | 0.00% | 48.18% | 100.00% |

### Return forecasts

| model | rmse | mae | direction_accuracy | r2_vs_zero |
|---|---|---|---|---|
| reg__basic__equal_ensemble | 11.21% | 8.25% | 55.40% | 0.0125 |
| selected_return | 11.19% | 8.26% | 55.06% | 0.0154 |
| baseline_monthly_return | 11.11% | 8.19% | 54.97% | 0.0301 |
| baseline_paper_return | 11.70% | 8.75% | 50.73% | -0.0756 |
| baseline_mean_return | 11.24% | 8.35% | 51.77% | 0.0078 |
| baseline_zero_return | 11.28% | 8.34% | 0.09% | 0.0000 |

RMSE/MAE percentages are arithmetic-return percentage points. Original regression direction accuracy treats predicted zero as neutral; binary tables treat it as an Up tie.

### Frozen recipes

Direction: `{"group": "basic", "name": "class__basic__xgb", "task": "class", "threshold": 0.4750000000000001, "weights": {"class__basic__xgb_1": 1.0}}`

Return: `{"group": "basic", "name": "reg__basic__xgb", "task": "reg", "threshold": 0.5, "weights": {"reg__basic__xgb_0": 1.0}}`

Each simple ensemble averages one CV-chosen linear model, one XGBoost model and one random forest with weights of 1/3. Its classifier threshold is also fixed from pre-2022 OOF. There is no learned stacker or holdout weight tuning.

## Feature and model comparisons

| model | accuracy | macro_f1 | balanced_accuracy | up_precision | up_recall | down_precision | down_recall |
|---|---|---|---|---|---|---|---|
| class__basic__linear | 51.12% | 49.54% | 51.84% | 54.83% | 32.22% | 49.50% | 71.45% |
| class__basic__xgb | 53.03% | 53.01% | 53.03% | 54.84% | 52.92% | 51.21% | 53.14% |
| class__basic__forest | 55.45% | 55.23% | 55.27% | 56.58% | 60.27% | 54.05% | 50.27% |
| class__basic__equal_ensemble | 55.80% | 55.22% | 55.46% | 56.40% | 64.77% | 54.91% | 46.14% |
| class__direction_core__linear | 51.21% | 51.18% | 51.19% | 52.99% | 51.75% | 49.39% | 50.63% |
| class__direction_core__xgb | 53.89% | 53.89% | 54.00% | 56.04% | 51.09% | 51.97% | 56.91% |
| class__direction_core__forest | 50.69% | 50.54% | 50.56% | 52.33% | 54.26% | 48.79% | 46.86% |
| class__direction_core__equal_ensemble | 52.51% | 51.06% | 53.20% | 56.98% | 34.06% | 50.50% | 72.35% |
| class__price_direction__linear | 50.78% | 50.74% | 50.95% | 52.86% | 46.24% | 49.05% | 55.66% |
| class__price_direction__xgb | 51.99% | 51.44% | 52.45% | 55.07% | 39.90% | 50.14% | 64.99% |
| class__price_direction__forest | 51.56% | 50.61% | 52.13% | 54.91% | 36.39% | 49.80% | 67.86% |
| class__price_direction__equal_ensemble | 51.30% | 50.45% | 51.84% | 54.43% | 36.89% | 49.60% | 66.79% |
| class__cot_direction__linear | 53.11% | 53.11% | 53.22% | 55.23% | 50.25% | 51.23% | 56.19% |
| class__cot_direction__xgb | 51.82% | 51.79% | 51.98% | 53.98% | 47.58% | 50.00% | 56.37% |
| class__cot_direction__forest | 48.53% | 48.35% | 48.81% | 50.41% | 41.07% | 47.16% | 56.55% |
| class__cot_direction__equal_ensemble | 49.65% | 49.58% | 49.58% | 51.41% | 51.59% | 47.75% | 47.58% |
| class__weather_direction__linear | 50.43% | 50.15% | 50.77% | 52.77% | 41.40% | 48.83% | 60.14% |
| class__weather_direction__xgb | 55.45% | 53.76% | 54.83% | 55.40% | 71.95% | 55.56% | 37.70% |
| class__weather_direction__forest | 54.24% | 53.70% | 53.92% | 55.13% | 62.77% | 52.95% | 45.06% |
| class__weather_direction__equal_ensemble | 53.29% | 53.28% | 53.31% | 55.17% | 52.59% | 51.45% | 54.04% |
| class__engineered_direction__linear | 54.33% | 54.21% | 54.21% | 55.77% | 57.26% | 52.68% | 51.17% |
| class__engineered_direction__xgb | 55.02% | 54.68% | 54.78% | 56.01% | 61.44% | 53.71% | 48.11% |
| class__engineered_direction__forest | 51.56% | 50.21% | 52.22% | 55.31% | 33.89% | 49.81% | 70.56% |
| class__engineered_direction__equal_ensemble | 55.10% | 54.81% | 54.88% | 56.15% | 60.93% | 53.75% | 48.83% |

| model | rmse | mae | direction_accuracy | r2_vs_zero |
|---|---|---|---|---|
| reg__basic__linear | 11.29% | 8.32% | 55.75% | -0.0005 |
| reg__basic__xgb | 11.19% | 8.26% | 55.06% | 0.0154 |
| reg__basic__forest | 11.26% | 8.30% | 54.54% | 0.0039 |
| reg__basic__equal_ensemble | 11.21% | 8.25% | 55.40% | 0.0125 |
| reg__direction_core__linear | 11.63% | 8.60% | 51.34% | -0.0633 |
| reg__direction_core__xgb | 11.35% | 8.42% | 53.24% | -0.0112 |
| reg__direction_core__forest | 11.33% | 8.40% | 52.72% | -0.0083 |
| reg__direction_core__equal_ensemble | 11.39% | 8.43% | 52.20% | -0.0198 |
| reg__price_direction__linear | 11.50% | 8.69% | 49.96% | -0.0392 |
| reg__price_direction__xgb | 11.30% | 8.37% | 52.81% | -0.0037 |
| reg__price_direction__forest | 11.45% | 8.49% | 49.27% | -0.0300 |
| reg__price_direction__equal_ensemble | 11.37% | 8.45% | 50.82% | -0.0159 |
| reg__cot_direction__linear | 11.74% | 8.72% | 51.77% | -0.0833 |
| reg__cot_direction__xgb | 11.25% | 8.36% | 53.67% | 0.0051 |
| reg__cot_direction__forest | 11.41% | 8.48% | 52.90% | -0.0230 |
| reg__cot_direction__equal_ensemble | 11.37% | 8.42% | 53.15% | -0.0160 |
| reg__weather_direction__linear | 11.57% | 8.55% | 55.23% | -0.0517 |
| reg__weather_direction__xgb | 11.18% | 8.28% | 56.96% | 0.0186 |
| reg__weather_direction__forest | 11.22% | 8.33% | 53.33% | 0.0103 |
| reg__weather_direction__equal_ensemble | 11.26% | 8.32% | 55.92% | 0.0038 |
| reg__engineered_direction__linear | 12.14% | 9.15% | 51.86% | -0.1570 |
| reg__engineered_direction__xgb | 11.34% | 8.38% | 54.19% | -0.0096 |
| reg__engineered_direction__forest | 11.39% | 8.47% | 50.48% | -0.0194 |
| reg__engineered_direction__equal_ensemble | 11.46% | 8.47% | 52.64% | -0.0315 |
| selected_return | 11.19% | 8.26% | 55.06% | 0.0154 |

These ablations keep definitions fixed but choose each family's configuration on CV. Differences include that train-only parameter selection; they are not isolated causal feature effects. Larger feature banks can underperform smaller banks.

## Optional uncertain indication

Frozen band: Down below 0.475, Up at/above 0.475, uncertain between. Holdout coverage is 100.00% (1156 accepted / 1156 binary origins).

Conditional Up precision/recall: 54.84% / 52.92%; Down: 51.21% / 53.14%. Unconditional recalls, counting withheld calls, are Up 52.92% and Down 53.14%.

## Precision-first and balanced-recall policies

These additional policies use the same 24 frozen classifier recipes and a fixed pre-2022 threshold grid. Precision-first maximizes the lower of Up/Down precision subject to both CV recalls being at least 40%; balanced-recall maximizes the lower class recall. Both make a call for every origin. CV gates do not guarantee holdout recall; this table records that tradeoff without replacing the main recipe.

| policy | recipe | threshold | accuracy | macro_f1 | up_precision | up_recall | down_precision | down_recall |
|---|---|---|---|---|---|---|---|---|
| precision_first | class__weather_direction__linear | 0.6000 | 48.88% | 46.35% | 51.31% | 26.21% | 48.00% | 73.25% |
| precision_first__original_macro_f1_threshold | class__weather_direction__linear | 0.5250 | 50.43% | 50.15% | 52.77% | 41.40% | 48.83% | 60.14% |
| recall_balanced | class__weather_direction__linear | 0.5750 | 50.35% | 48.88% | 53.46% | 32.22% | 48.93% | 69.84% |
| recall_balanced__original_macro_f1_threshold | class__weather_direction__linear | 0.5250 | 50.43% | 50.15% | 52.77% | 41.40% | 48.83% | 60.14% |
| original_selected_direction | class__basic__xgb | 0.4750 | 53.03% | 53.01% | 54.84% | 52.92% | 51.21% | 53.14% |

## Dependence and changes

The selected-minus-monthly macro-F1 difference has a paired 60-session block-bootstrap 95% interval [-8.15, +6.66] percentage points. Thirty-/90-session sensitivity intervals and 30 offsets of nonoverlapping outcomes are saved. Overlapping daily monthly labels are not independent samples.

The predefined simple basic ensemble's macro-F1 difference interval is [-5.15, +7.61] percentage points with 60-session blocks. Point gains do not establish a reliable advantage when this interval spans zero.

| model | delta_accuracy | delta_macro_f1 | delta_balanced_accuracy | delta_up_precision | delta_up_recall | delta_down_precision | delta_down_recall | rmse | baseline_rmse | rmse_relative_improvement |
|---|---|---|---|---|---|---|---|---|---|---|
| selected_direction | -1.99% | -0.62% | -1.43% | -0.37% | -16.86% | -3.43% | 14.00% | — | — | — |
| class__basic__equal_ensemble | 0.78% | 1.59% | 1.00% | 1.18% | -5.01% | 0.28% | 7.00% | — | — | — |
| selected_return | 0.09% | 0.07% | 0.08% | 0.06% | 0.17% | 0.14% | 0.00% | 11.19% | 11.11% | -0.75% |
| reg__basic__equal_ensemble | 0.43% | 0.85% | 0.53% | 0.57% | -2.17% | 0.25% | 3.23% | 11.21% | 11.11% | -0.90% |

## Research-paper parameter sweep

54 configurations varied ARIMA order/window/drift, AR lag/Ridge penalty, ELM width/activation/penalty/seed/window, and Holt seasonality/damping/window. Direction choice: `ar_lags63_ridge100`; return choice: `ar_lags5_ridge1000`.

| model | rmse | direction_accuracy | macro_f1 | up_precision | up_recall | down_precision | down_recall |
|---|---|---|---|---|---|---|---|
| legacy_fixed_arima_111 | 11.27% | 51.82% | 34.29% | 51.82% | 99.83% | 50.00% | 0.18% |
| legacy_fixed_holt_5 | 11.32% | 50.17% | 38.71% | 51.09% | 90.15% | 40.40% | 7.18% |
| paper_selected_rmse | 11.22% | 52.16% | 45.09% | 52.37% | 84.97% | 51.09% | 16.88% |
| paper_selected_macro_f1 | 11.23% | 56.40% | 55.27% | 56.41% | 69.78% | 56.39% | 42.01% |
| monthly_selected | 11.11% | 55.02% | 53.63% | 55.22% | 69.78% | 54.64% | 39.14% |

ARIMA uses the custom SciPy conditional-sum-of-squares extension of the existing implementation. It is not exact statsmodels maximum likelihood. Fits failing optimizer or stability checks cannot win. Details and official references are in [the paper search notes](../../../docs/PAPER_PARAMETER_SEARCH.md).

## Practical interpretation

The previous holdout has already influenced the wider research process. This run freezes choices before its holdout evaluation, but positive changes still need prospective confirmation. Direction and return selection serve different objectives; a direction gain can worsen return RMSE. COT/reanalysis revision caveats remain. Class-weighted classifier outputs are not independently calibrated market probabilities. News remains excluded.

No deep learning is promoted. There are only a few hundred nonoverlapping monthly training outcomes, and about 50–60 separate outcomes in the holdout. The paper ELM sweep provides a bounded nonlinear neural comparison without adding a large sequence network.
