# 28-calendar-day experiment

The direction recipe chosen before the 2022 holdout worsened holdout macro F1 by -1.73 percentage points versus the saved monthly model. Its accuracy is 52.80% versus 56.08%. These are exploratory reused-holdout results.

Binary metrics use 1,159 common nonzero-return origins; return metrics retain 1,159 origins. Up and down scores receive equal weight in macro F1. Pre-2022 selection used 2,016 expanding OOF origins and 72 classifier/regressor configurations.

## Selected and simple model comparisons

| model | accuracy | macro_f1 | balanced_accuracy | up_precision | up_recall | down_precision | down_recall |
|---|---|---|---|---|---|---|---|
| class__basic__equal_ensemble | 54.10% | 54.05% | 54.36% | 57.17% | 48.68% | 51.63% | 60.04% |
| reg__basic__equal_ensemble | 55.22% | 53.93% | 54.57% | 55.82% | 68.81% | 54.13% | 40.33% |
| selected_direction | 52.80% | 52.77% | 52.80% | 55.06% | 52.97% | 50.52% | 52.62% |
| selected_return | 54.79% | 53.02% | 54.01% | 55.27% | 70.96% | 53.81% | 37.07% |
| baseline_monthly_return | 56.08% | 54.50% | 55.35% | 56.31% | 71.45% | 55.64% | 39.24% |
| baseline_paper_return | 53.93% | 53.92% | 54.00% | 56.41% | 52.31% | 51.59% | 55.70% |
| baseline_majority | 47.71% | 32.30% | 50.00% | 0.00% | 0.00% | 47.71% | 100.00% |

### Return forecasts

| model | rmse | mae | direction_accuracy | r2_vs_zero |
|---|---|---|---|---|
| reg__basic__equal_ensemble | 10.74% | 7.97% | 55.22% | 0.0149 |
| selected_return | 10.71% | 7.98% | 54.79% | 0.0202 |
| baseline_monthly_return | 10.62% | 7.92% | 56.08% | 0.0360 |
| baseline_paper_return | 10.92% | 8.12% | 53.93% | -0.0182 |
| baseline_mean_return | 10.78% | 8.10% | 52.29% | 0.0079 |
| baseline_zero_return | 10.82% | 8.09% | 0.00% | 0.0000 |

RMSE/MAE percentages are arithmetic-return percentage points. Original regression direction accuracy treats predicted zero as neutral; binary tables treat it as an Up tie.

### Frozen recipes

Direction: `{"group": "weather_direction", "name": "class__weather_direction__equal_ensemble", "task": "class", "threshold": 0.4750000000000001, "weights": {"class__weather_direction__forest_1": 0.3333333333333333, "class__weather_direction__linear_1": 0.3333333333333333, "class__weather_direction__xgb_1": 0.3333333333333333}}`

Return: `{"group": "basic", "name": "reg__basic__xgb", "task": "reg", "threshold": 0.5, "weights": {"reg__basic__xgb_0": 1.0}}`

Each simple ensemble averages one CV-chosen linear model, one XGBoost model and one random forest with weights of 1/3. Its classifier threshold is also fixed from pre-2022 OOF. There is no learned stacker or holdout weight tuning.

## Feature and model comparisons

| model | accuracy | macro_f1 | balanced_accuracy | up_precision | up_recall | down_precision | down_recall |
|---|---|---|---|---|---|---|---|
| class__basic__linear | 53.41% | 53.40% | 53.47% | 55.83% | 52.15% | 51.10% | 54.79% |
| class__basic__xgb | 53.58% | 53.53% | 53.85% | 56.61% | 48.02% | 51.16% | 59.67% |
| class__basic__forest | 53.67% | 53.66% | 53.84% | 56.42% | 50.00% | 51.29% | 57.69% |
| class__basic__equal_ensemble | 54.10% | 54.05% | 54.36% | 57.17% | 48.68% | 51.63% | 60.04% |
| class__direction_core__linear | 51.08% | 50.53% | 51.67% | 54.52% | 38.78% | 49.04% | 64.56% |
| class__direction_core__xgb | 51.77% | 51.75% | 51.79% | 54.09% | 51.32% | 49.49% | 52.26% |
| class__direction_core__forest | 52.63% | 52.63% | 52.73% | 55.12% | 50.66% | 50.33% | 54.79% |
| class__direction_core__equal_ensemble | 51.94% | 51.46% | 52.51% | 55.61% | 40.10% | 49.72% | 64.92% |
| class__price_direction__linear | 50.47% | 50.43% | 50.45% | 52.73% | 50.99% | 48.17% | 49.91% |
| class__price_direction__xgb | 50.99% | 50.08% | 51.72% | 54.80% | 35.81% | 49.02% | 67.63% |
| class__price_direction__forest | 49.78% | 47.31% | 50.88% | 53.97% | 26.90% | 48.31% | 74.86% |
| class__price_direction__equal_ensemble | 51.16% | 50.49% | 51.81% | 54.78% | 37.79% | 49.12% | 65.82% |
| class__cot_direction__linear | 53.41% | 53.41% | 53.56% | 56.07% | 50.33% | 51.06% | 56.78% |
| class__cot_direction__xgb | 52.98% | 52.82% | 52.83% | 54.93% | 56.11% | 50.74% | 49.55% |
| class__cot_direction__forest | 49.44% | 49.27% | 49.81% | 52.06% | 41.75% | 47.55% | 57.87% |
| class__cot_direction__equal_ensemble | 51.25% | 51.11% | 51.11% | 53.33% | 54.13% | 48.90% | 48.10% |
| class__weather_direction__linear | 51.94% | 51.49% | 52.49% | 55.56% | 40.43% | 49.72% | 64.56% |
| class__weather_direction__xgb | 52.98% | 48.99% | 51.81% | 53.48% | 77.39% | 51.42% | 26.22% |
| class__weather_direction__forest | 53.32% | 53.32% | 53.43% | 55.86% | 51.16% | 50.99% | 55.70% |
| class__weather_direction__equal_ensemble | 52.80% | 52.77% | 52.80% | 55.06% | 52.97% | 50.52% | 52.62% |
| class__engineered_direction__linear | 50.47% | 50.42% | 50.73% | 53.11% | 45.05% | 48.37% | 56.42% |
| class__engineered_direction__xgb | 52.80% | 50.65% | 51.96% | 53.71% | 70.46% | 50.82% | 33.45% |
| class__engineered_direction__forest | 52.46% | 52.22% | 52.24% | 54.33% | 56.93% | 50.19% | 47.56% |
| class__engineered_direction__equal_ensemble | 54.53% | 54.13% | 54.22% | 55.98% | 61.06% | 52.61% | 47.38% |

| model | rmse | mae | direction_accuracy | r2_vs_zero |
|---|---|---|---|---|
| reg__basic__linear | 10.83% | 8.05% | 54.96% | -0.0014 |
| reg__basic__xgb | 10.71% | 7.98% | 54.79% | 0.0202 |
| reg__basic__forest | 10.77% | 7.99% | 55.05% | 0.0094 |
| reg__basic__equal_ensemble | 10.74% | 7.97% | 55.22% | 0.0149 |
| reg__direction_core__linear | 11.12% | 8.26% | 50.56% | -0.0562 |
| reg__direction_core__xgb | 10.85% | 8.13% | 51.51% | -0.0049 |
| reg__direction_core__forest | 10.84% | 8.13% | 52.63% | -0.0038 |
| reg__direction_core__equal_ensemble | 10.90% | 8.12% | 51.16% | -0.0144 |
| reg__price_direction__linear | 11.09% | 8.42% | 49.18% | -0.0506 |
| reg__price_direction__xgb | 10.86% | 8.11% | 52.20% | -0.0067 |
| reg__price_direction__forest | 11.04% | 8.35% | 49.78% | -0.0417 |
| reg__price_direction__equal_ensemble | 10.94% | 8.24% | 49.53% | -0.0227 |
| reg__cot_direction__linear | 11.24% | 8.42% | 51.34% | -0.0797 |
| reg__cot_direction__xgb | 10.77% | 8.07% | 54.53% | 0.0095 |
| reg__cot_direction__forest | 10.90% | 8.16% | 52.46% | -0.0143 |
| reg__cot_direction__equal_ensemble | 10.87% | 8.12% | 52.46% | -0.0102 |
| reg__weather_direction__linear | 11.10% | 8.24% | 53.93% | -0.0530 |
| reg__weather_direction__xgb | 10.70% | 8.01% | 54.62% | 0.0218 |
| reg__weather_direction__forest | 10.74% | 8.04% | 53.93% | 0.0142 |
| reg__weather_direction__equal_ensemble | 10.78% | 8.03% | 55.13% | 0.0064 |
| reg__engineered_direction__linear | 11.67% | 8.77% | 52.29% | -0.1631 |
| reg__engineered_direction__xgb | 10.84% | 8.10% | 54.70% | -0.0038 |
| reg__engineered_direction__forest | 10.89% | 8.12% | 53.15% | -0.0131 |
| reg__engineered_direction__equal_ensemble | 10.97% | 8.16% | 53.67% | -0.0276 |
| selected_return | 10.71% | 7.98% | 54.79% | 0.0202 |

These ablations keep definitions fixed but choose each family's configuration on CV. Differences include that train-only parameter selection; they are not isolated causal feature effects. Larger feature banks can underperform smaller banks.

## Optional uncertain indication

Frozen band: Down below 0.425, Up at/above 0.525, uncertain between. Holdout coverage is 68.33% (792 accepted / 1159 binary origins).

Conditional Up precision/recall: 55.96% / 57.07%; Down: 54.59% / 53.47%. Unconditional recalls, counting withheld calls, are Up 37.95% and Down 37.61%.

## Precision-first and balanced-recall policies

These additional policies use the same 24 frozen classifier recipes and a fixed pre-2022 threshold grid. Precision-first maximizes the lower of Up/Down precision subject to both CV recalls being at least 40%; balanced-recall maximizes the lower class recall. Both make a call for every origin. CV gates do not guarantee holdout recall; this table records that tradeoff without replacing the main recipe.

| policy | recipe | threshold | accuracy | macro_f1 | up_precision | up_recall | down_precision | down_recall |
|---|---|---|---|---|---|---|---|---|
| precision_first | class__basic__xgb | 0.5000 | 53.58% | 53.53% | 56.61% | 48.02% | 51.16% | 59.67% |
| precision_first__original_macro_f1_threshold | class__basic__xgb | 0.5000 | 53.58% | 53.53% | 56.61% | 48.02% | 51.16% | 59.67% |
| recall_balanced | class__weather_direction__equal_ensemble | 0.5000 | 53.49% | 53.35% | 56.85% | 45.87% | 51.04% | 61.84% |
| recall_balanced__original_macro_f1_threshold | class__weather_direction__equal_ensemble | 0.4750 | 52.80% | 52.77% | 55.06% | 52.97% | 50.52% | 52.62% |
| original_selected_direction | class__weather_direction__equal_ensemble | 0.4750 | 52.80% | 52.77% | 55.06% | 52.97% | 50.52% | 52.62% |

## Dependence and changes

The selected-minus-monthly macro-F1 difference has a paired 60-session block-bootstrap 95% interval [-9.53, +4.64] percentage points. Thirty-/90-session sensitivity intervals and 30 offsets of nonoverlapping outcomes are saved. Overlapping daily monthly labels are not independent samples.

The predefined simple basic ensemble's macro-F1 difference interval is [-7.50, +5.99] percentage points with 60-session blocks. Point gains do not establish a reliable advantage when this interval spans zero.

| model | delta_accuracy | delta_macro_f1 | delta_balanced_accuracy | delta_up_precision | delta_up_recall | delta_down_precision | delta_down_recall | rmse | baseline_rmse | rmse_relative_improvement |
|---|---|---|---|---|---|---|---|---|---|---|
| selected_direction | -3.28% | -1.73% | -2.55% | -1.25% | -18.48% | -5.12% | 13.38% | — | — | — |
| class__basic__equal_ensemble | -1.98% | -0.45% | -0.99% | 0.86% | -22.77% | -4.01% | 20.80% | — | — | — |
| selected_return | -1.29% | -1.48% | -1.33% | -1.04% | -0.50% | -1.84% | -2.17% | 10.71% | 10.62% | -0.82% |
| reg__basic__equal_ensemble | -0.86% | -0.57% | -0.78% | -0.48% | -2.64% | -1.51% | 1.08% | 10.74% | 10.62% | -1.09% |

## Research-paper parameter sweep

54 configurations varied ARIMA order/window/drift, AR lag/Ridge penalty, ELM width/activation/penalty/seed/window, and Holt seasonality/damping/window. Direction choice: `elm_h128_a10_tanh_s42`; return choice: `ar_lags5_ridge1000`.

| model | rmse | direction_accuracy | macro_f1 | up_precision | up_recall | down_precision | down_recall |
|---|---|---|---|---|---|---|---|
| legacy_fixed_arima_111 | 10.81% | 52.20% | 34.46% | 52.25% | 99.67% | 33.33% | 0.18% |
| legacy_fixed_holt_5 | 10.85% | 51.42% | 34.74% | 51.89% | 97.52% | 25.00% | 0.90% |
| paper_selected_rmse | 10.76% | 52.29% | 43.82% | 52.64% | 87.13% | 50.00% | 14.10% |
| paper_selected_macro_f1 | 11.17% | 51.34% | 50.43% | 52.96% | 62.05% | 48.78% | 39.60% |
| monthly_selected | 10.62% | 56.08% | 54.50% | 56.31% | 71.45% | 55.64% | 39.24% |

ARIMA uses the custom SciPy conditional-sum-of-squares extension of the existing implementation. It is not exact statsmodels maximum likelihood. Fits failing optimizer or stability checks cannot win. Details and official references are in [the paper search notes](../../../docs/PAPER_PARAMETER_SEARCH.md).

## Practical interpretation

The previous holdout has already influenced the wider research process. This run freezes choices before its holdout evaluation, but positive changes still need prospective confirmation. Direction and return selection serve different objectives; a direction gain can worsen return RMSE. COT/reanalysis revision caveats remain. Class-weighted classifier outputs are not independently calibrated market probabilities. News remains excluded.

No deep learning is promoted. There are only a few hundred nonoverlapping monthly training outcomes, and about 50–60 separate outcomes in the holdout. The paper ELM sweep provides a bounded nonlinear neural comparison without adding a large sequence network.
