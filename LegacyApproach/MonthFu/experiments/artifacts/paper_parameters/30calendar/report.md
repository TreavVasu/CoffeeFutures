# Paper parameter sweep: 30-calendar-day return

Selection used the original four pre-2022 chronological folds. Configurations and family winners were frozen before evaluating the same 2022+ origins.

The RMSE-selected configuration is `ar_lags5_ridge1000`; the macro-F1-selected configuration is `ar_lags63_ridge100`.

Macro-F1 winner holdout direction: **56.40%**, macro-F1: **0.553**, RMSE: **11.2294%**. The matched monthly baseline has **55.02%** direction and **0.536** macro-F1.

RMSE winner holdout RMSE: **11.2189%**, direction accuracy: **52.16%**, macro-F1: **0.451**. Original monthly selected RMSE: **11.1114%**; direction: **55.02%**.

| Frozen configuration/reference | RMSE | Direction | Down precision | Down recall | Up precision | Up recall | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ar_lags5_ridge1000 | 11.2189% | 52.16% | 51.09% | 16.88% | 52.37% | 84.97% | 0.451 |
| ar_lags63_ridge100 | 11.2294% | 56.40% | 56.39% | 42.01% | 56.41% | 69.78% | 0.553 |
| arima_100_w252 | 14.0034% | 52.51% | 50.59% | 61.94% | 55.27% | 43.74% | 0.523 |
| arima_111_no_drift_w1260 | 11.2857% | 50.26% | 48.53% | 53.50% | 52.21% | 47.25% | 0.503 |
| elm_h32_a1000_relu_s42 | 11.2399% | 54.24% | 54.32% | 31.60% | 54.21% | 75.29% | 0.515 |
| elm_h32_a10_relu_s42 | 11.3463% | 51.64% | 49.75% | 35.19% | 52.62% | 66.94% | 0.501 |
| holt_s0_phi098_w252 | 11.4001% | 48.10% | 46.06% | 45.06% | 49.92% | 50.92% | 0.480 |
| holt_s21_phi098_w252 | 11.4254% | 49.65% | 47.64% | 45.24% | 51.36% | 53.76% | 0.495 |
| legacy_fixed_arima_111 | 11.2660% | 51.82% | 50.00% | 0.18% | 51.82% | 99.83% | 0.343 |
| legacy_fixed_holt_5 | 11.3172% | 50.17% | 40.40% | 7.18% | 51.09% | 90.15% | 0.387 |
| simple_smoothing_w1260 | 11.2815% | 49.91% | 48.07% | 49.19% | 51.71% | 50.58% | 0.499 |
| simple_smoothing_w252 | 11.2801% | 50.00% | 48.15% | 49.19% | 51.79% | 50.75% | 0.500 |
| paper_selected_rmse | 11.2189% | 52.16% | 51.09% | 16.88% | 52.37% | 84.97% | 0.451 |
| paper_selected_macro_f1 | 11.2294% | 56.40% | 56.39% | 42.01% | 56.41% | 69.78% | 0.553 |
| monthly_selected | 11.1114% | 55.02% | 54.64% | 39.14% | 55.22% | 69.78% | 0.536 |
| previous_cv_ensemble | 11.7009% | 50.78% | 49.00% | 52.78% | 52.70% | 48.91% | 0.508 |
| previous_transferred_ensemble | 11.6820% | 50.43% | 48.48% | 45.78% | 52.06% | 54.76% | 0.502 |
| historical_mean | 11.2379% | 51.82% | 0.00% | 0.00% | 51.82% | 100.00% | 0.341 |
| zero | 11.2823% | 51.82% | 0.00% | 0.00% | 51.82% | 100.00% | 0.341 |
| training_majority_return_sign | 11.2823% | 48.18% | 48.18% | 100.00% | 0.00% | 0.00% | 0.325 |

These are regressors converted to direction by return sign. Exact-zero realized returns are excluded from binary metrics; predicted zero is assigned Up. Strict-sign accuracy retains the old neutral-zero convention and is also saved.

The development search contained 54 configurations; 0 had an optimizer/fitting failure and were ineligible. Every failure is retained in `fit_diagnostics.csv`, with no silent replacement or favorable-row selection.

ARIMA uses custom scipy CSS estimation, stable AR/MA coefficients, d=0/1, and 252/1,260-session histories. It is not an exact statsmodels likelihood replication. Price states use every observed close through each forecast origin; coefficients are frozen within each development fold and refitted quarterly in the holdout. Calendar forecast steps use only weekday counts, not the actual future market calendar.

Holdout results are exploratory: the historical holdout was examined previously, and overlapping monthly returns require block/non-overlapping diagnostics. No deep recurrent network was introduced: the small effective monthly sample and existing cache limitations do not justify its complexity. ELM width, ridge penalty, activation and seeds provide a controlled nonlinear sensitivity check.

See `cv_ranking.csv`, `holdout_by_year.csv`, `nonoverlapping_metrics.csv`, `metrics.json`, and the source/research note for exact parameters and limitations.

Paired 60-session moving-block direction change versus monthly selected: +1.38%; 95% interval [-5.62%, +7.97%]. An interval spanning zero leaves improvement uncertain.
