# Paper parameter sweep: 28-calendar-day return

Selection used the original four pre-2022 chronological folds. Configurations and family winners were frozen before evaluating the same 2022+ origins.

The RMSE-selected configuration is `ar_lags5_ridge1000`; the macro-F1-selected configuration is `elm_h128_a10_tanh_s42`.

Macro-F1 winner holdout direction: **51.34%**, macro-F1: **0.504**, RMSE: **11.1725%**. The matched monthly baseline has **56.08%** direction and **0.545** macro-F1.

RMSE winner holdout RMSE: **10.7614%**, direction accuracy: **52.29%**, macro-F1: **0.438**. Original monthly selected RMSE: **10.6232%**; direction: **56.08%**.

| Frozen configuration/reference | RMSE | Direction | Down precision | Down recall | Up precision | Up recall | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ar_lags21_ridge100 | 10.7254% | 56.95% | 57.89% | 35.80% | 56.55% | 76.24% | 0.546 |
| ar_lags5_ridge1000 | 10.7614% | 52.29% | 50.00% | 14.10% | 52.64% | 87.13% | 0.438 |
| arima_100_w252 | 13.4263% | 52.98% | 50.58% | 62.57% | 56.42% | 44.22% | 0.528 |
| arima_111_no_drift_w1260 | 10.8235% | 49.87% | 47.73% | 53.16% | 52.30% | 46.86% | 0.499 |
| elm_h128_a10_tanh_s42 | 11.1725% | 51.34% | 48.78% | 39.60% | 52.96% | 62.05% | 0.504 |
| elm_h32_a1000_relu_s42 | 10.7878% | 53.75% | 52.72% | 29.84% | 54.14% | 75.58% | 0.506 |
| holt_s5_phi098_w1260 | 10.8338% | 49.35% | 46.43% | 39.96% | 51.39% | 57.92% | 0.487 |
| holt_s5_phi098_w252 | 11.2948% | 44.78% | 41.42% | 37.97% | 47.39% | 50.99% | 0.444 |
| legacy_fixed_arima_111 | 10.8053% | 52.20% | 33.33% | 0.18% | 52.25% | 99.67% | 0.345 |
| legacy_fixed_holt_5 | 10.8546% | 51.42% | 25.00% | 0.90% | 51.89% | 97.52% | 0.347 |
| simple_smoothing_w1260 | 10.8194% | 50.65% | 48.34% | 50.09% | 52.90% | 51.16% | 0.506 |
| simple_smoothing_w252 | 10.8181% | 50.73% | 48.43% | 50.09% | 52.98% | 51.32% | 0.507 |
| paper_selected_rmse | 10.7614% | 52.29% | 50.00% | 14.10% | 52.64% | 87.13% | 0.438 |
| paper_selected_macro_f1 | 11.1725% | 51.34% | 48.78% | 39.60% | 52.96% | 62.05% | 0.504 |
| monthly_selected | 10.6232% | 56.08% | 55.64% | 39.24% | 56.31% | 71.45% | 0.545 |
| previous_cv_ensemble | 10.9175% | 53.93% | 51.59% | 55.70% | 56.41% | 52.31% | 0.539 |
| previous_transferred_ensemble | 11.1462% | 53.41% | 51.26% | 47.74% | 55.12% | 58.58% | 0.531 |
| historical_mean | 10.7768% | 52.29% | 0.00% | 0.00% | 52.29% | 100.00% | 0.343 |
| zero | 10.8198% | 52.29% | 0.00% | 0.00% | 52.29% | 100.00% | 0.343 |
| training_majority_return_sign | 10.8198% | 47.71% | 47.71% | 100.00% | 0.00% | 0.00% | 0.323 |

These are regressors converted to direction by return sign. Exact-zero realized returns are excluded from binary metrics; predicted zero is assigned Up. Strict-sign accuracy retains the old neutral-zero convention and is also saved.

The development search contained 54 configurations; 0 had an optimizer/fitting failure and were ineligible. Every failure is retained in `fit_diagnostics.csv`, with no silent replacement or favorable-row selection.

ARIMA uses custom scipy CSS estimation, stable AR/MA coefficients, d=0/1, and 252/1,260-session histories. It is not an exact statsmodels likelihood replication. Price states use every observed close through each forecast origin; coefficients are frozen within each development fold and refitted quarterly in the holdout. Calendar forecast steps use only weekday counts, not the actual future market calendar.

Holdout results are exploratory: the historical holdout was examined previously, and overlapping monthly returns require block/non-overlapping diagnostics. No deep recurrent network was introduced: the small effective monthly sample and existing cache limitations do not justify its complexity. ELM width, ridge penalty, activation and seeds provide a controlled nonlinear sensitivity check.

See `cv_ranking.csv`, `holdout_by_year.csv`, `nonoverlapping_metrics.csv`, `metrics.json`, and the source/research note for exact parameters and limitations.

Paired 60-session moving-block direction change versus monthly selected: -4.75%; 95% interval [-8.71%, +0.78%]. An interval spanning zero leaves improvement uncertain.
