# Thirty-day Arabica futures return model

The new pipeline predicts `Close[target_end_date] / Close[Date] - 1` using information available after the forecast origin's daily close. It rebuilds features from local source files rather than extending a previously filled five-day training table. The default horizon is **30 observed trading sessions**, approximately six weeks. `--horizon-unit calendar` instead uses the first observed market session on or after `Date + 30 calendar days`. Neither interpretation silently substitutes 21 or 28 sessions.

Run these commands from the repository root:

```sh
.venv/bin/python Scripts/project/scripts/train_30d_return_model.py
.venv/bin/python Scripts/project/scripts/predict_30d_returns.py
```

The saved model uses the selected recipe and all labels matured by the final cached price close. Prediction rebuilds current features from the local caches and uses those saved weights; it does not refresh data or refit. Retrain after refreshing sources when an updated fit is required.

Use separate output directories for alternate experiments:

```sh
.venv/bin/python Scripts/project/scripts/train_30d_return_model.py \
  --horizon 30 --horizon-unit calendar \
  --output-dir Scripts/project/artifacts/returns_30calendar

.venv/bin/python Scripts/project/scripts/predict_30d_returns.py \
  --bundle Scripts/project/artifacts/returns_30calendar/model.joblib \
  --output Scripts/project/artifacts/returns_30calendar/refreshed_forecast.csv

.venv/bin/python Scripts/project/scripts/train_30d_return_model.py \
  --quick --output-dir Scripts/project/artifacts/returns_30d_quick
```

`--include-experimental-news` allows news candidates and marks the bundle experimental. News is excluded from model selection by default because the stored news history was selected retrospectively.

## Availability and row alignment

The forecast rows are the unique, increasing dates with positive finite closes in `data/yahoo/arabica_coffee_futures_history.csv`. Weekends, holidays and missing price sessions are not inserted. Every session lag counts this observed calendar. A missing market observation can therefore postpone the modeled arrival of an external release.

| Input | When usable | Alignment and missing values |
|---|---|---|
| Price, OHLC, volume | After that session's close | Trailing features only. Inconsistent OHLC fields and volume ≤ 1 are masked; quality flags remain. No implicit forward fill in percentage changes. |
| COT | First observed session **strictly after publication day** | Backward asof join on usable session, separately for legacy and disaggregated futures-only reports. Most recently available report carries forward until a new release. Missing pre-release history remains missing. Snapshot/release age, estimated-date and stale-over-14-days flags accompany the carried values. |
| Weather | Observation date + **5 calendar days** | Roll on a complete daily weather calendar, then backward asof. Missing observations are not backfilled. An observation may carry for at most seven further calendar days after modeled availability; older feature values become missing and staleness/age remain visible. |
| Daily news, optional | Source session + **6 observed sessions** | Delay whitelisted counts before session-window rolling. Absent source rows stay missing, rather than becoming zero-event days. Features expire when the last source is more than 35 calendar days old. |
| Weekly news, optional | Last observed session on/before period end + **6 observed sessions** | Backward asof after availability. No incomplete future period is admitted. Feature values expire 35 calendar days after period end. |

All missing values remain missing until a model is fitted. Within each training fold or quarterly refit, features require at least 20% observed training coverage and more than one distinct observed value. Median imputation, missingness indicators and scaling are fitted only on those training rows. No full-history imputation, variance filtering or target correlation screening is used.

### COT publication date is different from the snapshot date

`data/COT/coffee_c_all_cot_data.csv` records the date positions were measured. Using a Tuesday snapshot in Tuesday's model, or filling it forward from Tuesday before publication, would introduce unavailable information. Normal COT publication is modeled at 15:30 America/New_York; the pipeline conservatively waits for the next observed coffee session. For an ordinary Tuesday report released Friday, it is generally Monday that first receives the new values. An observed-calendar holiday or data gap can postpone that further. The `cot_release_audit.csv` output records snapshot, publication, feature-publication and first usable dates for every report.

Ordinary historical publication dates are **estimates**: three federal business days after the report date, including specified mourning closures. Explicit dates or conservative bounds cover the 2013 shutdown, 2018–2019 shutdown catch-up, 2023 ION delays, 2025 mourning/shutdown interruptions, and published 2026 exceptions. Estimated versus announced/confirmed methods are retained in the audit. This is not a complete archive of verified historical release timestamps. The supporting [CFTC release calendar](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm), [special announcements](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm), and individual announcement links are also recorded by the module.

Supply additional verified dates in optional `data/COT/cot_release_overrides.csv`; these supersede built-in rules:

```csv
report_date,publication_date,source
2023-01-31,2023-02-24,https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm
```

Dates must be unique and nonmissing; `source` is required; publication cannot precede the snapshot. If release dates are out of snapshot order, a rolling feature waits until every preceding input report is published. Reports sharing the same usable session are collapsed to the latest snapshot for daily alignment, while earlier reports remain in the weekly calculations. Disaggregated history before September 2009 is excluded, including from rolling windows, because those earlier observations were published as retrospective backcasts.

COT changes, z-scores and positioning indices are calculated on **weekly reports before daily alignment**. A four-week change therefore compares reports four positions apart, not four repeated daily rows. Report-gap and staleness fields expose interruptions. COT is aggregate participant positioning, not physical coffee delivery or individual counterparty information; the 30-session forecast horizon does not alter its publication schedule.

### Weather and news limitations

Weather features cover Minas Gerais, Huila and Dak Lak. Calendar windows of 7, 30, 60 and 90 days describe temperature, precipitation, soil moisture, vapor-pressure deficit, water balance, dryness and heat/cold stress. Seasonal anomalies use the previous five occurrences of the same calendar month, excluding the entire current month and requiring at least two previous years and 45 observations. Brazil flowering, harvest and winter interactions use broad month indicators as hypotheses. Vietnam is an indirect competing-origin input.

The five-day weather lag is an availability proxy for historical reanalysis. The local cache has no original publication vintages, so later revisions cannot be removed with a lag. An audit of forecast vintages or archived observations would be required for a stronger point-in-time claim.

News inputs reuse local GDELT daily/weekly summaries, with fixed word indicators for disruption, support and coffee relevance. Impact scores, future returns, price-response scores, deterministic scores and LLM scores are excluded; numbers embedded in text are not parsed. Six sessions cover the upstream five-session return-maturity dependency, but upstream event retention also used retrospective normalization, including year/full-history fallback quantiles. That selected universe cannot be repaired by shifting rows. News results are an explicitly optional exploratory ablation.

## Features, model search and evaluation

Price features include returns and lags, trend/volatility, range position, drawdown, volume, RSI, downside movement and seasonal harmonics. COT adds net/gross positions normalized by open interest, weekly changes and acceleration, 26/52-week z-scores, 52/156-week indices, trader counts, concentration and old-crop open-interest shares. Fixed positioning–price and weather–season interactions extend these inputs without selecting features from future target relationships.

The feature groups are `compact_legacy`, `price`, `price_cot`, `price_cot_weather`, `engineered`, and optional `experimental_news`. The compact group provides a retrained approximation to the older approach on the **same 30-day target and splits**; old five-day metrics are not a comparable baseline. Full search considers regularized Ridge, histogram gradient boosting and Extra Trees across groups, plus engineered Elastic Net, random forest, ten-year training-window variants and the existing Extreme Learning Machine implementation. Zero-return and historical-mean forecasts are explicit baselines. Boosting's internal random-validation early stopping is disabled.

The local research motivates testing positioning extremes, weather-sensitive supply conditions and the ELM's regularized nonlinear features. These are hypotheses implemented from available inputs, not exact replications of every paper. Missing spot/FX/contract-delivery inputs are not manufactured, and research on price levels or other frequencies does not establish 30-session return predictability.

Selection and evaluation follow these steps:

1. Build targets with their **actual `target_end_date`**. The final 30 session rows have no session-horizon label and remain eligible for inference rather than fitting.
2. Reserve forecast origins from `2022-01-01` by default. Before that boundary, use four expanding validation folds of 504 sessions each; validation labels must also end before the holdout boundary.
3. At every validation/refit cutoff, train only on rows whose `target_end_date` is **strictly earlier** than the cutoff. This purges overlapping label information across the boundary using real dates, including for the calendar-day variant.
4. Rank candidates by pooled out-of-fold RMSE. Compare individual models, equal-weight combinations of up to three leading nontrivial candidates, and shrinkage factors 0.25/0.50/0.75/1 toward zero. Save the chosen recipe before measuring holdout performance.
5. Freeze that recipe through the holdout. Refit at each quarter's first observed holdout session using only then-mature labels; earlier holdout labels may enter a later quarter's training once matured. This is walk-forward evaluation, not a single frozen-weight holdout.
6. Compare selected forecasts with zero, expanding historical mean and the compact baseline on identical holdout rows. Report RMSE, MAE, directional accuracy, correlation and improvement against zero squared error. A zero forecast is neutral, so its sign accuracy is not a useful always-up/down classifier baseline.

Thirty-session forward returns from adjacent daily origins overlap heavily. The outputs include true-date non-overlapping samples, moving-block bootstrap RMSE comparisons with a 60-session primary block and 30/90-session sensitivity. The displayed empirical 80% error band is based on selection-CV absolute residuals; it has no prospective coverage guarantee. Selection and repeated use of historical data remain sources of uncertainty.

## Outputs and interpretation

Default outputs are in `Scripts/project/artifacts/returns_30d/`. Read the generated [report](../artifacts/returns_30d/report.md) for the actual selected model, evaluation and latest cached forecast. `metrics.json` contains detailed results and source-policy metadata; `selection.json`, `cv_folds.csv`, `cv_ranking.csv`, `holdout_refits.csv`, `feature_manifest.csv` and `cot_release_audit.csv` make the selection and alignment reviewable. `model.joblib` contains the final fitted bundle, and `latest_forecast.csv` records its origin and horizon.

The latest forecast starts at the **last local price date**, not today's date. Its projected session-horizon target date uses weekdays and does not project exchange holidays. The continuous-contract close series may contain roll effects, inconsistent OHLC and low-volume observations. A close-to-future-close forecast is not an executable profit estimate: fees, roll implementation and achievable entry prices are not modeled. The historical holdout has already been used in prior project experiments, so this evaluation is exploratory rather than fresh prospective evidence.

The earlier scripts, notebooks, prepared tables and model artifacts are preserved for reproducibility. New logic lives in [return_forecasting.py](../src/return_forecasting.py), [cot_release_features.py](../src/cot_release_features.py), [external_return_features.py](../src/external_return_features.py) and the new train/predict entry points. This keeps the existing acquisition pipelines and research available while making the changed horizon, publication handling and validation independently auditable.
