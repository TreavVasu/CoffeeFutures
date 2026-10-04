# Monthly feature construction

`MonthFu/src/features.py` rebuilds predictors from original Yahoo, COT and weather inputs. It does not extend the old, already-filled five-day model table. `build_feature_frame(repo_root)` returns a daily origin frame without targets, a fixed feature manifest and source audits. `build_frame(repo_root, horizon, unit)` adds the independently defined target through the training module.

The forecast origin is **after the observed Coffee C session close**. Same-session close, OHLC and volume therefore belong to the information set. A before-close application must lag those features again. All rolling statistics use the current or earlier rows, with initial history left missing. Source fields containing future returns, source identifiers and prepared targets never enter the feature groups.

## Horizons and units

Price-return windows of 21, 28 and 30 count **observed market sessions**, while weather `28d` or `30d` means **calendar days**. COT `4w` means four source reports; outages or catch-up releases can make its calendar duration differ from four weeks. No feature or label silently equates those units.

A 30-calendar-day target normally includes about one month of sessions and roughly four weekly COT releases. A 21-session target provides another practical monthly definition. Twenty-eight calendar days is exactly four calendar weeks. Thirty observed sessions is closer to six calendar weeks. COT's weekly cadence motivates looking at positioning over several reports; it does not require the future-return target to end on a release date. Training compares horizon definitions separately and never compares their raw RMSE as if their return distributions were identical.

## Price and volume

The feature bank preserves the existing normalized trend/volatility approach and extends it with fixed monthly hypotheses:

| Family | Examples | Construction |
| --- | --- | --- |
| Return and reversal | 21/28/30-session returns, short/long momentum changes, monthly acceleration | Trailing close ratios and trailing differences |
| Trend quality | Moving-average distance, drawdown, rebound, range position, log-price slope, efficiency | Only the trailing window; slope scaled by realized volatility |
| Volatility regime | Realized volatility, upside/downside asymmetry, skew/kurtosis, volatility ratios, volatility of volatility | Trailing return moments and short/long scale ratios |
| Gaps and daily ranges | Overnight gap mean/RMS, gap variance share, Parkinson and Garman-Klass range volatility | Current and earlier OHLC; inconsistent OHLC observations are masked |
| Participation | Relative volume, signed volume, return-volume correlation, illiquidity proxy | Trailing volume statistics; volume <= 1 is masked |
| Calendar | Annual harmonics, weekday harmonics, month progress, days to month end, session gaps | Dates already known at the origin |

Raw OHLCV is reserved for the `previous_technical` reference group. The larger engineered groups use dimensionless price transforms where possible. Relative volume uses the trailing median to reduce sensitivity to isolated volume extremes. Continuous-contract roll adjustments and exchange execution are not certified by these transformations.

## COT positioning and row alignment

`MonthFu/src/cot_release_features.py` owns source-family mappings and publication availability. It distinguishes the position snapshot date, the publication date and the first usable observed Coffee C session. Reports normally publish after the Coffee C trading session, so the first usable date is strictly later than publication. Known interruptions and announced exceptions override normal estimated processing dates. Historical CFTC archives may contain subsequent revisions; the release audit does not certify original unrevised vintages.

Weekly positioning features are calculated **before** carrying a report onto daily rows. A net change over four reports therefore represents a real report-to-report change, rather than four daily forward-filled rows. The bank includes participant net/gross/open-interest shares, weekly positioning flow, crowding relative to trailing report history, open-interest changes, concentration and age/quality indicators. The legacy and disaggregated definitions remain separate. Pre-launch disaggregated backcasts are retained but cannot influence predictors before their October 20, 2009 archive publication; the next observed session activates their historical windows. Pre-2009 legacy positioning is available after its own report releases.

Daily alignment uses backward as-of joins from the first usable session. There is no Tuesday snapshot-to-Tuesday predictor join and no backwards fill of early missing reports. The per-report CSV audit records publication and availability dates, exceptions and estimates. During an interruption, freshness/age features make carried values visible as older information. Missing unreleased values remain missing for training-fitted imputation.

## Weather

Weather observations are rolled on a complete **daily calendar**, with absent observations left missing. The lag rule then makes source date + five calendar days available to the observed price calendar. A backward as-of join carries only an already-available observation, and values become missing after seven additional calendar days without a fresh source observation. At the cache's last market date, 2026-09-09, this policy uses weather no later than 2026-09-04.

Rolling weather features span 7, 14, 28, 30, 60 and 90 calendar days. They cover precipitation and water balance, temperature, soil moisture, vapor-pressure deficit, dry spells, heavy rain, heat and cold. Added monthly stress variables combine dry conditions with heat/VPD, measure soil and rainfall persistence, and track 30-day water deficits. A missing rainfall day interrupts a dry spell instead of counting as drought.

Seasonal anomalies compare a source observation with the previous five occurrences of the **same calendar month**, requiring two previous month observations and at least 45 valid daily measurements. The entire current month is excluded from its own climatology; later days in that month cannot alter the baseline. Equal-weight Brazil/Colombia means and differences are regional risk summaries, with no assumed production weights. Vietnam remains a competing-origin proxy.

Brazil flowering September-November, harvest May-September and winter June-August interactions remain broad modeling hypotheses from the existing methodology. They are not farm-level phenology or quantitative yield-loss forecasts.

The cache is Open-Meteo historical **reanalysis**, not an archived publication-vintage dataset. A five-day lag controls the implemented timing proxy; it cannot remove revisions already present in historical reanalysis. Weather-based conclusions remain conditional on that limitation.

## Interactions

Interactions are specified before fitting. Participant positioning is combined with 21/28/30-session returns, four-report flows with normalized trends, and crowding with monthly RSI. Other fixed interactions connect trend with volatility regime, signed volume and gaps, and weather stress with trend or annual season. No full-history target correlation chooses them.

## Removed news layer

The GDELT event layer was removed rather than merely disabled. Its retained events were
selected using future five-session price responses and retrospective full-history
normalization, so the six-session maturity delay matured the direct return window but
could not reverse upstream selection, normalization or missing publication vintages. No
primary candidate group used it, and the earlier matched comparison found no return or
direction benefit. Its builders, constants, group, artifacts and documentation are gone,
and tests assert that no `news_` column or builder can reappear.

This is a source removal, not evidence that real-time news is worthless. A point-in-time
publication-vintage news corpus would be a different dataset and would need its own
evaluation.

## Fixed groups and validation

The updated supplied-cache feature manifest contains (original benchmark groups had 150 COT features):

| Group | Features | Purpose |
| --- | ---: | --- |
| `previous_technical` | 18 | Basic OHLCV/technical reference adapted to the longer target |
| `previous_history` | 13 | Trailing technical history for the original autoregressive recipe |
| `compact_legacy` | 18 | Small price/COT/weather reference; not an exact reproduction of old weights |
| `price` | 245 | Full trailing price/volume/calendar bank |
| `price_cot` | 694 | Price plus release-aligned positioning |
| `price_cot_weather` | 975 | Source ablation with regional weather |
| `monthly_core` | 104 | Original curated benchmark group |
| `monthly_core_all_cot` | 521 | Curated price/weather/interaction core plus all 449 COT predictors |
| `engineered` | 1032 | Full bank plus fixed interactions |

The former `experimental_news` group was removed with the GDELT layer.

All-COT mode retains every one of the 449 numeric COT predictors in each selected fit, including sparse/constant/empty fields. The source measurement map covers 118 legacy and 178 disaggregated columns, including All/Old/Other crop positions, declared changes, percentages, trader counts and concentration. Positions and changes are divided by current All open interest, percentages by 100, and nonnegative counts/open interest use log1p. Identifiers remain metadata. Non-COT counts describe candidate predictors, rather than the number necessarily retained in a fitted model. Availability/variance screening, imputation, scaling, supervised selection and regularization must use training rows inside each fold. The manifest's full-cache missing fractions are audit metadata, not a basis for selecting usable predictors.

The focused tests verify that changing future OHLCV cannot alter earlier features, rebuilding a price prefix reproduces the full-history prefix exactly, source-added target fields are rejected, monthly returns count observed sessions, weather cannot arrive before its lag, climatology excludes future/current-month observations, missing rainfall interrupts dry spells, stale gaps remain missing, and no `news_` column or news builder can reappear.

```sh
.venv/bin/python -m unittest discover -s MonthFu/tests -p 'test_features.py' -v
.venv/bin/python -m unittest discover -s MonthFu/tests -p 'test_external_return_features.py' -v
```
