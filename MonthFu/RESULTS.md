# MonthFu results

The primary target was fixed at 30 calendar days before inspecting the monthly holdout. Alternatives check sensitivity to one month and the weekly COT cycle. Horizon comparisons use common holdout origins and normalized squared-error skill; lower raw RMSE at a shorter horizon does not establish a better model.

Primary 30-calendar-day RMSE: 11.15%; adapted previous ensemble: 11.71%; relative RMSE improvement: +4.78%.

Best pre-holdout normalized CV skill among these horizon experiments: `28calendar` (+1.75%). This does not replace the fixed primary horizon.

| Horizon | CV skill vs zero | Holdout RMSE | Previous RMSE | Retuned previous RMSE | Improvement vs previous | Direction | Always up |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30calendar | +1.58% | 11.15% | 11.71% | 11.73% | +4.78% | 54.97% | 51.74% |
| 21sessions | +1.55% | 11.58% | 11.57% | 11.58% | -0.15% | 51.48% | 51.39% |
| 28calendar | +1.75% | 10.66% | 11.18% | 10.96% | +4.62% | 56.01% | 52.18% |
| 21calendar | +1.46% | 9.22% | 9.49% | 9.36% | +2.88% | 52.79% | 50.78% |
| 30sessions | +1.11% | 13.03% | 13.51% | 13.03% | +3.58% | 42.60% | 53.40% |

All horizon holdout comparisons above use 1148 common forecast origins. Full per-horizon tables can contain additional dates.

Previous is an adaptation of the existing RF/Holt-Winters/AR research ensemble to each new target. Its five-day validation weights are preserved, RF/AR are retrained on the new return target, and Holt-Winters forecasts the extended horizon. Basic features approximate the older feature family. It is not the exact saved five-day model.

COT snapshots are shifted to publication, then the first following observed Coffee C session; changes and extremes are calculated on weekly reports before daily filling. Delayed/corrected/backcast reports receive conservative guards. Weather uses a five-calendar-day availability proxy, and news is excluded by default because retrospective selection cannot be undone by delaying rows.

The historical holdout has been inspected in earlier project research. Positive improvement is exploratory. Read each horizon's report and moving-block confidence interval before interpreting small differences as reliable forecasting value.

Latest predictions originate from the last cached price date, not today's date. Empirical bands are not probability guarantees; futures rolls, fees, slippage and an executable strategy are outside this experiment.

Primary selected recipe: `{'price_extra': 0.5}`. The larger engineered COT/weather groups were tested but did not beat this conservative price model in validation.

Primary 60-session block interval for RMSE difference versus adapted previous: `[-0.010461766177803951, -0.0019695129086978727]`; versus zero: `[-0.003737931173259585, 0.0002966894393840861]`. The zero-reference interval spans zero, so a reliable edge over no change is not established.
