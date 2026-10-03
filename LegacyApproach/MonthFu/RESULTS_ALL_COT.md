# MonthFu results

## Required-COT update versus the saved price approach

All five selected recipes use the full supervised history and retain all 449 COT inputs. The cache retains every supplied report, including the pre-launch archive with publication-based activation. This meets the inclusion requirement; it does not demonstrate an accuracy improvement.

On the primary complete holdout of 1,157 origins, required-COT RMSE is **11.2537%**, compared with **11.1114%** for the former price-only recipe. Strict-sign accuracy is **50.8211%**, compared with **54.9697%**. The RMSE-difference 60-session block interval is **[+0.0206, +0.2734] percentage points**, favoring the price reference.

The table below uses the same 1,148 common origins at every horizon. The price reference fixes the former primary recipe (0.5 × price Extra Trees) and refits it to each target; it is not necessarily the original winner of each alternative horizon. RMSE percentages measure arithmetic-return error. Direction is the original strict three-way sign metric.

| Horizon | Required-COT RMSE | Fixed price RMSE | Required-COT direction | Fixed price direction |
|---|---:|---:|---:|---:|
| 30calendar | 11.2924% | 11.1490% | 50.7840% | 54.9652% |
| 21sessions | 11.1524% | 10.9981% | 51.2195% | 54.4425% |
| 28calendar | 10.8237% | 10.6602% | 53.0488% | 56.0105% |
| 21calendar | 9.2736% | 9.2159% | 55.0523% | 52.7875% |
| 30sessions | 12.6135% | 12.4750% | 54.7038% | 51.8293% |

Some shorter/longer targets improve direction point estimates while worsening return RMSE. They do not replace the fixed primary horizon. Every selected recipe has negative pre-2022 CV skill against zero return. The adapted older paper ensemble below is a separate, weaker reference; gains against it do not establish gains against the former price-only winner.

[Design decisions](docs/interview.md) · [COT inclusion and dataset evidence](docs/10_COT_INCLUSION_UPDATE.md) · [Matched price comparison CSV](artifacts/all_cot/fixed_price_comparison.csv)

The primary target was fixed at 30 calendar days before inspecting the monthly holdout. Alternatives check sensitivity to one month and the weekly COT cycle. Horizon comparisons use common holdout origins and normalized squared-error skill; lower raw RMSE at a shorter horizon does not establish a better model.

Primary 30-calendar-day RMSE: 11.29%; adapted previous ensemble: 11.71%; relative RMSE improvement: +3.55%.

Best pre-holdout normalized CV skill among these horizon experiments: `30sessions` (-0.23%). This does not replace the fixed primary horizon.

| Horizon | CV skill vs zero | Holdout RMSE | Previous RMSE | Retuned previous RMSE | Improvement vs previous | Direction | Always up |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30calendar | -0.56% | 11.29% | 11.71% | 11.73% | +3.55% | 50.78% | 51.74% |
| 21sessions | -0.38% | 11.15% | 11.57% | 11.58% | +3.58% | 51.22% | 51.39% |
| 28calendar | -0.65% | 10.82% | 11.18% | 10.96% | +3.16% | 53.05% | 52.18% |
| 21calendar | -0.33% | 9.27% | 9.49% | 9.36% | +2.28% | 55.05% | 50.78% |
| 30sessions | -0.23% | 12.61% | 13.51% | 13.03% | +6.66% | 54.70% | 53.40% |

All horizon holdout comparisons above use 1148 common forecast origins. Full per-horizon tables can contain additional dates.

Previous is an adaptation of the existing RF/Holt-Winters/AR research ensemble to each new target. Its five-day validation weights are preserved, RF/AR are retrained on the new return target, and Holt-Winters forecasts the extended horizon. Basic features approximate the older feature family. It is not the exact saved five-day model.

COT snapshots are shifted to publication, then the first following observed Coffee C session; changes and extremes are calculated on weekly reports before daily filling. Delayed/corrected/backcast reports receive conservative guards. Weather uses a five-calendar-day availability proxy, and news is excluded by default because retrospective selection cannot be undone by delaying rows.

The historical holdout has been inspected in earlier project research. Positive improvement is exploratory. Read each horizon's report and moving-block confidence interval before interpreting small differences as reliable forecasting value.

Latest predictions originate from the last cached price date, not today's date. Empirical bands are not probability guarantees; futures rolls, fees, slippage and an executable strategy are outside this experiment.

Primary selected recipe: `{'engineered_extra': 0.25}`. COT selection policy: `all`; all-COT runs constrain every selected member to receive the complete numeric COT feature schema.

Primary 60-session block interval for RMSE difference versus adapted previous: `[-0.009136620179630642, 0.00027497117138700164]`; versus zero: `[-0.0014444890978575662, 0.0011215202650153404]`. The zero-reference interval spans zero, so a reliable edge over no change is not established.
