# 05 — Monthly targets and horizons

The original target was a five-observed-session simple return: `Close[t+5] / Close[t] - 1`. Extending that task required an explicit definition of a month. Five-session and monthly error figures describe different return distributions and cannot be compared directly as a model improvement.

An existing implementation in [return_forecasting.py](../../Scripts/project/src/return_forecasting.py) rebuilt source features and supported actual-endpoint purging. Its default meant **30 observed trading sessions**, approximately six weeks; calendar days were optional. That implementation is part of the code history, but no corresponding saved `returns_30d` artifact directory was present during this review, so this account does not claim measured results for it.

MonthFu fixed **30 calendar days** as the primary target before inspecting the monthly holdout:

```text
scheduled date = forecast origin + 30 calendar days
target_end_date = first observed price date on or after the scheduled date
target_return = Close[target_end_date] / Close[origin] - 1
```

The actual endpoint is saved for every labeled origin. If the scheduled date is a weekend, holiday, or missing source session, the label uses the next observed close. Prices are not interpolated to create an endpoint. Origins whose future endpoint is not yet in the cache retain missing targets and remain usable for inference, rather than supervised fitting.

| Tested horizon | Meaning |
|---|---|
| 30 calendar days | Fixed approximately one-month primary question |
| 28 calendar days | Four calendar weeks |
| 21 calendar days | Three calendar weeks |
| 21 observed sessions | Approximately a trading month, with variable calendar duration |
| 30 observed sessions | Approximately six weeks |

COT's weekly positioning releases motivate multi-report flows and a 28-day comparison. They do not require the return target to end on publication day and are separate from physical futures delivery. Retrospective cycle diagnostics found a median of four usable COT update events inside both 28- and 30-calendar-day intervals. Those future event counts are diagnostics, never forecast features.

Horizon comparisons use common forecast origins and normalized squared-error skill against each horizon's own zero-return baseline: `1 - MSE(model) / MSE(zero)`. A shorter horizon's lower raw RMSE does not establish that it forecasts better. The 28-day alternative had the best pre-holdout normalized CV skill in the original monthly sweep, while 30 calendar days remained the fixed primary task.

Target construction lives in [modeling.py](../src/modeling.py). Read the [horizon results](../RESULTS.md) and [COT cycle diagnostics](COT_CYCLE_RESULTS.md) for the measured comparisons and their distinct row scopes.

[Previous: 04 — News](04_NEWS_PIPELINE_AND_EXCLUSIONS.md) · [Next: 06 — Availability and quality](06_AVAILABILITY_AND_DATA_QUALITY.md)
