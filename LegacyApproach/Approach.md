# Final Modeling Approach

## Objective

I designed the final workflow to predict the five-trading-day forward return of Arabica Coffee C futures while preserving point-in-time feature availability as far as the local data permits. My target is:

```text
target_return_5d = Close[t+5] / Close[t] - 1
```

I optimize return RMSE and report MAE and direction accuracy as complementary diagnostics. I do not optimize directly for a trading P&L because the repository does not yet include transaction costs, contract rolls, liquidity constraints, or a formal execution policy.

## Final Best Approach

My best supported approach is a two-layer design:

1. I use the validation-selected `research_ensemble` as the production return forecaster.
2. I keep a matched news-integrated model as a research and diagnostic layer, not as the primary forecast.

I chose this design because the production ensemble achieved a 5.3443% holdout RMSE, compared with 5.5055% for the matched price, COT, and weather model and 5.5111% for the news-integrated version. The news features did not improve direction accuracy either. Promoting the news model would therefore add complexity without a measured forecasting benefit.

## Data Design

I combine four information sets:

| Information set | Role in my design |
| --- | --- |
| Price and volume | Captures trend, mean reversion, volatility, liquidity, and recent market state |
| COT positioning | Represents weekly participant positioning and concentration after release timing |
| Regional weather | Represents growing-condition risk in selected coffee-producing regions |
| News and events | Adds delayed event volume, direction balance, intensity, coffee relevance, and fixed text counts |

I use the prepared daily market calendar as the master index. I only admit a feature on a date when its source information should be available under the implemented timing rule.

## Feature Construction

For price data, I use lagged returns, rolling volatility, relative volume, and normalized price-versus-average measures instead of relying on raw price levels alone.

For COT data, I use released positions, weekly changes, percent-of-open-interest measures, trader counts, concentration measures, and derived net positioning. I shift each report to reflect its publication delay before carrying it forward on the daily trading calendar.

For weather data, I use the existing Open-Meteo cache for Minas Gerais, Huila, and Dak Lak. I treat these variables as regional risk proxies rather than complete representations of global Arabica supply.

For news, I use only fields that do not directly encode future Coffee C returns:

- delayed daily event-count intensity;
- delayed daily bullish-minus-bearish share;
- delayed weekly event count;
- delayed weekly directional balance and concentration;
- delayed explicit-coffee share;
- delayed mean event intensity;
- fixed dictionary counts for disruption and coffee terms.

I exclude future returns, price-response scores, direction-alignment scores, final impact scores, deterministic scores, and LLM/Ollama scores from model inputs.

## News Timing Decision

The upstream event-selection process used five-session future returns when retaining and ranking events. Even apparently harmless aggregates, such as event counts, can inherit selection leakage from that process. I therefore wait six Coffee C trading sessions before making a summary available to the predictive model.

For daily data, I shift by six sessions and calculate a five-session rolling mean. For weekly data, I identify the final Coffee C session in the source week, add six sessions, and then use a backward as-of join. This is conservative, but it is the only defensible use of the stored selected summaries without reconstructing the raw publication-time archive.

## Model Selection And Evaluation

I split the data chronologically and purge five rows at each boundary so that forward-return labels cannot overlap the next period. I use validation RMSE to choose the model recipe. I then compare the news and no-news variants on identical observations, with the same learner settings and preprocessing.

I use a 20-session moving-block bootstrap when comparing direction accuracy because adjacent five-day targets overlap and are not independent. I treat the resulting interval as an uncertainty diagnostic, not as proof of economic significance.

## How I Use News In The Final Workflow

I retain news in three places:

- as a dashboard context layer that explains the recent event environment;
- as a sensitivity measure equal to the difference between the all-input and no-news model predictions;
- as a controlled research candidate for future point-in-time data improvements.

I do not treat the model difference as causal news attribution. I also do not claim that news adds alpha based on the current experiment.

## Decision Rule

My final selection rule is straightforward: I deploy the lowest-validation-error model unless a more complex candidate demonstrates a reproducible holdout improvement and passes the leakage audit. Under that rule, the `research_ensemble` remains the production model, while the corrected news model remains an experimental companion.

The implementation details are in `Scripts/project/docs/NEWS_EVENT_TRANSFORMATION.md`; the measured constraints are documented in [Limitations.md](Limitations.md).
