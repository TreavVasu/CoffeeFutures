# Feature Design

I designed the feature set around one rule: every predictor must represent information that was available before the five-trading-day target return began to unfold.

My dependent variable is:

```text
target_return_5d = Close[t+5] / Close[t] - 1
```

I use the sign of this return to calculate direction accuracy, but I train and select the main models as return regressors.

## Price And Volume Features

I source daily OHLCV history from Yahoo Finance. I use `Date` as the time index and `Close` to construct the target. I do not use future OHLCV observations as inputs.

| Feature family | Examples | Why I included it |
| --- | --- | --- |
| Returns | `return_1d`, `return_5d`, lagged returns | Captures momentum and short-horizon reversal |
| Volatility | rolling return standard deviation, intraday range | Represents current uncertainty and price dispersion |
| Trend | moving averages, price-versus-average ratios | Expresses trend without relying only on the nominal price level |
| Volume | relative volume and volume changes | Represents changing market participation |
| Calendar | month and seasonal indicators | Allows recurring harvest and market-calendar patterns |

I calculate features only from observations at or before the scoring date. If the forecast is intended before the daily close, the same-day high, low, close, and volume must be excluded or lagged again.

## COT Positioning Features

I use CFTC Commitments of Traders data as an explanatory input, not as the target. The report describes aggregate futures positioning by trader category; it does not identify physical coffee transactions or matched counterparties.

I align each report according to its release timing before carrying values forward to the daily price calendar. My principal COT inputs include:

- open interest and changes in open interest;
- managed-money, commercial, producer/merchant, swap-dealer, and noncommercial positions;
- net long-minus-short measures;
- positions as a percentage of open interest;
- week-over-week position changes;
- trader counts and concentration ratios.

I rely more heavily on normalized and differenced measures than on raw contract counts because market size changes over time.

| Derived feature | Definition | Interpretation |
| --- | --- | --- |
| Managed-money net | long positions minus short positions | Directional speculative exposure |
| Managed-money net share | long percent of OI minus short percent of OI | Speculative exposure normalized by market size |
| Producer/merchant net | long positions minus short positions | Net physical supply-chain hedging pressure |
| Commercial net | commercial longs minus commercial shorts | Broad legacy-report hedging pressure |
| Weekly net change | change in longs minus change in shorts | New positioning impulse |
| Open-interest change rate | change in OI divided by OI | Expansion or contraction in participation |

I exclude report identifiers, archive names, contract text, and other traceability metadata from the numerical model.

## Weather Features

I use the existing Open-Meteo cache for Minas Gerais, Huila, and Dak Lak. I construct rolling and lagged measures from temperature and precipitation variables to represent heat, dryness, excessive rainfall, and persistence.

These features are regional proxies. I do not treat them as a complete agronomic model, and I do not infer exact production losses directly from them.

## News Features

I use the committed GDELT daily and weekly summaries. Because the upstream event-selection process used five-session future returns, I delay every retained news feature by six Coffee C trading sessions.

My corrected news inputs are:

| Feature | Construction |
| --- | --- |
| Daily event intensity | Five-session mean of `log1p(event_count)`, after a six-session shift |
| Daily direction balance | Five-session mean of bullish minus bearish share, after a six-session shift |
| Weekly event count | `log1p(event_count)` |
| Weekly direction balance | Bullish count minus bearish count, divided by event count |
| Weekly direction concentration | Absolute weekly direction balance |
| Explicit-coffee share | Explicit coffee events divided by total retained events |
| Mean event intensity | Stored non-price intensity average |
| Text disruption | `log1p` count of fixed disruption terms |
| Text coffee | `log1p` count of fixed coffee terms |

A support-term feature was constant in the training data, so I removed it. I treat the remaining text counts as transparent lexical proxies, not learned sentiment.

I exclude every news field that directly or indirectly records a later market outcome, including future returns, price-response scores, direction-alignment scores, rule-impact scores, final-impact scores, and deterministic or LLM/Ollama period scores.

## Final Feature Groups

The matched all-input experiment contains 230 features:

| Source | Count |
| --- | ---: |
| Price and calendar | 31 |
| COT | 46 |
| Weather | 144 |
| News | 9 |
| Total | 230 |

I compare three fixed variants on the same rows:

1. price, COT, and weather;
2. the same inputs plus structured news;
3. the same inputs plus structured news and text counts.

This design isolates the incremental contribution of news without changing the learner, split, or evaluation dates.

## Feature Governance

Before accepting a feature, I check:

- whether its source value was available on the scoring date;
- whether any upstream filter or transformation used the future target;
- whether the training-only statistics are isolated from validation and test data;
- whether missing values have an economic meaning or require training-fitted imputation;
- whether the feature is constant, duplicated, or an identifier;
- whether its incremental value survives a matched out-of-sample comparison.

The complete news timing and join logic is documented in [NEWS_EVENT_TRANSFORMATION.md](NEWS_EVENT_TRANSFORMATION.md). The final model choice is documented in [../../../Approach.md](../../../Approach.md).

