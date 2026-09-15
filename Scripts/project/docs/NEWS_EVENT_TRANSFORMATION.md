# News And Event Transformation

I built this transformation to add local GDELT event information to the Arabica Coffee C return model without directly exposing the model to later price outcomes.

My modeling target is:

```text
target_return_5d = Close[t+5] / Close[t] - 1
```

I treat news as an independent input only after applying the availability rules below.

## Inputs

| Layer | File |
| --- | --- |
| Market, COT, and weather data | `data/centralData/arabica_ml_model_ready.csv` |
| Daily news summary | `Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_daily_summary.csv` |
| Weekly news summary | `Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_weekly_summary.csv` |
| Final training script | `Scripts/project/scripts/train_all_inputs_news_model.py` |
| News helpers | `Scripts/project/scripts/integrate_news_scores_final_model.py` |

I use the committed summaries for the final experiment. The much larger row-level event file under `data/events/` is only required when I rebuild the upstream extraction.

## Upstream Event Selection

The extraction scripts read GDELT archives in chunks, retain events with coffee or producing-region context, map calendar dates to Coffee C trading dates, and build compact daily and weekly summaries.

The upstream process also attached price context and used later returns in event ranking. As a result, fields such as price-response, direction-alignment, rule-impact, final-impact, future-return, and optional LLM/Ollama scores are retrospective diagnostics. I do not use them as predictive features.

More importantly, the retained set itself was selected using a five-session future-return window. I therefore assume that every aggregate derived from the selected rows is unavailable until that window has matured.

## Daily Features

I create daily features in `daily_news_features()` from:

| Source field | Use |
| --- | --- |
| `trading_date` | Trading-calendar join key |
| `event_count` | Retained event volume |
| `bullish_events` | Count with bullish inferred direction |
| `bearish_events` | Count with bearish inferred direction |

I then:

1. reindex the summary to the Coffee C trading calendar;
2. convert counts to numeric values;
3. calculate `log1p(event_count)`;
4. calculate `(bullish_events - bearish_events) / event_count`;
5. shift both series by six trading sessions;
6. calculate a five-session rolling mean with `min_periods=1`;
7. retain the latest source date for auditability.

The final columns are `news_daily_event_count_log_5d` and `news_daily_bull_minus_bear_share_5d`.

## Weekly Features

I create weekly features in `weekly_news_features()`. For each source week, I find the final Coffee C trading session on or before `period_end`, add six Coffee C sessions, and store that date as `weekly_available_date`.

I derive:

| Final field | Definition |
| --- | --- |
| `news_weekly_event_count_log` | `log1p(event_count)` |
| `news_weekly_bull_minus_bear_share` | Bullish minus bearish events divided by event count |
| `news_weekly_direction_concentration` | Absolute bullish-minus-bearish share |
| `news_weekly_explicit_coffee_share` | Explicit coffee events divided by event count |
| `news_weekly_mean_event_intensity` | Stored non-price mean event intensity |
| `news_weekly_text_disruption` | Log-transformed disruption-term count |
| `news_weekly_text_coffee` | Log-transformed coffee-term count |

I sort the weekly table by `weekly_available_date` and use a backward `merge_asof`. A market date can therefore receive only the most recent weekly row whose delayed availability date is not later than the market date.

## Text Indicators

I use fixed word-boundary dictionaries against `top_event_digest`:

| Indicator | Terms |
| --- | --- |
| Disruption | frost, drought, flood, wildfire, strike, conflict, war, sanctions, blockade, disease, shortage, export ban |
| Support | bumper, recovery, surplus, ceasefire, agreement, reopen |
| Coffee | coffee, arabica, cafe, caffeine, roaster |

Each value is `log1p(term_count)`. The support indicator is constant in the training slice and is removed by feature selection. I use these variables for transparent event characterization; they are not a trained sentiment model.

## Join And Split

I assemble the final frame in `build_frame()`:

1. I load the prepared market, COT, and weather data.
2. I verify the five-session target against `Close`.
3. I add derived COT positioning features.
4. I use the prepared `Date` values as the Coffee C calendar.
5. I left-join the delayed daily features on exact `Date` with a one-to-one validation.
6. I backward-as-of join the delayed weekly features.
7. I add `_row_id` for split and audit bookkeeping.
8. I split chronologically and purge five rows at the train/validation and validation/test boundaries.

## Matched Comparison

I compare three variants:

| Variant | Inputs |
| --- | --- |
| `without_news` | Price, COT, and weather |
| `structured_news` | Base inputs plus structured daily and weekly news |
| `all_inputs` | Structured-news inputs plus usable text counts |

I keep the learner recipe, sample rows, split dates, and preprocessing fixed across variants. I select the learner using no-news validation RMSE so that news does not influence the choice of algorithm before the matched comparison.

## Excluded Fields

I explicitly exclude:

- `future_return_1d`, `future_return_5d`, `future_return_10d`, and `future_return_20d`;
- price-response and direction-alignment scores;
- rule-impact and final-impact scores;
- deterministic-period scores;
- LLM/Ollama period scores;
- row-level post-event outcome fields.

I retain some of these columns in artifact files for retrospective audit only.

## News Impact

In the dashboard, I calculate:

```text
news impact = all-input predicted return - no-news predicted return
```

This quantity shows model sensitivity to the delayed news feature set. Because the two models are fitted separately and the inputs are observational, I do not describe it as causal attribution.

## Result

The news-integrated candidate achieved 5.5111% holdout RMSE and 50.55% direction accuracy, compared with 5.5055% and 50.65% without news. I found no evidence of an incremental forecasting benefit from the current news summaries.

I retain this implementation because it provides an auditable baseline for future work with a complete point-in-time news archive. The final decision is in [../../../Approach.md](../../../Approach.md), and the interpretation constraints are in [../../../Limitations.md](../../../Limitations.md).

