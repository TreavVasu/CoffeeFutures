# News And Event Transformation Logic

This note documents how local GDELT news/events are transformed into model features and joined into the Arabica Coffee C return scoring build.

The final modeling target is `target_return_5d = Close[t+5] / Close[t] - 1`. News is treated as an independent input only after applying the timing rules below.

## Source Files

The final all-input news model reads local files only:

| Layer | File |
| --- | --- |
| Market, COT, weather model-ready data | `data/centralData/arabica_ml_model_ready.csv` |
| Daily news summary | `Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_daily_summary.csv` |
| Weekly news summary | `Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_weekly_summary.csv` |
| Final training script | `Scripts/project/scripts/train_all_inputs_news_model.py` |
| News feature helpers | `Scripts/project/scripts/integrate_news_scores_final_model.py` |

The large row-level raw event file, when regenerated locally, is written under `data/events/`. It is not required to run the saved notebook/dashboard model.

## Raw Event Extraction

Raw events are built by `gdelt_stream_all_years_coffee_events.py` and `gdelt_coffee_event_impact.py`.

The extraction pipeline:

1. Reads GDELT event archives in chunks.
2. Keeps only events that match either explicit coffee terms or coffee-producing country/region context.
3. Maps each event calendar date to the next Coffee C trading date within the allowed lag window.
4. Attaches Yahoo price/COT context for interpretation and event ranking.
5. Builds an event summary string from actors, event code, tone, Goldstein score and location.
6. Computes relevance, event intensity, expected direction and rule impact fields.
7. Writes daily and weekly summaries used by downstream experiments.

Important: the raw row-level event scores include fields such as `price_response_score_0_1`, `direction_alignment_score_0_1`, `rule_impact_score_0_1`, `final_impact_score_0_1`, future returns and optional LLM/Ollama scores. Those are useful for retrospective diagnostics and for creating compact historical summaries, but they are not valid direct inputs for the final predictive scoring model because some of them depend on subsequent Coffee C returns.

## Daily Transformation

Daily features are created by `daily_news_features()` in `integrate_news_scores_final_model.py`.

Input columns:

| Source column | Use |
| --- | --- |
| `trading_date` | Join key after converting to `Date` |
| `event_count` | Local event volume |
| `bullish_events` | Count of events with bullish inferred direction |
| `bearish_events` | Count of events with bearish inferred direction |

Transformation:

1. Reindex the daily summary to the Coffee C trading calendar.
2. Convert counts to numeric values.
3. Build `news_daily_event_count_log_5d = log1p(event_count)`.
4. Build `news_daily_bull_minus_bear_share_5d = (bullish_events - bearish_events) / event_count`.
5. Shift both daily series by `NEWS_DELAY = 6` trading sessions.
6. Apply a rolling 5-session mean with `min_periods=1`.
7. Add `daily_latest_source_date` to show which source trading date the delayed feature represents.

The six-session delay is deliberate: the upstream selected-event summaries were produced with a five-session future-return ranking step, so the final model waits for that five-session outcome to mature plus one additional session before using even the non-price daily counts.

## Weekly Transformation

Weekly features are created by `weekly_news_features()` in `integrate_news_scores_final_model.py`.

Input columns:

| Source column | Use |
| --- | --- |
| `period_id` | Weekly period identifier |
| `period_end` | Calendar end of the source week |
| `event_count` | Weekly event volume |
| `bullish_event_count` | Weekly bullish count |
| `bearish_event_count` | Weekly bearish count |
| `explicit_coffee_event_count` | Explicit coffee-related event count |
| `mean_event_intensity_score_0_1` | Non-price event intensity average |
| `top_event_digest` | Short digest used for fixed text indicators |

Transformation:

1. Convert `period_end` to a timestamp.
2. Find the last Coffee C trading session at or before `period_end`.
3. Set `weekly_available_date = last_source_session + 6 trading sessions`.
4. Drop weeks whose delayed availability date falls outside the local price calendar.
5. Convert numeric weekly columns to numeric values.
6. Build `news_weekly_event_count_log = log1p(event_count)`.
7. Build `news_weekly_bull_minus_bear_share = (bullish_event_count - bearish_event_count) / event_count`.
8. Build `news_weekly_direction_concentration = abs(news_weekly_bull_minus_bear_share)`.
9. Build `news_weekly_explicit_coffee_share = explicit_coffee_event_count / event_count`.
10. Carry forward `news_weekly_mean_event_intensity = mean_event_intensity_score_0_1`.
11. Add text indicators from `top_event_digest`.
12. Sort by `weekly_available_date` so the model can use an as-of join.

The final model uses `pd.merge_asof()` so each market date receives the latest weekly news row whose `weekly_available_date` is not after the market date.

## Text Indicators

Text features are created by `text_news_features()` from the weekly `top_event_digest`.

These are fixed dictionary counts, not a trained sentiment model:

| Feature | Terms counted |
| --- | --- |
| `news_weekly_text_disruption` | frost, drought, flood, wildfire, strike, conflict, war, sanctions, blockade, disease, shortage, export ban |
| `news_weekly_text_support` | bumper, recovery, surplus, ceasefire, agreement, reopen |
| `news_weekly_text_coffee` | coffee, arabica, cafe, caffeine, roaster |

Each feature is `log1p(term_count)` using word-boundary matching. If a text feature is constant in the training slice, the feature-selection step drops it.

## Join Into The Scoring Build

The all-input training frame is assembled by `build_frame()` in `train_all_inputs_news_model.py`.

Join order:

1. Load `arabica_ml_model_ready.csv` through the existing prepared-data helper.
2. Verify `target_return_5d` equals `Close.shift(-5) / Close - 1`.
3. Add derived COT positioning features.
4. Use the prepared market `Date` values as the Coffee C trading calendar.
5. Left-merge daily news features on exact `Date` with `validate="one_to_one"`.
6. As-of-merge weekly news features on `Date >= weekly_available_date`.
7. Add `_row_id` for split/audit bookkeeping.

The resulting frame is then split chronologically with a five-row purge between train, validation and test periods.

## Final Feature Groups

The direct all-input experiment compares three matched feature variants:

| Variant | Inputs |
| --- | --- |
| `without_news` | Yahoo price, COT positioning and weather |
| `structured_news` | `without_news` plus structured daily/weekly news indicators |
| `all_inputs` | `structured_news` plus weekly text indicators |

The same learner recipe, rows and chronological split are used for all variants. The learner is selected using the no-news validation RMSE, then fitted separately to each feature variant for a fair comparison.

## Excluded From Final Model Features

The final predictive scoring model does not use these retrospective fields as features:

- Future returns: `future_return_1d`, `future_return_5d`, `future_return_10d`, `future_return_20d`
- Price-response scores
- Direction-alignment scores
- Rule/final impact scores
- Deterministic period scores
- LLM/Ollama period scores
- Any row-level price response or post-event outcome columns

They remain in some artifact files for auditability and diagnostic plots, but the corrected final model only uses delayed event counts, direction shares, intensity, explicit-coffee share and fixed text indicators.

## Scoring Output Interpretation

In the dashboard and latest forecast output, `news impact` is computed as the difference between two separately fitted models:

```text
news impact = all-input predicted 5-day return - no-news predicted 5-day return
```

This is not causal attribution. It is a model-difference diagnostic showing how the model with delayed news features differs from the matched model trained without news.

## Known Limits

- The available summaries are retrospectively selected. The full unfiltered publication-time event history is not available locally.
- The six-session delay reduces leakage risk but means the experiment evaluates delayed selected-news summaries, not immediate article sentiment.
- COT and weather alignment follows the prepared dataset; this note focuses on the news/event transformation layer.
- The local text indicators are simple term-count proxies, not an LLM sentiment system.
