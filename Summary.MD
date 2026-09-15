# Arabica Futures Returns: All-Input Model Results

Updated 2026-09-12. The scope is to predict **Arabica Coffee C five-trading-day futures returns**, using price history, participant positioning, weather, and text-derived news features. The target is `Close[t+5] / Close[t] - 1`; direction accuracy is a secondary measure derived from the sign of that predicted return.

## Current Model And Inputs

A direct histogram gradient boosting model has now been trained with all four requested sources. This is a new all-input experiment, separate from the existing notebook ensemble.

The companion [ResearchModelTrainingNews.ipynb](ResearchModelTrainingNews.ipynb) preserves the original notebook's stages and adds an interactive news dashboard. Its tabs cover inputs, predictions, dashboard, news impact and source context; both latest-refit and pre-test holdout modes are available. The original notebook remains unchanged. See the [saved dashboard](Scripts/project/artifacts/plots/latest_arabica_news_prediction_console.png) and the execution check in `Scripts/project/artifacts/outputs/news_notebook_execution_check.json`.

| Source | Features supplied | Local input |
| --- | ---: | --- |
| Yahoo Finance futures history | 31 | Prepared OHLCV, returns, technical and calendar features |
| COT participant positioning | 46 | Released COT positioning and derived positioning/stress features |
| Coffee-region weather | 144 | Prepared features from the existing Open-Meteo cache |
| GDELT news | 9 | Seven event indicators plus two text-derived indicators |
| Total | 230 | 221 without news; 230 with news |

Market, COT and weather features come from `data/centralData/arabica_ml_model_ready.csv`. News comes from the existing `gdelt_coffee_events_2000_2026_daily_summary.csv` and `gdelt_coffee_events_2000_2026_weekly_summary.csv` under `Scripts/project/artifacts/outputs/`. All data and outputs remain inside this repository; this work downloaded no data and made no cloud-model scoring calls.

The news/event transformation logic is documented in [NEWS_EVENT_TRANSFORMATION.md](Scripts/project/docs/NEWS_EVENT_TRANSFORMATION.md). It traces raw GDELT extraction, daily and weekly transformations, six-session availability timing, as-of joins, leakage exclusions and the final news-impact calculation used for scoring.

The two text indicators count disruption terms and coffee-related terms in local event digests. A third candidate, support terms, was constant in training and was dropped. These are simple text proxies, not a trained article-sentiment model. COT describes aggregate participant categories, not individual counterparty transactions.

## Accuracy Changes

All rows below use the same **1,003 test dates, 2022-09-07 through 2026-09-01**. RMSE and MAE are expressed as percentages of the five-day return, not price-level errors.

| Model | Return RMSE | Return MAE | Direction accuracy |
| --- | ---: | ---: | ---: |
| Price + COT + weather, without news | 5.5055% | 4.3304% | 50.65% |
| Same model + structured news | 5.5111% | 4.3157% | 50.55% |
| Same model + structured news + text | 5.5111% | 4.3157% | 50.55% |
| Existing notebook ensemble, reference only | 5.3443% | 4.2291% | 50.15% |

Adding news changed direction accuracy by **-0.10 percentage points**, increased RMSE by **0.103% relative**, and slightly reduced MAE. There is **no demonstrated accuracy improvement from news in this run**. Test R-squared was -0.0695 without news and -0.0717 with news.

The extra text columns did not change holdout predictions: the evaluation trees used seven structured news features and no text features. The final refitted model does use all four source groups, including one text feature: 21 price, 42 COT, 101 weather and 8 news features appear in its tree splits. These usage counts establish participation, not predictive value.

Validation also preferred the model without news: RMSE 5.4227% versus 5.4441% with news. The paired 20-session block bootstrap gives a 95% interval of approximately -3.09 to +2.99 percentage points for news's direction-accuracy change. It provides no clear evidence of a gain. The existing notebook ensemble still has lower return RMSE than this direct all-input experiment; the new model has not replaced it in the notebook.

## Evaluation And Leakage Correction

- Training: 2000-01-03 to 2018-09-04, 4,674 rows.
- Validation: 2018-09-12 to 2022-08-29, 998 rows.
- Evaluation model refit: through 2022-08-29, 5,677 rows. Test labels were excluded.
- Five trading rows are purged at both boundaries so forward-return labels cannot cross into the next period.
- Ridge and histogram gradient boosting were compared using no-news validation RMSE. The chosen boosting recipe, training rows and dates are identical across the three feature variants.
- Future-price-derived impact scores are excluded. News is delayed **six trading sessions**, because the stored event selection itself used future five-day returns. Weekly features wait until the final source trading day's selection outcome has matured.
- The earlier **61.35% direction accuracy and positive news-lift claim are withdrawn**. That result contained future-price information. Delaying the original scores by only one day or one week was insufficient.

This evaluates delayed, retrospectively selected news summaries. The full unfiltered news history and original publication-time snapshots are unavailable locally, so this does not establish the value of immediate news sentiment. Weather and COT reuse the prepared dataset's alignment; historical release-vintage correctness has not been independently certified. The holdout has already been inspected in earlier experiments, so these are exploratory results.

## Saved Model And Reproduction

- Training script: [train_all_inputs_news_model.py](Scripts/project/scripts/train_all_inputs_news_model.py).
- All-input model bundle: [arabica_all_inputs_news_model.joblib](Scripts/project/artifacts/models/arabica_all_inputs_news_model.joblib).
- Full metrics and source-feature audit: [arabica_all_inputs_news_metrics.json](Scripts/project/artifacts/outputs/arabica_all_inputs_news_metrics.json).
- Paired comparison: [arabica_all_inputs_news_comparison.csv](Scripts/project/artifacts/outputs/arabica_all_inputs_news_comparison.csv).
- Test predictions: [arabica_all_inputs_news_holdout_predictions.csv](Scripts/project/artifacts/outputs/arabica_all_inputs_news_holdout_predictions.csv).
- Latest forecasts: [arabica_all_inputs_news_latest_predictions.csv](Scripts/project/artifacts/outputs/arabica_all_inputs_news_latest_predictions.csv).
- Evaluation plot: [arabica_all_inputs_news_evaluation.png](Scripts/project/artifacts/plots/arabica_all_inputs_news_evaluation.png).

The bundle's `model` and `features` are the refitted all-input candidate. `evaluation_models` retain the pre-test models for exact holdout reproduction. The production candidate was refitted on 6,685 labeled rows through 2026-09-01; those refitted weights were not used to calculate the test scores above.

Latest available forecast, dated **2026-09-09**: predicted five-day return **+1.3821%**, implied close **323.61**, from a close of **319.20**. This forecast has no measured future accuracy in the supplied data. It is included as a runnable model output, not as evidence of improvement.

```bash
.venv/bin/python Scripts/project/scripts/train_all_inputs_news_model.py
.venv/bin/python -m unittest discover -s Scripts/project/tests -p 'test_news_integration.py' -v
```

All 16 project tests passed, including seven focused tests for news maturity, holiday/session alignment, overlap purging, exclusion of price-derived scores, text matching, baseline comparability, and repository path limits. Saved evaluation and latest-model predictions were checked against their CSV outputs.

## Archived Notebook Output Summary

The following is the previous notebook-run summary, retained for context. Its feature counts describe all evaluated candidates, not necessarily the selected ensemble's active inputs. Inspection of the saved ensemble weights shows only `random_forest_technical`, `holt_winters_5d`, and `ar_ridge`; that selected blend does not consume weather, COT or news.

`ResearchModelTraining.ipynb` ends with a saved Arabica Coffee Futures prediction dashboard and a trained 5-trading-day return model.

## Final End State

- Target: predict Arabica Coffee C 5-trading-day forward return and direction.
- Final deployable model: `research_ensemble`.
- Saved model: `Scripts/project/artifacts/models/research_paper_ensemble.joblib`.
- Main prepared dataset: `data/centralData/arabica_ml_model_ready.csv`.
- Latest dashboard plot: `Scripts/project/artifacts/plots/latest_research_paper_return_prediction_console.png`.

## Data Used

- Dataset size: 6,690 daily rows from 2000-01-03 to 2026-09-09.
- Prepared dataset shape: 218 columns.
- Yahoo futures data: OHLCV price history and forward return targets.
- COT data: released-positioning features, forward-filled after report release timing.
- Weather data: Open-Meteo coffee-region daily weather cache.
- Events data: GDELT/news was not used in this run because the large local event file was missing.

## Feature Counts

- Final feature audit checked: 235 model features.
- Weather features: 144.
- COT features: 46 in the final model audit.
- Regime/context features: 13.
- News/event features: 0 in this run.

## Best Model Notes

- Best validation model: `research_ensemble`.
- Best holdout RMSE diagnostic: `arima_111`.
- Best holdout direction diagnostic: `elm_price_history`.
- Final choice kept: `research_ensemble`, because it was the validation-optimized blend used for the dashboard workflow.

## Accuracy Snapshot

- Validation `research_ensemble`: 54.91% direction accuracy, 5.02% return RMSE.
- Test `research_ensemble`: 50.15% direction accuracy, 5.34% return RMSE.
- Test best RMSE: `arima_111` at 5.33% return RMSE.
- Test best direction: `elm_price_history` at 51.74% direction accuracy.

## Main Outputs

- Predicted 5-day return.
- Positive, neutral, or negative threshold signal.
- Forecasted target close.
- Empirical 90% forecast range from holdout residuals.
- Dashboard plots for price history, COT positioning, context features, forecast path, and signal thresholds.
