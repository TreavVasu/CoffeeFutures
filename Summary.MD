# Arabica Futures Research Summary

Updated 2026-09-12.

I built this project to estimate the five-trading-day forward return of Arabica Coffee C futures. I defined the target as:

```text
target_return_5d = Close[t+5] / Close[t] - 1
```

I treated return magnitude as the primary modeling problem and used the sign of the predicted return as a secondary direction measure. My final comparison combines price history, CFTC Commitments of Traders positioning, coffee-region weather, and delayed GDELT-derived news features.

## My Final Decision

I retained the existing `research_ensemble` as the deployable model because it produced the lowest holdout RMSE among the production candidates. I did not promote the news-integrated candidate because news did not improve out-of-sample performance after I corrected the feature timing.

| Model | Five-day return RMSE | Five-day return MAE | Direction accuracy |
| --- | ---: | ---: | ---: |
| Price + COT + weather | 5.5055% | 4.3304% | 50.65% |
| Same model + delayed structured news | 5.5111% | 4.3157% | 50.55% |
| Same model + delayed structured news + text | 5.5111% | 4.3157% | 50.55% |
| Existing `research_ensemble` | **5.3443%** | **4.2291%** | 50.15% |

I calculated these results on the same 1,003 holdout dates from 2022-09-07 through 2026-09-01. Adding news reduced direction accuracy by 0.10 percentage points, increased RMSE by 0.103% on a relative basis, and produced a small MAE reduction. I therefore found no defensible evidence that the available news features improved forecasting accuracy.

The paired 20-session block bootstrap placed the 95% interval for the news-related direction change at approximately -3.09 to +2.99 percentage points. Because the interval spans zero widely, I interpret the estimated news effect as statistically inconclusive.

## What I Used

| Source | Feature count | How I used it |
| --- | ---: | --- |
| Yahoo Finance | 31 | OHLCV history, returns, technical indicators, and calendar effects |
| CFTC COT | 46 | Released positioning, position changes, market concentration, and stress measures |
| Open-Meteo | 144 | Weather conditions for selected coffee-producing regions |
| GDELT summaries | 9 | Seven delayed event indicators and two usable text-count indicators |
| Total | 230 | 221 non-news features and 9 news features |

I sourced the market, COT, and weather features from `data/centralData/arabica_ml_model_ready.csv`. I sourced the news features from the committed daily and weekly GDELT summaries under `Scripts/project/artifacts/outputs/`. I used local data only for the final experiment; I did not call a cloud language model or download replacement data during training.

The news text features are intentionally simple. I counted disruption and coffee-related terms in the stored event digests. A support-term feature was constant in the training sample, so I removed it. I do not describe these counts as article sentiment.

## How I Controlled Leakage

The most important correction in this work concerned news timing. The original event-selection pipeline ranked retained events partly with five-session future returns. That made the earlier 61.35% direction result invalid. I withdrew that result and excluded every future-price-derived score from the predictive feature set.

I then delayed the remaining news counts and descriptors by six Coffee C trading sessions: five sessions for the upstream outcome window to mature and one additional session before availability. I applied the same rule to the daily and weekly summaries, using exact-date joins for daily features and backward as-of joins for weekly features.

I also used chronological train, validation, and test partitions with a five-row purge at both boundaries. This prevents a five-day forward-return label from crossing into the next evaluation period. I selected the learner on validation RMSE and fitted matched news and no-news models on identical rows.

My split was:

| Partition | Dates | Rows |
| --- | --- | ---: |
| Training | 2000-01-03 to 2018-09-04 | 4,674 |
| Validation | 2018-09-12 to 2022-08-29 | 998 |
| Holdout | 2022-09-07 to 2026-09-01 | 1,003 |

## What I Learned From News

The corrected experiment answers a narrow question: whether delayed, retrospectively selected event summaries improve this five-day return model. They did not.

This result does not show that publication-time news has no value. The repository does not contain a complete, unfiltered, point-in-time news archive, so I could not test a true real-time news signal. I therefore use the current news layer for context, diagnostics, and sensitivity analysis rather than as a claimed source of alpha.

In the dashboard, I define news impact as:

```text
all-input predicted return - matched no-news predicted return
```

This is a difference between two separately fitted models. I do not interpret it as causal attribution.

## Final Artifacts

- Notebook: [ResearchModelTrainingNews.ipynb](ResearchModelTrainingNews.ipynb)
- Selected production bundle: `Scripts/project/artifacts/models/research_paper_ensemble.joblib`
- News experiment bundle: `Scripts/project/artifacts/models/arabica_all_inputs_news_model.joblib`
- Full news metrics: `Scripts/project/artifacts/outputs/arabica_all_inputs_news_metrics.json`
- Paired comparison: `Scripts/project/artifacts/outputs/arabica_all_inputs_news_comparison.csv`
- Holdout predictions: `Scripts/project/artifacts/outputs/arabica_all_inputs_news_holdout_predictions.csv`
- Evaluation chart: `Scripts/project/artifacts/plots/arabica_all_inputs_news_evaluation.png`
- Saved dashboard: `Scripts/project/artifacts/plots/latest_arabica_news_prediction_console.png`

The news bundle keeps pre-test evaluation models for exact holdout reproduction. Its refitted all-input candidate uses all labeled observations through 2026-09-01, but I did not use those refitted weights to calculate the holdout metrics above.

## Reproduction

```bash
.venv/bin/python Scripts/project/scripts/train_all_inputs_news_model.py
.venv/bin/python -m unittest discover -s Scripts/project/tests -p 'test_news_integration.py' -v
```

I verified all 16 project tests, including checks for news maturity, session alignment, split purging, exclusion of price-derived scores, text matching, baseline comparability, and repository path handling.

For my rationale and final news-aware design, see [Approach.md](Approach.md). For the constraints on interpretation and deployment, see [Limitations.md](Limitations.md).
