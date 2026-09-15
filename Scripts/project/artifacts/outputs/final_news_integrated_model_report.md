# Corrected Local News Evaluation

I reran the local news experiment after removing future-price-derived news scores and correcting event availability. The delayed news candidate did not improve RMSE or direction accuracy relative to the calibrated base on this holdout. I treat this result as exploratory because the local summaries were selected retrospectively.

This report supersedes the earlier local-news evaluation and saved model.

| Model | RMSE | MAE | Direction accuracy |
| --- | ---: | ---: | ---: |
| Base model | 0.05752 | 0.04530 | 53.19% |
| Calibrated base | 0.05734 | 0.04525 | 51.79% |
| News only | 0.06156 | 0.04871 | 47.01% |
| Final news blend | 0.06162 | 0.04855 | 47.01% |

I evaluated 502 test rows from 2024-09-04 through 2026-09-01 after using 491 calibration rows from 2022-09-14 through 2024-08-26. Calibration validation RMSE selected the base model.

## Method

I excluded all price-derived news scores. I delayed every news summary by six Coffee C trading sessions because the upstream row selection used five-session future returns. At every calibration, validation, and test boundary, I purged labels whose target maturity reached the next period.

I fixed Ridge alpha at 3 for the calibrated base and news blends and used the same fitted preprocessing, training rows, and test dates. I calculated daily news features over five eligible sessions. I made each weekly row available six price-calendar sessions after its source week's last trading session. I fitted missing-value handling on training data only.

The retained event counts describe the upstream selected rows, not total news volume. I made no cloud-scoring calls and downloaded no additional data for this evaluation.

My leakage audit found three specific upstream issues:

- `gdelt_coffee_event_impact.py:score_events` used future returns in impact scores;
- `gdelt_stream_all_years_coffee_events.py:update_day_buffers` selected events using those scores;
- `gdelt_weekly_ollama_batch_score.py:add_deterministic_period_score` used future returns and full-sample normalization.

I excluded the affected scores from the corrected feature set and applied the six-session maturity delay to the remaining selected-summary aggregates.

## Walk-Forward Validation

| Candidate | Pooled RMSE | Direction accuracy |
| --- | ---: | ---: |
| Base model | 0.05752 | 53.19% |
| Calibrated base | 0.05750 | 50.00% |
| News only | 0.05910 | 46.81% |
| Final news blend | 0.05879 | 47.61% |

I pooled three expanding-window folds. I also ran a paired 20-session moving-block bootstrap against the calibrated base using 1,000 samples. Positive values represent improvement:

- RMSE reduction 95% interval: [-0.0094087613, 0.0000013501]
- Direction-accuracy gain 95% interval: [-0.1673306773, 0.0637450199]

Both comparisons fail to establish a reliable news benefit.

## Interpretation

I withdrew the earlier 61.35% direction result and its positive-news-lift conclusion because it contained future-price information. The corrected experiment evaluates delayed, selected summaries rather than publication-time news sentiment.

The overlapping five-session targets create serial dependence, and this holdout has been reviewed during prior research. I therefore use these metrics as an internal diagnostic rather than a final unbiased estimate of live performance.

## Reproduction

From the repository root, I run:

```bash
.venv/bin/python Scripts/project/scripts/integrate_news_scores_final_model.py
```

The saved `.joblib` bundle keeps the validation-selected pipeline and its ordered input list. When `model` is `None`, I use the existing base prediction unchanged. The `news_blend_model` field retains the evaluated news candidate. These pipelines reproduce the saved evaluation predictions and are not refitted on test labels.

