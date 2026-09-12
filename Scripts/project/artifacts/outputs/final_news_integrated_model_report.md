# Corrected Local News Evaluation

Delayed news does not improve RMSE or direction accuracy against the calibrated base on this holdout. This is exploratory evidence from previously selected local news summaries.

This report supersedes the earlier evaluation and saved model.

| Model | RMSE | MAE | Direction accuracy |
| --- | ---: | ---: | ---: |
| base_model | 0.05752 | 0.04530 | 53.19% |
| calibrated_base | 0.05734 | 0.04525 | 51.79% |
| news_only | 0.06156 | 0.04871 | 47.01% |
| final_news_blend | 0.06162 | 0.04855 | 47.01% |

Test: {'start': '2024-09-04', 'end': '2026-09-01', 'rows': 502}. Calibration: {'start': '2022-09-14', 'end': '2024-08-26', 'rows': 491}.

Recommendation selected on calibration validation RMSE: `base_model`.

## Method

Price-derived news scores are excluded. All news summaries are delayed six trading sessions because upstream row selection used five-session future returns. All calibration/validation/test boundaries purge labels whose maturity is on or after the next period's first date. Base-only calibration and news blends share a fixed Ridge alpha=3. No cloud scoring or downloads.

Daily features average five eligible trading sessions. Weekly features become eligible six price-calendar sessions after the week's last trading session. Missing news is imputed from training only. Counts describe retained events, not total news volume.

`gdelt_coffee_event_impact.py:score_events` includes future returns in impact scores; `gdelt_stream_all_years_coffee_events.py:update_day_buffers` selects events using those scores. `gdelt_weekly_ollama_batch_score.py:add_deterministic_period_score` also uses future returns and full-sample normalization. These scores are excluded from the corrected features.

## Validation

All fitted comparisons use the same preprocessing, Ridge alpha=3, training rows and test dates. Selection uses an inner purged calibration split; no test-driven tuning is performed. The bundle retains that selection plus the news candidate and base control, fitted on calibration only.

| Walk-forward model | RMSE | Direction accuracy |
| --- | ---: | ---: |
| base_model | 0.05752 | 53.19% |
| calibrated_base | 0.05750 | 50.00% |
| news_only | 0.05910 | 46.81% |
| final_news_blend | 0.05879 | 47.61% |

Three expanding-window folds are pooled above. Paired 20-session moving-block bootstrap versus calibrated base (1,000 samples; positive means improvement):

- RMSE reduction, 95% interval: [-0.009408761310160378, 1.3500852601472096e-06]
- Direction accuracy gain, 95% interval: [-0.16733067729083664, 0.06374501992031872]

## Limits

- The earlier 61.35% directional accuracy and positive news-lift conclusion are withdrawn: the impact scores and event selection used future returns.
- The stored daily-cap selection uses future five-session returns. Even counts and intensity are delayed until those returns have matured; this evaluates delayed selected-news summaries, not immediate news sentiment. Raw unfiltered history is not available locally.
- No original publication-time snapshots are available. The base prediction pipeline is reused, not independently certified as point-in-time correct. No separately identified Astra artifact was found.
- Five-session returns overlap. Block-bootstrap intervals and chronological checks describe this local sample; the previously inspected holdout is not a new untouched test set.

## Reproduce

Run from the repository root:

```bash
.venv/bin/python Scripts/project/scripts/integrate_news_scores_final_model.py
```

No network access or external data is used. The `.joblib` bundle's `model` is the validation-selected pipeline and `features` is its ordered input list. When `model` is `None`, use the existing base prediction unchanged. `news_blend_model` always retains the evaluated news candidate. The saved pipelines reproduce evaluation predictions; they are not refitted on test labels.
