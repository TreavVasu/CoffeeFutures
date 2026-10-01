# Arabica 21 calendar day return experiment

Frozen CV recipe: `blend_0_shrink_0.25`, weights `{'price_cot_weather_extra': 0.25}`.

Holdout RMSE change versus adapted previous ensemble: +2.15% improvement. Skill versus zero squared error: +0.99%.

Origins 2022-01-03–2026-08-19. Quarterly fits use only labels ending before the quarter's first origin.

| Model | RMSE | MAE | Direction | Skill vs zero |
|---|---:|---:|---:|---:|
| selected | 9.28% | 6.98% | 55.15% | +0.99% |
| zero | 9.32% | 7.03% | 0.09% | +0.00% |
| historical_mean | 9.29% | 7.04% | 50.95% | +0.61% |
| previous_transferred_ensemble | 9.48% | 7.17% | 51.03% | -3.40% |
| previous_cv_ensemble | 9.36% | 7.04% | 50.52% | -0.73% |
| compact_legacy | 9.23% | 7.06% | 53.52% | +1.92% |
| prior_price_recipe | 9.22% | 6.93% | 53.01% | +2.22% |

Training-only majority-direction accuracy: 48.97%. Zero is neutral, so its direction accuracy is not a binary classifier benchmark.

Selected-minus-previous RMSE 95% interval (60-session blocks): [-0.0047694035729063935, 0.0003720302626644767].

Selected-minus-zero RMSE 95% interval: [-0.0012109685945617507, 0.00042387694445124183]. An interval spanning zero does not establish a reliable improvement over the no-change forecast.

Latest cached origin: 2026-09-09, predicted return +0.42%, target approximately 2026-09-30. This is a cached-data forecast.

See CSVs for label maturity, release alignment, every candidate's CV predictions, quarterly refits, non-overlapping samples and the complete feature manifest.

## Limitations

- Historical holdout has been reused in project research; results are exploratory.
- Previous ensemble is a method adaptation: five-day weights transferred, monthly RF/AR re-estimated on basic features, HW horizon extended; not the exact old saved model.
- COT historical releases include estimates and correction guards; no complete original-vintage archive.
- Weather cache is retrospective reanalysis; five-calendar-day lag does not remove future revisions.
- News excluded from primary search; even delayed summaries retain retrospective selection bias.
- Daily monthly labels overlap; block intervals and non-overlap samples account partly for dependence.
- Continuous contract rolls, price quality, fees and execution are not modeled.
- Latest forecast begins at last cached price date, with an approximate projected target date.
- Selected-CV error bands are exploratory and do not guarantee future coverage.
