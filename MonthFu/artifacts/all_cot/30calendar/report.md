# Arabica 30 calendar day return experiment

Frozen CV recipe: `blend_0_shrink_0.25`, weights `{'engineered_extra': 0.25}`.

Holdout RMSE change versus adapted previous ensemble: +3.67% improvement. Skill versus zero squared error: +0.51%.

Origins 2022-01-03–2026-08-10. Quarterly fits use only labels ending before the quarter's first origin.

| Model | RMSE | MAE | Direction | Skill vs zero |
|---|---:|---:|---:|---:|
| selected | 11.25% | 8.30% | 50.82% | +0.51% |
| zero | 11.28% | 8.34% | 0.09% | +0.00% |
| historical_mean | 11.24% | 8.35% | 51.77% | +0.78% |
| previous_transferred_ensemble | 11.68% | 8.74% | 50.39% | -7.21% |
| previous_cv_ensemble | 11.70% | 8.75% | 50.73% | -7.56% |
| compact_legacy | 11.19% | 8.39% | 54.36% | +1.70% |
| prior_price_recipe | 11.11% | 8.19% | 54.97% | +3.01% |

Training-only majority-direction accuracy: 48.14%. Zero is neutral, so its direction accuracy is not a binary classifier benchmark.

Selected-minus-previous RMSE 95% interval (60-session blocks): [-0.009136620179630642, 0.00027497117138700164].

Selected-minus-zero RMSE 95% interval: [-0.0014444890978575662, 0.0011215202650153404]. An interval spanning zero does not establish a reliable improvement over the no-change forecast.

Latest cached origin: 2026-09-09, predicted return +0.53%, target approximately 2026-10-09. This is a cached-data forecast.

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
