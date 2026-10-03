# Arabica 30 sessions day return experiment

Frozen CV recipe: `previous_cv_ensemble`, weights `{'previous_rf': 0.25}`.

Holdout RMSE change versus adapted previous ensemble: +3.58% improvement. Skill versus zero squared error: -5.36%.

Origins 2022-01-03–2026-07-28. Quarterly fits use only labels ending before the quarter's first origin.

| Model | RMSE | MAE | Direction | Skill vs zero |
|---|---:|---:|---:|---:|
| selected | 13.03% | 10.08% | 42.60% | -5.36% |
| zero | 12.69% | 9.73% | 0.09% | +0.00% |
| historical_mean | 12.63% | 9.71% | 53.40% | +1.04% |
| previous_transferred_ensemble | 13.51% | 10.58% | 43.90% | -13.33% |
| previous_cv_ensemble | 13.03% | 10.08% | 42.60% | -5.36% |
| compact_legacy | 12.87% | 10.06% | 49.56% | -2.86% |

Training-only majority-direction accuracy: 46.52%. Zero is neutral, so its direction accuracy is not a binary classifier benchmark.

Selected-minus-previous RMSE 95% interval (60-session blocks): [-0.00990873327446038, -0.000704175700063069].

Selected-minus-zero RMSE 95% interval: [0.0004132945006263641, 0.007108345721911013]. An interval spanning zero does not establish a reliable improvement over the no-change forecast.

Latest cached origin: 2026-09-09, predicted return +3.49%, target approximately 2026-10-21. This is a cached-data forecast.

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
