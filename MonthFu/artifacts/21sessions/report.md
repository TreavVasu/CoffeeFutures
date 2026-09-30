# Arabica 21 sessions day return experiment

Frozen CV recipe: `previous_cv_ensemble`, weights `{'previous_rf': 0.5}`.

Holdout RMSE change versus adapted previous ensemble: -0.15% improvement. Skill versus zero squared error: -7.65%.

Origins 2022-01-03–2026-08-10. Quarterly fits use only labels ending before the quarter's first origin.

| Model | RMSE | MAE | Direction | Skill vs zero |
|---|---:|---:|---:|---:|
| selected | 11.56% | 8.70% | 51.60% | -7.65% |
| zero | 11.14% | 8.34% | 0.09% | +0.00% |
| historical_mean | 11.09% | 8.34% | 51.51% | +0.82% |
| previous_transferred_ensemble | 11.54% | 8.70% | 51.69% | -7.33% |
| previous_cv_ensemble | 11.56% | 8.70% | 51.60% | -7.65% |
| compact_legacy | 11.00% | 8.33% | 54.11% | +2.47% |

Training-only majority-direction accuracy: 48.40%. Zero is neutral, so its direction accuracy is not a binary classifier benchmark.

Selected-minus-previous RMSE 95% interval (60-session blocks): [-0.0012223521662219528, 0.0014730004070981234].

Selected-minus-zero RMSE 95% interval: [-0.0004981942044001636, 0.010206814972266953]. An interval spanning zero does not establish a reliable improvement over the no-change forecast.

Latest cached origin: 2026-09-09, predicted return +2.08%, target approximately 2026-10-08. This is a cached-data forecast.

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
