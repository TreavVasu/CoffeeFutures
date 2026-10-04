# Arabica 28 calendar day return experiment

Frozen CV recipe: `blend_0_shrink_0.5`, weights `{'price_extra': 0.5}`.

Holdout RMSE change versus adapted previous ensemble: +4.69% improvement. Skill versus zero squared error: +3.60%.

Origins 2022-01-03–2026-08-12. Quarterly fits use only labels ending before the quarter's first origin.

| Model | RMSE | MAE | Direction | Skill vs zero |
|---|---:|---:|---:|---:|
| selected | 10.62% | 7.92% | 56.08% | +3.60% |
| zero | 10.82% | 8.09% | 0.00% | +0.00% |
| historical_mean | 10.78% | 8.10% | 52.29% | +0.79% |
| previous_transferred_ensemble | 11.15% | 8.30% | 53.41% | -6.13% |
| previous_cv_ensemble | 10.92% | 8.12% | 53.93% | -1.82% |
| compact_legacy | 10.77% | 8.15% | 55.31% | +0.84% |

Training-only majority-direction accuracy: 47.71%. Zero is neutral, so its direction accuracy is not a binary classifier benchmark.

Selected-minus-previous RMSE 95% interval (60-session blocks): [-0.009117718830137019, -0.0021367907548433813].

Selected-minus-zero RMSE 95% interval: [-0.0038903058693803964, -0.0001558261341049882]. An interval spanning zero does not establish a reliable improvement over the no-change forecast.

Latest cached origin: 2026-09-09, predicted return +0.90%, target approximately 2026-10-07. This is a cached-data forecast.

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
