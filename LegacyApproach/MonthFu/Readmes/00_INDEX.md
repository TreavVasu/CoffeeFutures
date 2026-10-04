# Arabica futures research: ordered documentation

Documented October 1, 2026. This sequence records the complete explanation from the existing extraction and five-session research through the monthly models and direction experiments. It distinguishes implemented historical approaches, later corrections, candidates tested, exclusions, and measured results. Documentation updates do not refresh data or retrain models.

Read these files in numerical order:

| Order | Document | Contents |
|---|---|---|
| 01 | [Source extraction](01_SOURCE_EXTRACTION.md) | Yahoo, CFTC, Open-Meteo, local caches and acquisition boundaries |
| 02 | [ELT, ETL and data preparation](02_ELT_ETL_DATA_PREPARATION.md) | Staging, cleaning, joins, early COT filling and prepared tables |
| 03 | [Early models and leakage review](03_EARLY_MODELS_AND_LEAKAGE_REVIEW.md) | Five-session targets, research ensemble, historical selection problems and corrections |
| 04 | [Monthly targets and horizons](04_MONTHLY_TARGETS_AND_HORIZONS.md) | Calendar/session definitions, actual endpoints, COT cycles and fair comparisons |
| 05 | [Availability and data quality](05_AVAILABILITY_AND_DATA_QUALITY.md) | Forecast clock, publication alignment, missingness, weather lags and source quality |
| 06 | [Feature engineering](06_FEATURE_ENGINEERING.md) | Original monthly bank, directional additions, curated groups and train-only filtering |
| 07 | [Monthly model search](07_MONTHLY_MODEL_SEARCH.md) | Paper hypotheses, 46 candidates, parameter optimization, ensembles, shrinkage and references |
| 08 | [Direction and parameter experiments](08_DIRECTION_AND_PARAMETER_EXPERIMENTS.md) | Classifiers/regressors, simple ensembles, precision/recall policies, ARIMA/AR/ELM/Holt searches |
| 09 | [Validation, discards and results](09_VALIDATION_DISCARDS_AND_RESULTS.md) | Purging, quarterly refits, audits, exclusions, measured tradeoffs and current model decision |
| 10 | [Complete COT inclusion update](10_COT_INCLUSION_UPDATE.md) | Pre-2009 history, complete source mapping, required training/inference schema, saved datasets and revised evidence |

The former GDELT news layer and document 04 on it were removed. Its event retention
and ranking used subsequent five-session returns and full-history normalization, so a
delay could not make it point-in-time, and no primary candidate group used it. The
source is therefore absent rather than merely excluded, and no primary result depends
on a news decision.

```mermaid
flowchart LR
    A[External sources] --> B[Local source caches]
    B --> C[Clean and validate]
    C --> D[Align by information availability]
    D --> E[Trailing features and separate targets]
    E --> F[Purged chronological validation]
    F --> G[Freeze models and thresholds]
    G --> H[Quarterly walk-forward evaluation]
    H --> I[Saved models, forecasts and audits]
```

The original source caches are reused. The price cache ends on **2026-09-09**, so a latest-model forecast is a forecast from that cached close, not a current October market quote. The existing MonthFu return reference remains **0.5 × price-only Extra Trees**; direction challengers and parameter searches live separately under `MonthFu/experiments`.

The October 1 COT inclusion update changes the default training/prediction path to require all COT predictors and retains the pre-launch archive after its actual publication. New artifacts live under `MonthFu/artifacts/all_cot` and `MonthFu/experiments/artifacts/all_cot`; earlier benchmark results remain historical evidence. Read [interview.md](interview.md) for the full design discussion and the revised decision record.

## Detailed references and saved evidence

The existing specialist documents remain at their original paths so earlier links keep working:

- [Research and paper review](RESEARCH_REVIEW.md)
- [Monthly feature definitions](FEATURES.md)
- [COT publication alignment](COT_ALIGNMENT.md)
- [COT release-cycle diagnostics](COT_CYCLE_RESULTS.md)
- [Original monthly independent audit](INDEPENDENT_AUDIT.md)
- [Required-COT monthly independent audit](ALL_COT_AUDIT.md)
- [Required-COT monthly horizon results](../RESULTS_ALL_COT.md)
- [Original monthly horizon results](../RESULTS.md)
- [Original direction experiment results](../experiments/RESULTS.md)
- [Direction feature definitions](../experiments/docs/FEATURE_ENGINEERING.md)
- [Paper parameter search](../experiments/docs/PAPER_PARAMETER_SEARCH.md)
- [Experiment validation and final verification](../experiments/docs/VALIDATION.md)

The historical holdout has been reused, monthly labels overlap, and original source vintages are incomplete. The ordered account preserves gains and failures without presenting experimental point improvements as established future skill.

[Start with 01: Source extraction](01_SOURCE_EXTRACTION.md)
