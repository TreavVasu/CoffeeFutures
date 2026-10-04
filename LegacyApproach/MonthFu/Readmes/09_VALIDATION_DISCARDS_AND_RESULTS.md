# 10 — Validation, discards and results

Every experiment first constructed the target's **actual observed endpoint**. Supervised training required `target_end_date < first_validation_or_refit_origin`; a fixed row purge could misalign calendar horizons around holidays. Four expanding pre-2022 validation blocks of 504 origins produced 2,016 development predictions. Recipes and thresholds were frozen before quarterly 2022+ holdout refits, which admitted earlier labels only after maturity.

Feature screening retained nonconstant predictors with at least **20% observed training coverage**. Medians, missing indicators, scaling and class weights were fitted inside that training slice. Historical comparisons used saved out-of-sample forecasts on identical origins, rather than applying final models to earlier dates.

Discarding data and rejecting a model have different meanings:

| Decision | Treatment and reason |
| --- | --- |
| Invalid ancillary OHLC or volume ≤1 | Mask values and retain quality flags; preserve a valid close and its observed session. Invalid dates/nonpositive or nonfinite closes are rejected. |
| Stale/missing source state | Keep source-derived values unknown before train-fitted imputation; never repair earlier rows with future releases. |
| Constant or <20%-observed predictor | Remove from that fit using training statistics only. |
| Retrospectively selected news | Source removed. Delaying rows could not undo upstream future-return selection, so the GDELT layer, its group and its artifacts were deleted rather than excluded. |
| Larger feature banks or unsuccessful challengers | Retain research artifacts; do not promote them merely for complexity or a favorable holdout slice. |

The original monthly reference remains **ExtraTrees price-group prediction ×0.5**. Its full 30-day holdout RMSE was **11.11%** across 1,157 origins. The broader horizon comparison uses 1,148 common origins, explaining its different aggregate figure. Binary comparisons exclude one neutral realized return, leaving 1,156 rows; the old strict-sign score was 54.97%, versus 55.02% under the shared binary convention.

| 30-day direction indication | Accuracy | Macro-F1 | UP recall | DOWN recall |
| --- | ---: | ---: | ---: | ---: |
| Original monthly reference | 55.02% | 0.5363 | 69.78% | 39.14% |
| Predefined basic classifier ensemble | 55.80% | 0.5522 | 64.77% | 46.14% |
| Development-selected classifier | 53.03% | 0.5301 | 52.92% | 53.14% |

The basic ensemble improved both precision point estimates and DOWN recall while losing UP recall. Its macro-F1 change, +0.0159, had a 60-session block interval **[-0.0515, +0.0761]**. The paper AR(63)/Ridge challenger reached 56.40% direction but worsened return RMSE. New selected/simple return regressions also worsened RMSE, to 11.19%/11.21%. These [results](../experiments/RESULTS.md) do not justify replacing the reference.

Paired 30/60/90-session bootstrap blocks addressed overlapping labels; actual-endpoint nonoverlap samples contained only 56/61 intervals for 30/28 days. **139 tests** passed. [Independent audits](../experiments/artifacts/independent_direction_audit.json) reconstructed 137,012 scalar values, 1,944 fits and eight latest forecast replays, alongside the [original monthly audit](INDEPENDENT_AUDIT.md).

The historical holdout was reused, release/revision vintages remain incomplete, weather is delayed reanalysis, and classifier probabilities lack independent calibration. Latest inference starts at the cached **September 9, 2026** close. Point gains therefore remain exploratory; tests establish consistency and timing, not future accuracy.

The October 1 requirement changes the default to a COT-complete recipe while preserving these earlier measured benchmarks. Read [11 — Complete COT inclusion update](10_COT_INCLUSION_UPDATE.md) and [interview.md](interview.md) for the revised decision and validation evidence.

[Previous: 08 — Direction and parameter experiments](08_DIRECTION_AND_PARAMETER_EXPERIMENTS.md) · [Next: 10 — Complete COT inclusion](10_COT_INCLUSION_UPDATE.md)
