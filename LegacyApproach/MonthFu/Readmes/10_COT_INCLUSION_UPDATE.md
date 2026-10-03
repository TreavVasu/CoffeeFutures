# Requiring all available COT in training and prediction

The October 1, 2026 requirement changes the default from unconstrained source selection to a model that receives the complete COT predictor schema. The earlier price-only winner remains a saved comparison. Full inclusion is a design constraint; it is not evidence that every field improves forecasts.

## Source history and publication

The supplied Coffee C cache has **2,447 reports**: 1,391 legacy futures-only and 1,056 disaggregated futures-only. The updated builder retains all of them, including **168** disaggregated snapshots previously excluded before September 2009. It maps all **118 legacy and 178 disaggregated measurement fields** into predictors. These counts are family-specific mappings, so shared source column names are counted within each report schema. This includes All/Old/Other position blocks, declared changes, percentages, trader counts, and concentration. Report dates, contract identifiers, source-file names and text remain traceability metadata.

An independent [archive coverage check](../artifacts/all_cot/cot_archive_coverage_audit.json) compares the combined cache with all **39 supplied per-archive extracts**: 27 legacy files and 12 disaggregated files. Every family/report-date key and original source value matches. No additional futures-and-options report family is present in the supplied COT folder.

Legacy reports begin January 4, 2000. The observed price table has **2,238 pre-2009 sessions** with released legacy positioning. Disaggregated live reports began September 4, 2009; older classifications were backcast and first published October 20, 2009. The archive's trailing-history contribution first becomes usable **October 21, 2009** in this cache. It never appears in pre-publication origin rows. September and early-October live reports keep their original release timing; archive activation uses the latest available snapshot rather than replacing it with older positioning. [CFTC launch announcement](https://www.cftc.gov/PressRoom/PressReleases/5710-09), [CFTC backcast release announcement](https://www.cftc.gov/PressRoom/PressReleases/5737-09).

The [release audit](COT_ALIGNMENT.md) distinguishes raw-value availability, current-report feature availability and expanded-history dependency availability. Known corrections and override delays still apply. Original publication vintages remain uncertified.

## Training constraint and missingness

The expanded bank contains **449 COT predictors**. The full bank has 1,032 predictors; `monthly_core_all_cot` has 521. All-COT direction groups contain 469 basic+COT, 501 core+COT and 653 engineered+COT inputs.

`--cot-policy all` is the default. Every eligible supervised candidate receives all numeric `cot_*` columns. Each selected member retains those columns even if they have less than 20% observed training coverage, are constant, or are entirely missing before their availability. Other predictors retain fold-local coverage/variance screening. Median imputation with empty-column retention, missing indicators and scaling is learned only on mature training rows. Source and exported feature rows retain their actual missing values. A model may learn zero contribution for an included field.

Selection considers only COT-complete recipes. Price-only, neutral and earlier-method candidates remain explicit comparisons. Recipe weights and thresholds are fixed using pre-2022 chronological validation; supervised labels must end strictly before each fold or quarterly refit. Direction models select by macro-F1 with per-class precision/recall reporting; return models select by RMSE. Thirty calendar days remains primary.

## Saved evidence and inference

Updated monthly artifacts are under `MonthFu/artifacts/all_cot/<horizon>`, and direction artifacts under `MonthFu/experiments/artifacts/all_cot/direction/<horizon>`. Earlier saved benchmarks are preserved.

- `training_dataset.csv.gz` contains the final supervised training rows with mature labels and complete source features.
- `prediction_inputs.csv.gz` contains the monthly causal inputs at all observed origins, including rows whose targets have not matured.
- `cot_feature_inclusion.csv.gz` records monthly per-fit COT coverage, uniqueness and retention; direction runs save equivalent structured audits.
- `feature_schema.json` and model metadata retain the COT feature schema and source mapping. Inference validates both schema and availability policy and requires retraining when they change.
- [The complete weekly source archive](../artifacts/all_cot/cot_source_reports.csv.gz) preserves all 2,447 source rows and their traceability columns; `cot_source_manifest.json` records the source hash. Finite trailing windows need not contain every past report at every prediction origin.
- CV/holdout forecasts, quarterly fit boundaries, metrics, source hashes and final bundles retain the existing reproducibility design.

Training and prediction read local caches. The last price origin remains **September 9, 2026**. Historical replay uses saved walk-forward predictions; final weights never recreate earlier holdout forecasts.

## Validation and measured change

The new causal and policy checks cover pre-2009 legacy rows, archived history activation, live-report timing before activation, overrides, prefix invariance, source-field mapping, sparse/constant/empty COT retention, COT-constrained selection and inference schema rejection. The suites pass **166 tests**: 59 monthly, 69 experiment and 38 original-project checks. All **five monthly runs passed** the [independent artifact audit](ALL_COT_AUDIT.md), including 1,148 common holdout origins. The notebook's artifact tables passed verification without starting a kernel. Direction evaluations are still running; earlier published results describe the original policy.

The primary selected recipe is **0.25 × engineered Extra Trees**, using full training history and all 449 COT predictors. Its final supervised dataset has **6,669** mature rows; inference inputs preserve all **6,690** origins. Every one of **71,840** required COT feature/fit records is retained across development, quarterly and final fits. Latest bundle replay error is below `1e-10`.

| Primary full holdout, 1,157 origins | RMSE | Strict-sign direction |
|---|---:|---:|
| New required-COT return model | 11.2537% | 50.8211% |
| Preserved price-only recipe | 11.1114% | 54.9697% |

Requiring all COT worsened return RMSE by **0.1423 percentage points**, or about **1.28%** relative, and reduced direction accuracy by **4.15 percentage points**. The 60-session paired block interval for the RMSE difference is **[+0.0206, +0.2734] percentage points**, favoring the earlier price recipe. The new model still improves point RMSE against the adapted older five-day ensemble, but its interval against zero-return spans zero. The updated default satisfies the requested input constraint; it is not a demonstrated accuracy upgrade. This comparison changes the selected feature group and shrinkage as well as COT coverage; it is not an isolated causal test of COT. COT inputs account for about **40.34%** of final tree split importance, with 368 source/indicator inputs receiving positive importance. This establishes learned usage, not forecasting value; [the importance table](../artifacts/all_cot/30calendar/final_tree_feature_importance.csv) is explanatory. Read [the saved primary metrics](../artifacts/all_cot/30calendar/metrics.json) and [run report](../artifacts/all_cot/30calendar/report.md).

Read [interview.md](interview.md) for the requirement, alternatives, rationale and production considerations.

[The complete monthly comparison](../RESULTS_ALL_COT.md) distinguishes gains against the older paper ensemble from changes against the former price recipe. The [fixed price comparison](../artifacts/all_cot/fixed_price_comparison.csv) uses 1,148 matched origins across all horizons. Required-COT point RMSE is higher at each horizon; direction improves at 21 calendar days and 30 observed sessions but falls at the other three. All five selected recipes fit the full history, rather than a recent-only window, and all have negative pre-2022 CV skill versus zero return. A shorter horizon's lower raw error does not establish a better forecasting method.

[Previous: 09 — Validation and historical results](09_VALIDATION_DISCARDS_AND_RESULTS.md) · [Return to the reading index](00_INDEX.md)
