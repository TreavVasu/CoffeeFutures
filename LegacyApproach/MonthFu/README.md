# MonthFu: monthly Arabica futures returns

This folder extends the existing Arabica research into a monthly return model. It reuses the local price/COT/weather caches and paper-inspired model families, rebuilds trailing predictors with release-aware alignment, and preserves the earlier project files.

The fixed primary target is **30 calendar days**. For an origin `t`, the label is `Close[first observed session on or after t + 30 days] / Close[t] - 1`. Comparison experiments cover 21 observed sessions, 28 calendar days, 21 calendar days, and 30 observed sessions. The last interpretation is roughly six weeks, so it is explicitly separate from the monthly primary target. Calendar targets are scheduled day offsets, rather than a variable-length calendar month.

Open [MonthFu.ipynb](MonthFu.ipynb) for saved results and forecasts. [RESULTS_ALL_COT.md](RESULTS_ALL_COT.md) records the updated measured comparison, and [the COT inclusion update](docs/10_COT_INCLUSION_UPDATE.md) explains the constraint and data evidence. [RESULTS.md](RESULTS.md) preserves the original benchmark comparison. All generated work is stored in this folder; original source caches remain in their existing locations.

Read the [ordered documentation](docs/00_INDEX.md) for the complete start-to-now account of source extraction, ELT/ETL, data alignment, feature engineering, model selection, discarded approaches, direction experiments and validation.

The updated default requires the complete COT predictor schema in every selected model member. It retains all 2,447 cached reports, including 168 pre-launch disaggregated backcasts, and provides 449 COT features covering 118 legacy and 178 disaggregated source measurement fields. Legacy history before 2009 enters historical rows after release; disaggregated backcasts first inform history on the observed session after their October 20, 2009 publication. Read [interview.md](docs/interview.md) for the design decisions and [COT_ALIGNMENT.md](docs/COT_ALIGNMENT.md) for the timing policy. Original artifacts and results remain historical benchmarks.

## Run

From the repository root, using the existing environment:

```sh
.venv/bin/python MonthFu/scripts/train.py
.venv/bin/python MonthFu/scripts/predict.py
```

The first command runs the full 30-calendar-day search and produces `MonthFu/artifacts/all_cot/30calendar/`. The second rebuilds current features from local caches and predicts with those saved fitted weights. It does not refresh source data or retrain. Forecasts start at the **last cached price date**, not today's date. Refresh the existing acquisition pipelines and retrain when a newer model is required. `--cot-policy benchmark` permits source ablations in a separate `benchmark_updated` directory; it cannot overwrite the saved original benchmark by default.

All-COT runs export `training_dataset.csv.gz` with mature labels, `prediction_inputs.csv.gz` with causal inputs at every origin, `cot_feature_inclusion.csv.gz` with each fit's coverage/retention audit, and `feature_schema.json` with source-field mappings. Sparse, constant and not-yet-available COT fields stay in the fitted input schema. Their missing values are handled inside each training fit; historical source values remain missing. Inference rejects a changed COT schema or availability policy until retraining.

Run the complete horizon comparison:

```sh
.venv/bin/python MonthFu/scripts/run_experiments.py --jobs 2
```

Train an alternative or a separate smoke experiment:

```sh
.venv/bin/python MonthFu/scripts/train.py --horizon 21 --horizon-unit sessions
.venv/bin/python MonthFu/scripts/train.py --horizon 28 --horizon-unit calendar
.venv/bin/python MonthFu/scripts/train.py --quick --output-dir MonthFu/artifacts/smoke
```

Use a different saved model or replay an evaluated historical origin:

```sh
.venv/bin/python MonthFu/scripts/predict.py --bundle MonthFu/artifacts/all_cot/21sessions/model.joblib
.venv/bin/python MonthFu/scripts/predict.py --replay-date 2024-01-03
```

Historical replay reads the saved out-of-sample prediction. It does not apply final weights, which have seen later labels, to a historical date.

Verify the causal transformations and artifacts:

```sh
.venv/bin/python -m unittest discover -s MonthFu/tests -v
.venv/bin/python MonthFu/scripts/audit_artifacts.py --artifacts-dir MonthFu/artifacts/all_cot
.venv/bin/python MonthFu/scripts/check_notebook.py
```

The standalone dependency list is in [requirements.txt](requirements.txt). The existing repository environment is already sufficient.

Model tests and independent artifact audits were run. Notebook execution was blocked by the sandbox's local Jupyter socket restriction and its kernel permission request was canceled, so notebook execution and dashboard rendering remain unverified here. The per-horizon `holdout.png` plots were rendered and inspected. Run the notebook locally, or use `check_notebook.py --artifacts-only` to verify its source tables without starting a kernel.

## Publication dates and filling

| Source | First usable information | Alignment |
|---|---|---|
| Price/OHLCV | That observed session's close | Positive finite closes define the observed market calendar; inconsistent OHLC and volume ≤1 are masked. Trailing calculations never fill forward price changes. |
| COT | First observed session strictly after publication/correction availability | Calculate positioning and changes on weekly reports first, then backward-asof join to price dates. Carry only the latest published values; never backfill. Keep legacy/disaggregated definitions separate. |
| Weather | Observation date plus five calendar days | Compute weather windows on complete daily calendars, then backward-asof join. Missing/stale observations retain missingness. The cache is historical reanalysis, so this lag is a proxy rather than certified vintage availability. |

The former experimental news row was removed with the GDELT layer; price, COT and
weather are the only sources.

COT's Tuesday date describes the position snapshot. A normal Friday 15:30 Eastern release arrives after Coffee C's current 13:30 close; Monday usually first receives it. Holidays, shutdowns, delayed releases, and known corrections can postpone availability. A 28-day horizon spans roughly four report cycles; COT's weekly cadence is a reason to engineer four-report flows, not a guarantee of predictability or a coffee delivery schedule. See [COT_ALIGNMENT.md](docs/COT_ALIGNMENT.md) for sourced historical exceptions, uncertainty flags, override files, and row-level audit fields.

## Feature engineering and optimization

The updated feature universe has 1,032 engineered predictors and a 521-feature core with all COT. The earlier benchmark had 733 predictors and a 104-feature core. Price features cover monthly/quarterly momentum, lagged returns, slopes, volatility ratios, downside/upside asymmetry, range volatility, gaps, liquidity proxies, and seasonality. Weekly COT features capture open-interest-normalized net positions, 1/4/13-report flows, acceleration, z-scores and extremes, concentration, trader breadth, crop-year shares and release age. Weather contributes 7/14/28/30/60/90-day growing-condition stress and past-year seasonal anomalies. Fixed positioning/trend/volatility and weather/crop-season interactions connect these inputs.

The bounded full search tests Ridge, Elastic Net, histogram boosting, Extra Trees, XGBoost, and regularized ELMs across source groups. It also tests volatility-scaled targets using **origin-known** trailing volatility, recent five-/ten-year training windows, and exponential sample weighting. Median imputation, missing indicators and scaling are fitted only on each training fold. Coverage/variance filtering applies to non-COT inputs; required COT fields remain in the schema. Random internal validation splits are disabled. Cross-validation also selects conservative mixtures and shrinkage toward zero.

The [feature guide](docs/FEATURES.md) gives exact definitions and groups. The [research review](docs/RESEARCH_REVIEW.md) records which papers motivated the features and where price-level results cannot establish future-return accuracy.

The GDELT news layer was removed rather than merely disabled. Its stored event
selection used future five-session returns and full-history normalization, which a delay
cannot reverse, and no primary candidate group used it. `--include-experimental-news`
and the `experimental_news` group no longer exist. This is a source removal, not proof
that point-in-time news is worthless; see [FEATURES.md](docs/FEATURES.md).

## Evaluation and previous-model comparison

Four expanding validation blocks, each 504 observed sessions, precede a holdout beginning in 2022. All default horizons use the same conservative validation origins. At every fold and quarterly holdout refit, only labels with `target_end_date < cutoff` enter supervised training. The selected model recipe is saved before holdout evaluation and remains fixed through quarterly refits. A later refit may use earlier holdout labels after they mature; this is walk-forward evaluation.

The main references are zero return, historical mean, and the existing research ensemble **adapted to the new target**. Its RF and AR are re-estimated on basic technical inputs, and Holt-Winters forecasts the longer horizon. One reference preserves the old five-day validation weights, and another retunes the old component mixture on the same pre-holdout monthly folds. These are method comparisons, not a claim that the saved five-day model predicts a month directly. A compact older-style model is retained as a secondary reference.

Horizon comparisons use common holdout origins and normalized squared-error skill against zero. Direction is compared with a training-only majority-class benchmark. Monthly daily labels overlap, so reports include 60-session moving-block confidence intervals, 30-/90-session sensitivity, and real-date non-overlapping samples. `holdout_by_year.csv` exposes differences across market regimes.

[COT_CYCLE_RESULTS.md](docs/COT_CYCLE_RESULTS.md) records the original benchmark's comparison of new-report sessions with other forecast origins and describes how many weekly releases fall within each target. These are retrospective diagnostics, excluded from model features and selection. The default `cot_cycle_diagnostics.py` reads the original benchmark paths; it does not describe the new all-COT runs.

This holdout has been examined in earlier repository experiments. The results are exploratory rather than pristine prospective validation. Historical COT revisions, weather reanalysis, continuous-futures rolls, and missing contract-level execution details limit any trading interpretation. Empirical 80% error bands use selection-CV residuals and do not guarantee future coverage.

## Files

- `src/`: causal feature builders, publication calendar, model/evaluation primitives and existing paper-inspired algorithms.
- `scripts/`: training, horizon comparison, saved-model prediction, notebook verification and independent artifact audit.
- `tests/`: release/maturity, missingness, whitelist, prefix-invariance and evaluation tests.
- `docs/`: feature, COT, research and independent-review notes.
- `artifacts/all_cot/<horizon>/`: fitted bundle, latest forecast, candidate/CV predictions and ranking, holdout predictions/metrics/refits, annual metrics, non-overlap diagnostics, source/code hashes, feature manifest, availability rows and COT release audit.
- `artifacts/all_cot/horizon_comparison.csv` and `latest_forecasts.csv`: matched summary tables.
