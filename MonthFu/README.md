# MonthFu: monthly Arabica futures returns

This folder extends the existing Arabica research into a monthly return model. It reuses the local price/COT/weather/news caches and paper-inspired model families, rebuilds trailing predictors with release-aware alignment, and preserves the earlier project files.

The fixed primary target is **30 calendar days**. For an origin `t`, the label is `Close[first observed session on or after t + 30 days] / Close[t] - 1`. Comparison experiments cover 21 observed sessions, 28 calendar days, 21 calendar days, and 30 observed sessions. The last interpretation is roughly six weeks, so it is explicitly separate from the monthly primary target. Calendar targets are scheduled day offsets, rather than a variable-length calendar month.

Open [MonthFu.ipynb](MonthFu.ipynb) for saved results and forecasts, and read [RESULTS.md](RESULTS.md) for the measured comparison. All generated work is stored in this folder; original source caches remain in their existing locations.

## Run

From the repository root, using the existing environment:

```sh
.venv/bin/python MonthFu/scripts/train.py
.venv/bin/python MonthFu/scripts/predict.py
```

The first command runs the full 30-calendar-day search and produces `MonthFu/artifacts/30calendar/`. The second rebuilds current features from local caches and predicts with saved fitted weights. It does not refresh source data or retrain. Forecasts start at the **last cached price date**, not today's date. Refresh the existing acquisition pipelines and retrain when a newer model is required.

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
.venv/bin/python MonthFu/scripts/predict.py --bundle MonthFu/artifacts/21sessions/model.joblib
.venv/bin/python MonthFu/scripts/predict.py --replay-date 2024-01-03
```

Historical replay reads the saved out-of-sample prediction. It does not apply final weights, which have seen later labels, to a historical date.

Verify the causal transformations and artifacts:

```sh
.venv/bin/python -m unittest discover -s MonthFu/tests -v
.venv/bin/python MonthFu/scripts/audit_artifacts.py
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
| News, experimental | Source maturity plus six observed sessions | Only whitelisted counts/text descriptors are admitted. Stored event selection used future outcomes and retrospective normalization; a lag cannot remove all selection bias. News is excluded from primary selection. |

COT's Tuesday date describes the position snapshot. A normal Friday 15:30 Eastern release arrives after Coffee C's current 13:30 close; Monday usually first receives it. Holidays, shutdowns, delayed releases, and known corrections can postpone availability. A 28-day horizon spans roughly four report cycles; COT's weekly cadence is a reason to engineer four-report flows, not a guarantee of predictability or a coffee delivery schedule. See [COT_ALIGNMENT.md](docs/COT_ALIGNMENT.md) for sourced historical exceptions, uncertainty flags, override files, and row-level audit fields.

## Feature engineering and optimization

The feature universe has 733 non-news engineered predictors and a curated 104-feature monthly core. Price features cover monthly/quarterly momentum, lagged returns, slopes, volatility ratios, downside/upside asymmetry, range volatility, gaps, liquidity proxies, and seasonality. Weekly COT features capture open-interest-normalized net positions, 1/4/13-report flows, acceleration, z-scores and extremes, concentration, trader breadth, crop-year shares and release age. Weather contributes 7/14/28/30/60/90-day growing-condition stress and past-year seasonal anomalies. Fixed positioning/trend/volatility and weather/crop-season interactions connect these inputs.

The bounded full search tests Ridge, Elastic Net, histogram boosting, Extra Trees, XGBoost, and regularized ELMs across source groups. It also tests volatility-scaled targets using **origin-known** trailing volatility, recent five-/ten-year training windows, and exponential sample weighting. Median imputation, missing indicators, scaling, and availability/variance filtering are fitted only on each training fold. Random internal validation splits are disabled. Cross-validation also selects conservative mixtures and shrinkage toward zero.

The [feature guide](docs/FEATURES.md) gives exact definitions and groups. The [research review](docs/RESEARCH_REVIEW.md) records which papers motivated the features and where price-level results cannot establish future-return accuracy.

To explore delayed news separately, use `--include-experimental-news --output-dir MonthFu/artifacts/news_experiment` with `train.py`. The resulting bundle is marked experimental because the selected news universe cannot be made fully point-in-time with a lag.

## Evaluation and previous-model comparison

Four expanding validation blocks, each 504 observed sessions, precede a holdout beginning in 2022. All default horizons use the same conservative validation origins. At every fold and quarterly holdout refit, only labels with `target_end_date < cutoff` enter supervised training. The selected model recipe is saved before holdout evaluation and remains fixed through quarterly refits. A later refit may use earlier holdout labels after they mature; this is walk-forward evaluation.

The main references are zero return, historical mean, and the existing research ensemble **adapted to the new target**. Its RF and AR are re-estimated on basic technical inputs, and Holt-Winters forecasts the longer horizon. One reference preserves the old five-day validation weights, and another retunes the old component mixture on the same pre-holdout monthly folds. These are method comparisons, not a claim that the saved five-day model predicts a month directly. A compact older-style model is retained as a secondary reference.

Horizon comparisons use common holdout origins and normalized squared-error skill against zero. Direction is compared with a training-only majority-class benchmark. Monthly daily labels overlap, so reports include 60-session moving-block confidence intervals, 30-/90-session sensitivity, and real-date non-overlapping samples. `holdout_by_year.csv` exposes differences across market regimes.

[COT_CYCLE_RESULTS.md](docs/COT_CYCLE_RESULTS.md) compares new-report sessions with other forecast origins and describes how many weekly releases fall within each target. These are retrospective diagnostics, excluded from model features and selection. Rebuild them with `.venv/bin/python MonthFu/scripts/cot_cycle_diagnostics.py` after the horizon sweep completes.

This holdout has been examined in earlier repository experiments. The results are exploratory rather than pristine prospective validation. Historical COT revisions, weather reanalysis, continuous-futures rolls, and missing contract-level execution details limit any trading interpretation. Empirical 80% error bands use selection-CV residuals and do not guarantee future coverage.

## Files

- `src/`: causal feature builders, publication calendar, model/evaluation primitives and existing paper-inspired algorithms.
- `scripts/`: training, horizon comparison, saved-model prediction, notebook verification and independent artifact audit.
- `tests/`: release/maturity, missingness, whitelist, prefix-invariance and evaluation tests.
- `docs/`: feature, COT, research and independent-review notes.
- `artifacts/<horizon>/`: fitted bundle, latest forecast, candidate/CV predictions and ranking, holdout predictions/metrics/refits, annual metrics, non-overlap diagnostics, source/code hashes, feature manifest, availability rows and COT release audit.
- `artifacts/horizon_comparison.csv` and `latest_forecasts.csv`: matched summary tables.
