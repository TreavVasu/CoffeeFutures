# Monthly direction experiments

This folder contains a separate follow-up to MonthFu: more directional features, explicit classifiers, return regressions, simple equal-weight ensembles, and a broader search of the research-paper families. All experiments use 30 calendar days as primary and 28 calendar days as a sensitivity comparison.

The default now requires all **449 COT predictors** in each fitted member. The three eligible groups contain 469 basic+COT, 501 core+COT and 653 engineered+COT inputs. Each group receives logistic/Ridge, XGBoost and random-forest models with two fixed parameter configurations per task: 36 candidates per horizon. Each ensemble averages its three family predictions equally: classifier probabilities for direction, predicted returns for regression. Family configurations and classifier thresholds are selected on pre-2022 expanding validation data.

[RESULTS.md](RESULTS.md) preserves the original six-group experiment, whose groups contained 20, 58, 99, 104, 73 and 236 predictors. [docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md) describes the 140 direction features and the updated COT policy. Read [the design interview](../docs/interview.md) for the inclusion requirement and publication decisions. All 2,447 supplied COT reports are retained; legacy data before 2009 is available historically, and backcast disaggregated data enters trailing history only after its archive publication.

The study optimizes macro F1 for direction so that both up and down calls matter. Outputs report each class's precision and recall, balanced accuracy, confusion counts, PR-AUC, and classifier probability losses. Regressions report RMSE, MAE and original strict-sign accuracy separately. Exact-zero realized returns are excluded consistently from binary classification, and retained for regression. A predicted zero is Up for binary threshold reporting and neutral for original strict-sign accuracy.

Run from the repository root with the existing environment:

```sh
.venv/bin/python MonthFu/experiments/scripts/train_direction.py --horizons 30 28 --jobs 2 --cot-policy all
.venv/bin/python MonthFu/experiments/scripts/audit_direction_experiments.py --folder MonthFu/experiments/artifacts/all_cot/direction/30calendar --folder MonthFu/experiments/artifacts/all_cot/direction/28calendar --output MonthFu/experiments/artifacts/all_cot/independent_direction_audit.json
.venv/bin/python -m unittest discover -s MonthFu/experiments/tests -v
.venv/bin/python MonthFu/experiments/scripts/predict.py --horizon 30
.venv/bin/python MonthFu/experiments/scripts/predict.py --horizon 30 --replay-date 2022-01-03
```

The earlier paper-parameter and precision/recall-policy studies retain their original artifact paths:

```sh
.venv/bin/python MonthFu/experiments/scripts/run_paper_sweep.py --horizons 30 28
.venv/bin/python MonthFu/experiments/scripts/precision_recall_policies.py --horizons 30 28
.venv/bin/python MonthFu/experiments/scripts/report_results.py
.venv/bin/python MonthFu/experiments/scripts/audit_direction_experiments.py
```

`artifacts/all_cot/direction/<horizon>/` holds current CV candidates, feature/model comparisons, frozen decisions, quarterly holdout forecasts, both sets of metrics, year slices, changes, overlap-aware uncertainty, selected/simple model bundles and cached-origin forecasts. It also preserves complete COT schemas, per-fit inclusion audits, mature training datasets and causal prediction inputs. Sparse, constant and entirely missing COT columns remain in each model's fitted schema; imputation stays local to its training rows. Inference rejects a changed schema or availability policy.

`artifacts/direction/<horizon>/` preserves the original benchmark. `train_direction.py --cot-policy benchmark` trains an unconstrained comparison under `artifacts/benchmark_retrained/`, while `predict.py --cot-policy benchmark` explicitly reads the original saved benchmark. `artifacts/paper_parameters/<horizon>/` holds ARIMA/AR/ELM/Holt parameter results and saved states. `docs/PAPER_PARAMETER_SEARCH.md` explains the actual order/lag/season/width/regularization search and its fitting limits.

No deep sequence network is added. Monthly labels overlap heavily: thousands of daily rows yield only a few hundred separate monthly training outcomes. The paper experiment instead tests bounded ELM hidden widths and regularization. This is a fixed random-feature neural model, not deep learning. See [docs/VALIDATION.md](docs/VALIDATION.md) for this assessment.

COT availability remains publication-based: Tuesday positions normally become usable on the first observed session after Friday publication, then carry as report state. Weekly changes are computed at release events. Missing predictors are imputed using training-fold medians only. Weather remains a delayed historical-reanalysis proxy; original-vintage weather and COT revision histories are incomplete. Retrospectively selected news remains excluded. Class-weighted classifier outputs are not independently calibrated market probabilities; Brier and log-loss expose that limitation.

The 2022+ historical holdout has already been examined in the earlier work. These results are exploratory comparisons, not fresh prospective evidence. Thresholds, abstention bands and parameters are nevertheless frozen from pre-2022 data before this run's holdout evaluation. The optional uncertain band reports coverage and unconditional recall, so withholding calls cannot silently inflate recall. Historical replay reads saved out-of-sample rows. Latest-model inference is only for the latest cached origin, currently 2026-09-09; it is not a live quote.
