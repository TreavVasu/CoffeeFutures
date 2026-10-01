# 09 — Direction and parameter experiments

After the monthly return search, the next request focused on more accurate direction indications, precision and recall, additional features, and a simple XGBoost–linear–random-forest ensemble. This follow-up lives under [experiments](../experiments/README.md), preserving the original monthly model. Thirty calendar days remained primary; 28 calendar days supplied a fixed sensitivity comparison.

The follow-up added **140 causal predictors**, including trailing price ranks, persistence and reversal signals, COT release-event statistics and crowding interactions, and weather stress/season interactions. Six groups were fixed before validation:

| Group | Predictors | Purpose |
| --- | ---: | --- |
| `basic` | 20 | Compact price, volume and calendar reference |
| `direction_core` | 58 | Compact direction, COT and weather signals |
| `price_direction` | 99 | Price-family extension |
| `cot_direction` | 104 | Publication-aware positioning extension |
| `weather_direction` | 73 | Weather-family extension |
| `engineered_direction` | 236 | Curated union of the three families |

Each group received logistic regression, XGBoost and random-forest **classifiers**, and Ridge, XGBoost and random-forest **return regressors**. Two parameter configurations per family/task produced **72 configurations per horizon**. Logistic C was 0.01/0.1; Ridge alpha 100/1,000; forests varied depth 5/8 and minimum leaf size 60/30; XGBoost varied depth 2/3, child requirements and regularization. Class weights used mature training labels only. Each simple ensemble equally averaged its three development-selected family members: probabilities for classifiers, returns for regressors.

Direction selection used pooled development **macro-F1**, giving UP and DOWN equal importance; return recipes minimized development RMSE. Thresholds searched 0.35–0.65 in 0.025 steps. Positive realized returns were UP, negative returns DOWN; exact-zero outcomes were excluded uniformly from binary comparisons and retained for regression. A probability equal to its threshold predicted UP. Precision, recall, confusion counts, balanced accuracy and PR-AUC were reported for both classes. No probability calibrator was fitted.

Two additional frozen policies optimized minimum class precision, with both development recalls at least 40%, or minimum class recall. Optional abstention bands required coverage and recall gates; conditional and unconditional recall exposed the cost of withholding calls. These choices used pre-holdout predictions, with no holdout threshold tuning.

A separate [paper-family sweep](../experiments/docs/PAPER_PARAMETER_SEARCH.md) tested **54 configurations per horizon**: seven ARIMA orders with 252/1,260-session histories and a no-drift variant; 5/21/63-lag Ridge autoregression; ELM widths 32/128, tanh/ReLU, penalties, seed/window sensitivities; Holt seasonal periods 0/5/21, damping and history; smoothing and fixed legacy references. ARIMA used custom SciPy conditional sum of squares, not exact statsmodels likelihood. Together, 72 + 54 across two horizons yielded **252 horizon/configuration combinations**, before explicit ensemble comparisons.

No deep network was added: overlapping monthly labels provide only hundreds of broadly distinct training outcomes. The bounded ELM tested a nonlinear neural hypothesis with fixed random hidden weights; it is not deep learning. Larger networks await richer point-in-time data and fresh prospective evaluation.

[Previous: 08 — Monthly model search](08_MONTHLY_MODEL_SEARCH.md) · [Next: 10 — Validation, discards and results](10_VALIDATION_DISCARDS_AND_RESULTS.md)
