# 08 — Monthly model search

The monthly task predicts a future simple return, rather than a persistent future price level. The supplied papers motivated lagged returns, positioning extremes, weather/season interactions, autoregression and regularized nonlinear models. Their regional spot-price fits and price-level accuracy are not expected monthly futures-return performance.

Ideas requiring unavailable spot prices, FX, individual contracts, physical delivery/default outcomes or counterparties were not fabricated. GARCH describes conditional volatility and was not treated as a standalone return-direction forecast. The transfer and limitations of the seven supplied papers are recorded in [RESEARCH_REVIEW.md](RESEARCH_REVIEW.md).

The first MonthFu search evaluated **46 configurations per default horizon**, including zero/historical-mean baselines, adapted older references, Ridge, Elastic Net, histogram gradient boosting, Extra Trees, XGBoost and regularized ELM. Fixed feature groups ranged from compact references and the 104-column core to the full 733-column bank.

The bounded optimization varied:

- Ridge penalties of 100, 1,000 and 10,000 across source groups.
- Tree/boosting regularization and leaf/child requirements.
- Recent five-/ten-year supervised history and exponential time weighting.
- Return targets scaled by origin-known trailing volatility.
- ELM source groups and output regularization.
- Individual forecasts, limited equal-weight blends, and shrinkage factors 0.25/0.50/0.75/1 toward zero.

Random internal validation splits were disabled. Feature screening and preprocessing were fitted on each mature training fold. Candidate and recipe selection used pooled pre-2022 expanding-validation **RMSE**; the recipe was saved before quarterly holdout evaluation. It was a bounded search, not an exhaustive search over all possible parameters. Broad ARIMA order tuning was absent from this monthly stage and was added in the later paper sweep.

The original monthly selected recipe was:

```text
predicted return = 0.5 × price-only Extra Trees prediction
```

It retained 245 price predictors. The larger COT/weather banks and interactions were tested but did not win pre-holdout selection. Shrinkage makes the return forecast more conservative without changing the sign of a nonzero prediction. Those unselected groups remain available for ablations; their failure to win does not establish that their sources are universally useless.

References adapt the earlier RF/Holt–Winters/AR method to the same monthly target. One preserves its five-session validation weights; another retunes its components on monthly development folds. RF and AR are retrained for the longer return, and Holt–Winters forecasts the longer horizon. Neither is described as applying the exact saved five-session model directly to a month. Zero, expanding historical mean, and a compact older-style model are also reported on matched origins.

Full 30-calendar-day holdout RMSE was approximately 11.1114%, versus 11.6820% for the transferred older method and 11.7009% for the CV-retuned older components. Cross-horizon summaries use a smaller common-date intersection and have slightly different figures; those row scopes must stay distinct. The advantage over zero return remained uncertain.

Implementation and measured evidence: [modeling.py](../src/modeling.py), [train.py](../scripts/train.py), [the primary report](../artifacts/30calendar/report.md), and [horizon results](../RESULTS.md).

[Previous: 06 — Features](06_FEATURE_ENGINEERING.md) · [Next: 08 — Direction experiments](08_DIRECTION_AND_PARAMETER_EXPERIMENTS.md)
