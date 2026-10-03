# 02 — ELT/ETL data preparation and alignment

The original pipeline stages source files locally before transforming them. Its overall sequence is **extract → load source caches → transform → load prepared tables**. This resembles ELT staging, with ETL transformations between CSV layers; it is implemented through notebooks and scripts, rather than a scheduled warehouse service.

[DataProcessing.ipynb](../../Scripts/notebooks/DataProcessing.ipynb) loads Yahoo OHLCV and filtered COT tables, parses dates, converts numeric fields, sorts observations and deduplicates Yahoo dates. Price transformations include past returns, range and close/open ratios, volume changes, moving averages and price-versus-average ratios. COT transformations include long-minus-short positions, percentages of open interest and reported weekly position changes. Future-price columns created for analysis are labels or diagnostics and must be excluded from predictors.

The notebook full-outer-joins Yahoo trading dates with COT **snapshot/report dates**, preserving unmatched observations. It writes:

- `../DATA/centralData/yahoo_cot_full_outer_by_date.csv`;
- join-status counts and unmatched-date CSVs for inspection;
- the subsequent COT-filled and model-ready tables.

[fill_cot_forward_daily.py](../../Scripts/project/scripts/fill_cot_forward_daily.py) carries COT columns forward for at most ten calendar days after the snapshot date, without filling prices or targets. It adds freshness/status fields and a JSON fill report. This stage alone does not enforce publication availability. The original [arabica_modeling.py](../../Scripts/project/src/arabica_modeling.py) then reconstructs report history, estimates release as report date plus three calendar days, and uses a backward as-of join onto price sessions. That approximation was the supplied foundation, not a complete historical release calendar.

The original model builder joins weather by `Date`, adds five- and twenty-session rolling averages and five-session differences, and creates calendar indicators. [prepare_ml_dataset.py](../../Scripts/project/scripts/prepare_ml_dataset.py) keeps complete OHLCV observations, removes metadata and constant/unusable fields, and drops non-core predictors with more than 20% missingness. Its saved [cleanup report](../../DATA/centralData/arabica_ml_model_ready.report.json) records 6,690 rows, 218 columns and 144 weather features through 2026-09-09. This prepared table's screening used the available history; it should not be described as fold-specific preprocessing.

MonthFu rebuilds monthly features from original price, weekly COT and weather inputs instead of reusing five-day labels. [features.py](../src/features.py) retains valid positive-close sessions, masks inconsistent ancillary OHLC and volume of one or less, and preserves quality flags. It never interpolates prices or backfills from future rows. COT calculations happen weekly before release alignment; weather calculations use calendar days before a conservative lag and backward as-of join. Missing predictors are screened and median-imputed using each training slice, including at subsequent refits. Actual source dates, availability dates and ages are exported for audit.

[Previous: 01 — Source extraction](01_SOURCE_EXTRACTION.md) · [Next: 03 — Early models and leakage review](03_EARLY_MODELS_AND_LEAKAGE_REVIEW.md)
