# Independent MonthFu all-COT artifact audit

This audit independently reconstructs monthly labels from the raw price calendar and recalculates saved scores. It checks split/refit maturity and replays the latest forecast from the saved final bundle. Passing these checks establishes artifact consistency and tested timing behavior; it does not certify every historical source vintage or prove future forecasting skill.

Audit status: **passed**. Completed experiments checked: 5. Pending experiments: none.

| Experiment | Holdout rows | RMSE | Skill vs zero | RMSE improvement vs transferred old recipe | RMSE improvement vs monthly-retuned old components | Latest replay error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30calendar | 1157 | 11.254% | +0.51% | +3.67% | +3.82% | below 1e-10 |
| 21sessions | 1157 | 11.116% | +0.43% | +3.68% | +3.83% | below 1e-10 |
| 28calendar | 1159 | 10.785% | +0.63% | +3.24% | +1.21% | below 1e-10 |
| 21calendar | 1164 | 9.275% | +0.99% | +2.15% | +0.86% | below 1e-10 |
| 30sessions | 1148 | 12.614% | +1.26% | +6.66% | +3.19% | below 1e-10 |

## Interpretation of the measured primary result

The primary winner is `{'engineered_extra': 0.25}`, using feature groups `['engineered']`.

The all-COT policy retains all 449 numeric COT predictors in each eligible fit and final member. Its source audit contains all 2447 supported Coffee C reports, including 168 prelaunch disaggregated backcasts. The saved final training matrix has 6669 mature-label rows and the inference matrix has 6690 observed sessions. Backcasts first enter historical windows after their October 20, 2009 public release; legacy reports retain their earlier availability. Predictor inclusion does not require a model to assign nonzero importance.

On all 1157 primary holdout origins, selected-minus-retuned-previous RMSE has a 60-session block interval [-0.863, -0.070] percentage points. The corresponding interval versus zero is [-0.144, +0.112] percentage points, which includes zero. Negative intervals favor the selected recipe; an interval containing zero leaves the improvement uncertain.

Primary direction accuracy is 50.82%, compared with 48.14% for the training-only majority benchmark. The empirical nominal 80% band covers 75.80% of primary holdout returns and falls below nominal coverage. Latest origin is 2026-09-09, which is the cache date rather than today's market date.

Lower absolute error at a shorter horizon cannot by itself identify a better forecast. The common-origin horizon table can use a smaller shared sample than the full per-horizon table above; do not mix those scopes when reporting gains.

## Verification scope

Checks performed for every completed experiment:

- Raw-price origin, target close, target date and return reconstruction for CV and holdout rows.
- Scalar JSON/CSV metrics, every candidate/fold score, CV ranking and frozen recipe weights.
- Exact training rows and label endpoints before every CV/refit boundary; mature historical means and direction benchmarks.
- Year-specific and true-date non-overlapping metrics, empirical error-band construction and observed coverage.
- Paired 60-session moving-block RMSE intervals independently reconstructed, with 30/90-session zero-reference sensitivity.
- COT publication/first-usable-session chronology, raw input hashes and unchanged core training/inference source hashes.
- Full availability-row calendar, nonnegative source ages, final bundle feature membership and imputation medians from mature fitting rows.
- Latest final-bundle prediction replay; historical holdout scores use saved out-of-sample rows rather than final refitted weights.

Cross-horizon comparison audit: `{'status': 'passed', 'common_origin_rows': 1148}`.

The 30-calendar-day target remains the fixed primary experiment. Alternatives are sensitivity studies. The historical holdout was reused in earlier project research; daily monthly labels overlap, and current news retention/reanalysis/COT release-estimate limitations remain. Prior baselines adapt old methods and feature families rather than reproduce the original saved five-day model. Read moving-block intervals and yearly results before calling a point gain reliable.

Re-run from the repository root:

```sh
.venv/bin/python MonthFu/scripts/audit_artifacts.py --artifacts-dir MonthFu/artifacts/all_cot
```
