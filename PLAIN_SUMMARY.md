# Results Summary

**Arabica Coffee C — direction prediction. Results and current status only.**

All figures below are measured on data from **before 2022**, scored on
out-of-sample predictions. See *Status* below for the 2022+ test.

---

## Status

| Item | State |
|---|---|
| Development results (pre-2022), all 4 data stages | **Complete** |
| 2022+ holdout test | **Running** — 324 of 608 quarterly refits done |
| Tests passing | **156 / 156** |

---

## Headline result

| | Accuracy |
|---|---|
| Always-UP baseline (coin flip) | **49.65%** |
| **Model — price + candlestick patterns** | **52.50%** |

**Edge: +2.85 percentage points.**

Across the 8 setups tested in detail, accuracy ranged **52.07% – 53.21%** — the
result is consistent across configurations, not dependent on one.

---

## Accuracy by data source

Each row adds one more data source. A **negative** number means adding that
source made accuracy worse.

| Data sources | Predict 5 days ahead | Predict 7 days ahead |
|---|---:|---:|
| **S0** price + candlestick | **52.82%** | 52.18% |
| **S1** + CFTC positioning | 49.18% | 52.20% |
| **S2** + weather | 48.98% | **52.94%** |
| **S3** + both | 50.32% | 52.14% |

### Change vs the S0 baseline

| Added source | 5 days ahead | 7 days ahead |
|---|---:|---:|
| CFTC positioning | **−3.64 pts** | +0.03 pts |
| Weather | **−3.84 pts** | **+0.76 pts** |
| Both | −2.49 pts | −0.04 pts |

---

## Best configuration

| | |
|---|---|
| **Setup** | `L10_k7` + weather (S2) |
| **History used** | 10 trading sessions |
| **Forecast horizon** | 7 trading sessions ahead |
| **Accuracy** | **53.91%** |
| Macro-F1 | 0.5363 |
| Balanced accuracy | 0.5393 |
| Coverage (share of days a call is made) | 81.98% |

Every other setup tested fell between 49.65% and 53.21%.

---

## Balanced accuracy by data source

Accuracy alone can be inflated by favouring one direction. These figures show
whether both directions are actually being called.

| Data sources | Balanced accuracy | Up recall | Down recall | Coverage |
|---|---:|---:|---:|---:|
| S0 price + candlestick | 0.5262 | 0.641 | 0.412 | 94.1% |
| S1 + positioning | 0.5070 | 0.533 | 0.481 | 81.9% |
| S2 + weather | 0.5097 | 0.539 | 0.480 | 97.3% |
| S3 + both | 0.5130 | 0.572 | 0.454 | 81.7% |

Always-UP reference: balanced accuracy 0.500, up recall 1.000, down recall 0.000.

---

## Statistical significance

**Every comparison returned a confidence interval spanning zero.**

| Comparison | Result | 95% interval excludes zero? |
|---|---|---|
| Model vs always-UP baseline | +2.85 pts | **No** |
| All four data-source stages, 8 setups | mixed | **No** — 0 of 8 |

- The **−3.6 to −3.8 point** setbacks are large enough to be treated as real.
- The **+0.76 point** weather gain at 7 days is **not statistically established**
  and may not reproduce.
- The **+2.85 point** headline edge is directional only; no individual
  configuration is proven above the baseline.

---

## Data quality

| Item | Value |
|---|---|
| Sessions with unusable OHLC (high below body, etc.) | 719 of 6,690 (**10.7%**) |
| Sessions with volume ≤ 1 | 308 |
| Candlestick patterns that fired at least once | **34 of 38** |
| Patterns that never fired | 4 (`mat_hold`, `rising_three_methods`, `bearish_mat_hold`, `ladder_bottom`) |
| Candlestick feature bank chosen as best | **22 of 56** side-selections |

---

## Not measured

| Item | State |
|---|---|
| Trading costs, bid-ask spread, slippage | **Not included** |
| Contract expiry and roll cost | **Not included** |
| Profitability of any strategy | **Not measured** |
| News / GDELT data | Excluded from this work |

Direction accuracy above does not imply a profitable strategy.

---

## Files

| File | Contents |
|---|---|
| `LegacyApproach/MonthFu/experiments/artifacts/polarity/stage_summary.csv` | Accuracy by data source |
| `LegacyApproach/MonthFu/experiments/artifacts/polarity/stage_cell_metrics.csv` | Per-setup detail, all 32 rows |
| `LegacyApproach/MonthFu/experiments/artifacts/polarity/bootstrap_intervals.json` | Confidence intervals |
| `LegacyApproach/MonthFu/experiments/artifacts/polarity/holdout_metrics.csv` | 2022+ results (pending) |
| `LegacyApproach/MonthFu/experiments/artifacts/polarity/selection.json` | Every model choice, frozen pre-holdout |
| `LegacyApproach/MonthFu/experiments/docs/POLARITY_DIRECTION.md` | Method and protocol |
| `LegacyApproach/MonthFu/experiments/docs/CANDLESTICK_PATTERNS.md` | All 38 pattern definitions |
