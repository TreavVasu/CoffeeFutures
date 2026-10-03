# COT release-cycle diagnostics

Retrospective diagnostics on 1,148 shared finite holdout origins, 2022-01-03–2026-07-28. Every displayed model is scored on the same rows within each subgroup and horizon. Return endpoints differ by horizon. These diagnostics do not select a horizon, tune the model, or add features.

## Errors on newly usable COT sessions

A release origin is an observed coffee session where at least one COT family first becomes usable under the saved audit. Legacy/disaggregated duplicates count once. It is usually the session after Friday publication; holidays, catch-up releases, and conservative correction bounds can change that. Selected recipes were frozen before holdout. This split is descriptive and does not isolate COT's causal contribution.

RMSE and MAE are percentage points of return; lower is better. The previous-model columns retain the saved prediction names: transferred prior ensemble and its CV-retuned counterpart. Direction is shown for the selected model; zero forecasts are neutral.

| Horizon | Origins | N | Selected RMSE | Zero RMSE | Previous transferred RMSE | Previous CV RMSE | Selected MAE | Selected direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30calendar | All origins | 1148 | 11.149 | 11.321 | 11.708 | 11.728 | 8.231 | 55.0% |
| 30calendar | First usable COT session | 239 | 11.139 | 11.289 | 11.750 | 11.732 | 8.387 | 54.0% |
| 30calendar | Other origins | 909 | 11.152 | 11.329 | 11.697 | 11.727 | 8.189 | 55.2% |
| 28calendar | All origins | 1148 | 10.660 | 10.858 | 11.177 | 10.956 | 7.948 | 56.0% |
| 28calendar | First usable COT session | 239 | 10.911 | 11.090 | 11.451 | 11.211 | 8.207 | 53.6% |
| 28calendar | Other origins | 909 | 10.593 | 10.797 | 11.104 | 10.888 | 7.880 | 56.7% |
| 21calendar | All origins | 1148 | 9.216 | 9.321 | 9.490 | 9.363 | 6.909 | 52.8% |
| 21calendar | First usable COT session | 239 | 9.397 | 9.509 | 9.689 | 9.560 | 7.154 | 55.6% |
| 21calendar | Other origins | 909 | 9.168 | 9.271 | 9.436 | 9.310 | 6.845 | 52.0% |
| 21sessions | All origins | 1148 | 11.584 | 11.176 | 11.566 | 11.584 | 8.724 | 51.5% |
| 21sessions | First usable COT session | 239 | 11.700 | 11.280 | 11.707 | 11.700 | 8.910 | 51.9% |
| 21sessions | Other origins | 909 | 11.553 | 11.148 | 11.529 | 11.553 | 8.675 | 51.4% |
| 30sessions | All origins | 1148 | 13.029 | 12.694 | 13.513 | 13.029 | 10.082 | 42.6% |
| 30sessions | First usable COT session | 239 | 13.199 | 12.842 | 13.708 | 13.199 | 10.395 | 43.5% |
| 30sessions | Other origins | 909 | 12.984 | 12.654 | 13.461 | 12.984 | 10.000 | 42.4% |

## Update events inside the realized return interval

Count unique usable-session update events in (forecast origin, actual target-end date]. The origin's available report is excluded and an event on the endpoint is included. These are retrospective future counts, never model inputs. Delayed backlog releases need not follow a seven-day rhythm.

| Horizon | Mean events | Median | 10th–90th percentile | Event count: number of origins |
| --- | --- | --- | --- | --- |
| 30calendar | 4.39 | 4 | 4–5 | 0: 16; 1: 13; 2: 16; 3: 17; 4: 624; 5: 413; 6: 11; 7: 12; 8: 10; 9: 8; 10: 8 |
| 28calendar | 4.00 | 4 | 4–4 | 0: 18; 1: 17; 2: 14; 3: 27; 4: 998; 5: 32; 6: 8; 7: 17; 8: 3; 9: 13; 10: 1 |
| 21calendar | 3.00 | 3 | 3–3 | 0: 28; 1: 17; 2: 28; 3: 1004; 4: 34; 5: 13; 6: 14; 7: 4; 8: 6 |
| 21sessions | 4.36 | 4 | 4–5 | 0: 17; 1: 13; 2: 16; 3: 16; 4: 659; 5: 377; 6: 11; 7: 16; 8: 5; 9: 7; 10: 11 |
| 30sessions | 6.23 | 6 | 6–7 | 0: 8; 1: 7; 2: 7; 3: 17; 4: 17; 5: 17; 6: 778; 7: 240; 8: 12; 9: 16; 10: 6; 11: 6; 12: 11; 13: 6 |

Twenty-eight calendar days cover four ordinary weekly cycles. Thirty calendar days usually contain four or five update events; twenty-one calendar days cover about three, while twenty-one observed sessions approximate a trading month. The measured distributions explain cadence; they do not demonstrate that any horizon forecasts best. [CFTC release schedule](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm).

Daily monthly returns overlap heavily. Origins within and across the displayed subgroups are dependent, and weekly release timing is confounded with weekday/holiday effects. No confidence interval, significance claim, executable trading advantage, or holdout-based horizon choice follows from these tables. Historical release estimates and unavailable original COT vintages remain limitations; see [COT alignment policy](COT_ALIGNMENT.md).

## Reproduce

```sh
.venv/bin/python MonthFu/scripts/cot_cycle_diagnostics.py
```

Run after all horizon exports finish. The script reads saved predictions/audits only and writes this document. SHA256 input snapshots:

| Horizon | Holdout predictions SHA256 | COT audit SHA256 |
| --- | --- | --- |
| 30calendar | 9bf3fc25935f01409d1081cf7e117d251f80329627c9094c260b1de526a53e05 | 5c0cf2ced2bcf6a83c5a1ce0809e7e3a74fc6f9add8cebb44c0ec133e6e2750c |
| 28calendar | d939595a767742062e3077b6a9548cc18a982f8c072bf03229ed2c09ca875534 | 5c0cf2ced2bcf6a83c5a1ce0809e7e3a74fc6f9add8cebb44c0ec133e6e2750c |
| 21calendar | f030eaa37c3b7708d04f0944895b8ffe62201af7ffc1893f685fc46ee4acc75b | 5c0cf2ced2bcf6a83c5a1ce0809e7e3a74fc6f9add8cebb44c0ec133e6e2750c |
| 21sessions | 0880bbb6a8b7a2c0be5fb72f388270952881b7ef438472e90bb65366e7c974bd | 5c0cf2ced2bcf6a83c5a1ce0809e7e3a74fc6f9add8cebb44c0ec133e6e2750c |
| 30sessions | 0398525e9ca612d674d8bbdab619a72316be29671a8d6287aeec63103605dbbd | 5c0cf2ced2bcf6a83c5a1ce0809e7e3a74fc6f9add8cebb44c0ec133e6e2750c |
