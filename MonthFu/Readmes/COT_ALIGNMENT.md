# COT availability and monthly return forecasts

The model uses a daily Coffee C close as its forecast cutoff. A COT position snapshot is usable only after its public release and any identified correction. COT is a weekly positioning report; physical futures delivery follows a different calendar. Changing the return horizon to 21 sessions or 28/30 calendar days does not change when the positions become available.

## Snapshot, publication, and first usable session

The source CSV's `Report_Date_as_MM_DD_YYYY` is the position **snapshot date**, usually Tuesday. CFTC normally publishes at **15:30 America/New_York on Friday**. ICE currently lists Coffee C trading through **13:30 New York time**. Consequently, Friday's new report cannot explain or predict a return from that Friday close using information already known at the close. We wait for the first observed price session **strictly after** the publication day. [CFTC release schedule](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm), [ICE Coffee C specification](https://www.ice.com/products/15/Coffee-C-Futures).

For an ordinary week:

| Forecast origin | Latest newly available snapshot | Treatment |
|---|---|---|
| Tuesday 7 May 2024 close | Earlier report | Positions measured that Tuesday remain unavailable. |
| Friday 10 May 2024 close | Earlier report | The new report is published after the coffee session. |
| Monday 13 May 2024 close | Tuesday 7 May | New positioning and weekly-flow features first enter. |
| Tuesday 14 May through Friday 17 May close | Tuesday 7 May | Carry the published report forward; do not introduce the 14 May snapshot. |
| Monday 20 May 2024 close | Tuesday 14 May | Replace the carried report after the next Friday release. |

Price dates supply the trading calendar. We do not insert weekends, holidays, or unobserved sessions. A missing price session postpones modeled availability. Tests using a synthetic business-day calendar verify this rule, rather than claiming that every weekday is an actual ICE session.

## Historical schedule policy

CFTC says there is no complete historical release-date list. Historical daily availability must therefore be estimated where an explicit announcement is absent. The default takes the later of three federal business days after the snapshot and the report week's Friday, then rolls forward to a federal business day. Friday snapshots conservatively use the following Friday. Mourning closures supplement the federal holiday calendar. The Friday floor prevents unusual Monday snapshots from incorrectly entering Thursday's close. This is a conservative estimate, **not proof of an actual historical timestamp**. [CFTC release FAQ](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm).

Explicit announcements supersede the estimate:

| Interruption | Implemented treatment | Primary source |
|---|---|---|
| October 2013 shutdown | 1 October snapshot: 25 October release. Unverified 8–29 October backlog: withheld through 8 November, the announced return-to-schedule bound. | [CFTC 6745-13](https://www.cftc.gov/PressRoom/PressReleases/6745-13) |
| December 2018–February 2019 shutdown | 24 December starts the backlog on 1 February 2019; subsequent snapshots use the announced Tuesday/Friday catch-up sequence through 5 March. Dates remain scheduled estimates. | [CFTC 7864-19](https://www.cftc.gov/PressRoom/PressReleases/7864-19) |
| 2023 ION disruption | Confirmed releases: 31 January → 24 February; 7 February → 3 March; 14 February → 8 March; 21 February → 10 March; 28 February → 14 March; 7 March → 16 March; 14 March → 21 March. | [CFTC special announcements](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm) |
| 2025 shutdown | Apply the revised schedule for 30 September–23 December snapshots, starting 19 November and completing 29 December. The accelerated schedule replaces the superseded November announcement. | [CFTC 9147-25](https://www.cftc.gov/PressRoom/PressReleases/9147-25) |
| Holiday exceptions | Explicit 2008, 2014, 2020, 2021, January 2025, and published 2026 delayed dates. | [CFTC announcements](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm), [2026 schedule](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm) |

`release_date_status` distinguishes estimates, announced schedules, conservative announced bounds, confirmed releases, and explicit verified overrides. `release_date_estimated` remains true for schedules/bounds; a scheduled date is not presented as a confirmed event. This status is metadata about data quality, not a forward-looking prediction of future release timing.

## Historical corrections and disaggregated backcasts

The archive does not supply original data vintages. Known futures corrections and the non-market-specific 2012 update delay affected snapshots: 18 May 2010 → 27 May; 27 November 2012 → 5 December; 18 September 2018 legacy concentration → 26 September; 26 March 2019 → 3 April. Conservatively, the entire affected family report waits. Options-only corrections do not delay futures-only data. Other unidentified revisions remain a limitation. [CFTC corrections](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm).

Coffee C appears in the first disaggregated report for 1 September 2009, published 4 September. The earlier 2006–2009 history was released retrospectively on 20 October 2009 using backcast classifications. The updated pipeline retains all 168 pre-launch disaggregated rows. They cannot enter 2006–2009 forecast origins before publication; their contribution to trailing history activates on the next observed session, 21 October 2009 in this cache. Live September and early-October positioning keeps its ordinary release timing. Archive activation recalculates windows for the latest released snapshot, so an older backcast cannot replace current positioning. Delayed archive overrides or known corrections postpone its dependent windows. The original benchmark excluded these rows completely; its artifacts remain unchanged. [First Coffee C disaggregated report](https://www.cftc.gov/sites/default/files/files/dea/cotarchives/2009/futures/ag_lf090109.htm), [CFTC launch announcement](https://www.cftc.gov/PressRoom/PressReleases/5710-09), [historical backcast announcement](https://www.cftc.gov/PressRoom/PressReleases/5737-09).

Legacy futures-only reports are included from the cache's first 4 January 2000 snapshot; 2,238 pre-2009 price sessions receive released legacy positioning. The cache contains 2,447 total Coffee C reports. All 118 legacy and 178 disaggregated measurement fields are mapped to predictors, covering All/Old/Other position and change blocks, percentages, trader counts and concentration. Identifiers, source-file names, contract-unit text and fields belonging to the other report schema remain metadata. The expanded bank contains 449 COT predictors; each selected all-COT model member retains their complete schema through fitting and inference. Inclusion permits the model to learn their relevance; it does not force nonzero feature importance.

## Forward fill, lags, and audit

1. Keep legacy and disaggregated futures-only families separate and reject duplicate family/snapshot keys.
2. Compute normalized net/long/short/gross positioning, 1/4/13-report changes, acceleration, trailing z-scores, positioning indices, open interest, concentration, and trader features on ordered **weekly reports**. A four-report change is not a four-day change in repeated rows.
3. Set `raw_value_available_date` to the later of publication and identified revision. Set `feature_publication_date` to the cumulative maximum across preceding snapshots. This conservatively delays a rolling feature when an earlier required report was released out of order.
4. Locate the next observed session strictly after `feature_publication_date`. Backward-asof join on that usable session. If several backlog reports become usable together, the latest snapshot wins; earlier reports still contribute to the trailing weekly calculations.
5. Carry the most recently available report forward. Never backfill pre-release rows. Keep missing families and missing individual fields missing. Training-fold preprocessing handles model imputation later.

The daily features retain snapshot age, release/value age, release lag, dependency delay, estimated-date, identified-correction, new-release-session, and over-14-day staleness flags. Long shutdowns therefore retain old known positions while exposing their age; they do not silently replace them with unpublished snapshots.

The generated `cot_release_audit.csv` preserves snapshot, estimated/announced publication, known correction, value availability, feature availability, and first usable session with source URLs. Reports later than the final price session have `first_usable_session = NaT` and cannot enter daily model rows. Trailing calculations and tests confirm that changing future report values leaves earlier daily features unchanged.

Additional verified dates belong in `MonthFu/data/cot_release_overrides.csv`, which takes precedence over optional project-wide `data/COT/cot_release_overrides.csv`:

```csv
report_date,publication_date,source
2023-01-31,2023-02-24,https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm
```

Each file requires unique nonmissing snapshot dates, nonmissing publication dates/source, and publication on or after the snapshot. An override cannot bypass a known corrected-value availability bound.

## Choosing a monthly horizon

Twenty-eight calendar days cover roughly four weekly report cycles; thirty calendar days follow an ordinary month-length convention; twenty-one observed sessions approximate a trading month. These are distinct return targets and should be compared with their actual target-end dates and matured-label purges. COT's weekly cadence motivates four-report features, but it does not establish which return horizon will predict best. Coffee C delivery months are March, May, July, September, and December; precise delivery/roll features require contract-level expiry and continuous-series roll information beyond this aggregate COT file. [ICE contract specification](https://www.ice.com/products/15/Coffee-C-Futures).
