# 06 — Availability and data quality

MonthFu rebuilds predictors from the original source caches instead of extending the earlier filled modeling table. Its daily forecast clock is the observed Coffee C session close. Same-session close, OHLC and volume are allowed under that assumption; an application making a decision before the close would need to lag them again.

Positive finite closes define the unique, increasing market calendar. Weekends, holidays and absent source sessions are not inserted. In the supplied cache, 6,690 observed rows span 2000-01-03 through 2026-09-09. Inconsistent OHLC fields are masked on 719 rows; volume of one or less is masked on 308 rows. Quality flags remain. Those masks do not discard an otherwise usable close or manufacture replacement prices.

| Source | Implemented availability rule | Daily alignment |
|---|---|---|
| Prices | At that observed session's close | Current/past inputs and trailing windows only |
| COT | First observed session strictly after publication or correction availability | Backward as-of join of released report state |
| Weather | Observation date + five calendar days | Backward as-of after daily-calendar rolling |

The former optional news row was removed with the GDELT layer.

For COT, the snapshot date, publication date and first usable session are separate. A normal Tuesday snapshot is modeled as published Friday after the Coffee C close, so the next observed session, generally Monday, first receives it. Holidays, shutdown catch-up schedules, announced delays, corrections and dependency delays can postpone availability. Ordinary historical publication dates remain marked estimates unless verified by announcements or overrides.

Positioning changes and extremes are computed on ordered weekly reports **before** daily alignment. A four-report change compares four reports, not four repeated daily values. Legacy and disaggregated definitions remain separate. The updated all-COT policy retains disaggregated backcasts and activates their trailing-history contribution only after the October 20, 2009 archive release; current pre-archive live reports keep their normal timing. Pre-2009 legacy reports already belong in historical predictors after publication. A carried report is known information between releases; a future or unreleased report is not filled backward. Age, freshness, archive-activation and staleness fields expose the information state.

Weather rolls on a complete daily calendar while missing observations remain missing. A source becomes usable five calendar days later, then expires after seven additional days without a fresh observation. Seasonal anomalies use previous occurrences of the same month, excluding the entire current month. The cache is historical reanalysis: the lag is an availability proxy and cannot remove later revisions already present in historical values.

Unknown predictor values remain unknown until train-fitted preprocessing. There is no general backward fill, price interpolation or full-history imputation. The retrospective news layer was removed, so no source decision depends on delaying a selected news universe.

See [COT_ALIGNMENT.md](COT_ALIGNMENT.md), [FEATURES.md](FEATURES.md), [cot_release_features.py](../src/cot_release_features.py), and [external_return_features.py](../src/external_return_features.py) for exact policies and audits. Original COT and weather vintages are not certified by these rules.

[Previous: 04 — Targets](04_MONTHLY_TARGETS_AND_HORIZONS.md) · [Next: 06 — Features](06_FEATURE_ENGINEERING.md)
