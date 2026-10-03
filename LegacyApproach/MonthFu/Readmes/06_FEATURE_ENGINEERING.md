# 07 — Feature engineering

Feature definitions use only information known at the forecast origin. Targets are attached after predictor definitions and fixed groups exist. Rolling calculations are trailing; initial history stays missing. Price windows count observed sessions, weather windows generally count calendar days, and COT windows count source reports. Those units are not silently interchangeable.

The original MonthFu bank contains:

| Component | Predictors | Examples |
|---|---:|---|
| Price, volume and calendar | 245 | Momentum, moving-average distance, drawdown, trend efficiency, volatility ratios, gaps, range volatility, signed/relative volume, seasonality |
| COT | 150 | Net/gross positions normalized by open interest, weekly flows, acceleration, crowding, concentration, participant breadth, release ages |
| Weather | 281 | Rain, temperature, soil moisture, water deficit, dry/heat/cold stress, past-year seasonal anomalies |
| Fixed interactions | 57 | Positioning × momentum, crowding × reversal, trend × volatility, weather × crop-season indicators |
| Full bank | **733** | All four components |

Source ablations contain 245 price predictors, 395 price+COT, and 676 price+COT+weather. The curated monthly core contains 104. The optional experimental-news group and its 37 predictors were removed with the GDELT layer. Raw OHLCV remains available for label construction and older-method reference groups; most monthly features use relative or normalized values.

The later direction experiments add **140 definitions**: 56 price, 52 COT and 32 weather. Price additions include signed streaks, exhaustion, prior-extrema breakout distance, positive/negative semivariance balance, momentum ranks, reversal persistence, and volume/direction divergence. COT additions include released-flow ranks/streaks, exponential freshness, price/flow divergence and crowding reversal. Weather additions include compound drought/heat severity, excess rain, stress changes and season/reversal interactions.

These additions are tested in a curated bank of **236**, rather than appending every new column to the old 733. Six fixed groups contain 20 basic, 58 direction-core, 99 price-direction, 104 COT-direction, 73 weather-direction, and 236 full directional predictors. Each family ablation includes the same 20 basic predictors. COT event statistics count newly usable release states; repeated weekdays do not create artificial weekly samples.

In the earlier unconstrained runs, usable columns required **at least 20% observed training coverage** and more than one distinct observed value. This differs from the earlier prepared-table global rule that dropped columns with more than 20% missingness. Medians, missingness indicators and scaling are learned exclusively on mature training rows. Full-cache missing fractions are audit metadata and do not choose columns. No full-history target correlation ranks the new groups.

Missing current source components remain missing; stale weather cannot be revived by an old rank. Identifiers and future target fields are excluded, and the news layer no longer exists. A feature can be valid yet unhelpful, so larger banks receive no automatic preference.

The October 1 all-COT update expands the COT bank to **449** predictors covering all **296** source measurement fields across the two families. This brings the full bank to **1,032** and the core with all COT to **521**. All required COT columns remain in each selected fit even when sparse, constant or not yet available. Fold-local imputation and missing indicators handle that state without writing invented values into historical source rows. See [the COT inclusion update](10_COT_INCLUSION_UPDATE.md) and [interview.md](interview.md).

Definitions and causality checks are in [FEATURES.md](FEATURES.md), [features.py](../src/features.py), and [the directional feature guide](../experiments/docs/FEATURE_ENGINEERING.md). Prefix rebuild and future-input mutation tests verify that later data cannot change earlier predictor rows.

[Previous: 05 — Availability](05_AVAILABILITY_AND_DATA_QUALITY.md) · [Next: 07 — Model search](07_MONTHLY_MODEL_SEARCH.md)
