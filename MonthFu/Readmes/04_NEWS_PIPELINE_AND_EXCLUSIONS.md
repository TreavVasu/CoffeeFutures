# 04 — News pipeline, leakage correction and exclusions

The repository's news layer comes from **structured GDELT event records and constructed digests**, not a complete archive of original article text. [gdelt_stream_all_years_coffee_events.py](../../Scripts/project/scripts/gdelt_stream_all_years_coffee_events.py) processes historical ZIPs in 250,000-row tab-separated chunks. It filters coffee keywords, producing-country geography and region/severe-supply-risk context, then forward-maps event dates to the next observed Coffee C session within five calendar days.

The upstream pipeline attaches price and COT context, computes relevance/intensity/direction/impact scores, ranks records and retains up to fifty events per mapped day per archive by default. It deduplicates `GLOBALEVENTID`, appends the filtered scored CSV, rebuilds daily and Sunday-ending weekly summaries, deletes temporary ZIPs and records resumable processing status. Large raw outputs are intentionally excluded from the committed footprint. The supplied [all-year processing manifest](../../Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_processing_manifest.json) records 4,978 completed archives, 781,957 retained rows and 22 failures: this is a partial selected corpus.

Optional [Ollama scoring](../../Scripts/project/scripts/gdelt_weekly_ollama_batch_score.py) was implemented and used in an earlier diagnostic experiment. The saved 2000 weekly metadata records eleven scored periods with `llama3.2:latest`; scoring is not merely a proposal. Those scores are excluded from the corrected forecast inputs.

The critical problem was upstream ranking by **future five-session returns and observed direction alignment**. Impact scores therefore encoded later outcomes, and the earlier **61.35% direction result was withdrawn**. The corrected predictive layer excludes future returns, price-response, direction-alignment, rule/final-impact, deterministic and LLM/Ollama scores.

Its retained nine inputs use a conservative six-session delay:

- Daily counts are reindexed to observed sessions. `log1p(event_count)` and bullish-minus-bearish share are shifted six sessions, then averaged over five sessions and joined by exact date.
- Weekly descriptors become available six sessions after the source week's final observed session, then join backward as-of. They describe count, direction balance/concentration, explicit-coffee share and non-price intensity.
- Fixed word dictionaries count disruption and coffee terms in stored digests. A constant support-term indicator was removed. These are lexical proxies, not learned article sentiment.

[The matched corrected comparison](../../Scripts/project/docs/NEWS_EVENT_TRANSFORMATION.md) selected histogram boosting on no-news validation, then held learner settings, rows and splits fixed. Price/COT/weather achieved **5.5055% RMSE and 50.65% direction**, versus **5.5111% and 50.55%** with delayed news. News was retained for context and sensitivity; the all-input minus no-news prediction difference is not causal attribution.

Six sessions allow the upstream outcome window to mature, but cannot undo the retained-universe bias or full-history normalization. MonthFu therefore keeps news optional and exploratory. A stronger test needs unfiltered publication-time records, deduplication rules fixed before examining returns, preserved vintages and a future untouched evaluation. The current result does not establish that real-time news lacks value.

[Previous: 03 — Early models and leakage review](03_EARLY_MODELS_AND_LEAKAGE_REVIEW.md) · [Next: 05 — Monthly targets and horizons](05_MONTHLY_TARGETS_AND_HORIZONS.md)
