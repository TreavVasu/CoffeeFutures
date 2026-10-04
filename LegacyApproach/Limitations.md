# Limitations

I document these limitations explicitly because they determine what I can and cannot conclude from the reported metrics.

## News Data Is Not Point-In-Time Complete

The committed news data consists of retrospectively selected daily and weekly GDELT summaries. I do not have the complete unfiltered event history or original publication-time snapshots in the repository. The upstream selection process used five-session future Coffee C returns, so I delay all retained news aggregates by six trading sessions. This reduces leakage risk, but it changes the research question from immediate news prediction to delayed selected-news evaluation.

I therefore cannot conclude that real-time news sentiment has no value. I can only conclude that the available delayed summaries did not improve this model on the reported holdout.

## The Earlier News Result Was Invalid

The earlier 61.35% direction accuracy included future-price-derived information. I withdrew that result. I exclude price-response, direction-alignment, rule-impact, final-impact, future-return, deterministic-period, and LLM/Ollama score fields from the corrected predictive model.

## Holdout Reuse Makes The Result Exploratory

The holdout period has been inspected during previous experiments. Although I preserve chronological splits, purge overlapping labels, and keep test labels out of model fitting, repeated human review can still influence research decisions. I treat the reported holdout as an exploratory benchmark rather than a pristine final estimate.

## Five-Day Targets Overlap

Consecutive five-session returns share future price observations. Standard independent-sample confidence assumptions are therefore inappropriate. I use a 20-session moving-block bootstrap for the news comparison, but the interval remains dependent on block length and this single historical sample.

## Source Timing Is Not Fully Certified

I explicitly correct COT release timing and news availability in the implemented workflow. However, the prepared weather and COT histories have not been independently reconstructed from source-vintage snapshots. Revisions, reporting changes, publication delays, and backfilled observations may remain.

## Weather Coverage Is Partial

The weather layer represents selected regions in Minas Gerais, Huila, and Dak Lak. It does not cover every Arabica-growing area, farm-level exposure, soil moisture condition, crop stage, irrigation practice, or logistics disruption. I treat it as a regional proxy, not a complete supply model.

## Futures Data Has Market-Structure Limitations

The model uses a continuous Coffee C history. Contract rolls, curve shape, basis behavior, limit moves, liquidity differences, and execution slippage are not modeled explicitly. A return forecast from this dataset is not automatically equivalent to the return of a tradeable strategy.

## Direction Accuracy Is Near Chance

The final candidates produce direction accuracy close to 50% on the main holdout. Small differences between candidates are not practically convincing, and the bootstrap interval for the news-related change spans zero. I do not present the model as a reliable standalone directional trading signal.

## Forecast Intervals Are Empirical

The dashboard range is derived from historical holdout residuals. It is not a calibrated probability guarantee and may understate risk during structural breaks, extreme weather, policy shocks, or market dislocations.

## Feature Importance Is Not Causality

Tree usage counts and model sensitivities show that a feature participated in a fitted prediction. They do not establish economic causality, stable importance, or incremental value. I require matched out-of-sample comparisons before claiming that a source improves the model.

## No Trading-Cost Evaluation

I have not incorporated bid-ask spreads, brokerage fees, margin requirements, contract sizing, turnover, roll costs, or market impact. The reported RMSE, MAE, and direction accuracy are forecasting metrics, not evidence of profitable execution.

## Latest Forecast Is Not Validation Evidence

The saved latest prediction is an operational output from the refitted model. Its future target is not observed in the supplied data, so I do not use it as evidence of accuracy or model improvement.

## Deployment Position

Given these constraints, I use the `research_ensemble` as the current deployable research model and keep news as a transparent contextual and sensitivity layer. Before promoting news into the primary forecast, I would rebuild a complete point-in-time corpus, preserve publication timestamps, define article deduplication before looking at returns, and evaluate the frozen pipeline on a genuinely untouched future period.
