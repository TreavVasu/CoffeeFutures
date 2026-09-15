# Modeling Workflow

I built the pipeline to forecast the five-trading-day return of Arabica Coffee C futures from price, COT positioning, regional weather, and an optional delayed-news layer.

## Modeling Objective

I define the regression target as:

```text
target_return_5d = Close[t+5] / Close[t] - 1
```

I use RMSE as the primary selection metric because large return errors matter materially in this market. I report MAE to show typical error and direction accuracy to show how often the predicted sign matches the realized sign.

## Data Assembly

I start from `data/centralData/arabica_ml_model_ready.csv`. I verify the target against the close series, add derived positioning features, and align all inputs to the Coffee C trading calendar.

I apply a three-calendar-day release lag to COT observations before carrying them forward. I use cached Open-Meteo observations for Minas Gerais, Huila, and Dak Lak. For news, I use exact-date daily joins and backward as-of weekly joins after a six-trading-session maturity delay.

I exclude source metadata, archive identifiers, raw target components that would create leakage, and all future-price-derived news scores.

## Chronological Evaluation

I do not randomly shuffle this time series. I use chronological training, validation, and holdout partitions and purge five rows at each boundary. The purge ensures that a training or validation label cannot use a future close from the following partition.

For the final matched news comparison, I use:

| Partition | Dates | Rows |
| --- | --- | ---: |
| Training | 2000-01-03 to 2018-09-04 | 4,674 |
| Validation | 2018-09-12 to 2022-08-29 | 998 |
| Holdout | 2022-09-07 to 2026-09-01 | 1,003 |

I select the learner recipe using no-news validation RMSE. I then fit the no-news, structured-news, and structured-plus-text candidates on the same observations with the same hyperparameters. This prevents the news comparison from benefiting from a different sample or tuning process.

## Final Model Decision

The existing `research_ensemble` remains my production choice because it achieved the best holdout RMSE among the deployable candidates.

| Candidate | Holdout RMSE | Holdout MAE | Direction accuracy |
| --- | ---: | ---: | ---: |
| Price + COT + weather | 5.5055% | 4.3304% | 50.65% |
| Price + COT + weather + delayed news | 5.5111% | 4.3157% | 50.55% |
| `research_ensemble` | **5.3443%** | **4.2291%** | 50.15% |

The news model is retained for analysis, but I do not promote it as the primary forecast. The measured news difference is too small and statistically inconclusive.

## Training

I train the base project model with:

```bash
.venv/bin/python Scripts/project/scripts/train_arabica_model.py
```

I add `--include-xgboost` only when I want to run the slower experimental XGBoost candidates.

I train the corrected matched news comparison with:

```bash
.venv/bin/python Scripts/project/scripts/train_all_inputs_news_model.py
```

The main model artifacts are written under `Scripts/project/artifacts/models/`; metrics, comparisons, and predictions are written under `Scripts/project/artifacts/outputs/`; diagnostic figures are written under `Scripts/project/artifacts/plots/`.

## Prediction

I generate recent base-model predictions with:

```bash
.venv/bin/python Scripts/project/scripts/predict_arabica_returns.py --latest-rows 10
```

For news-aware inspection, I use `ResearchModelTrainingNews.ipynb`. Its holdout replay uses the saved pre-test weights, while latest-refit mode uses the final locally refitted candidate.

## Verification

I run the focused news integration tests with:

```bash
.venv/bin/python -m unittest discover -s Scripts/project/tests -p 'test_news_integration.py' -v
```

I verify feature maturity, exchange-session joins, purged boundaries, exclusion of future-price-derived scores, text matching, model comparability, and saved prediction reproduction.

## Interpretation

I treat the dashboard's news-impact value as a model sensitivity:

```text
all-input predicted return - no-news predicted return
```

I do not interpret it as a causal effect. I also treat the residual-based forecast range as an empirical historical range rather than a guaranteed confidence interval.

My final design rationale is in [../../../Approach.md](../../../Approach.md), and the unresolved risks are in [../../../Limitations.md](../../../Limitations.md).

