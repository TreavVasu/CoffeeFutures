# Arabica Futures Research Workspace

I built this repository to research and evaluate five-trading-day return forecasts for Arabica Coffee C futures. My workflow combines Yahoo Finance price history, CFTC Commitments of Traders positioning, coffee-region weather, and a corrected GDELT-derived news layer.

![My Arabica Coffee C dashboard with news, weather, COT positioning, and return forecasts](Scripts/project/artifacts/plots/latest_arabica_news_prediction_console.png)

## Final Research Position

My selected deployable model is the existing `research_ensemble`. I keep the news-integrated model as a companion experiment and dashboard diagnostic because delayed news did not improve the matched holdout result.

| Candidate | Holdout direction accuracy | Five-day return RMSE |
| --- | ---: | ---: |
| Price + COT + weather | 50.65% | 5.5055% |
| Same model with delayed news and text | 50.55% | 5.5111% |
| Selected `research_ensemble` | 50.15% | **5.3443%** |

I evaluated the matched news comparison on 1,003 dates from 2022-09-07 through 2026-09-01. I withdrew the earlier 61.35% direction result because its event scores included future-price information. In the corrected experiment, I exclude future-price-derived scores and delay the remaining news summaries by six trading sessions.

My conclusion is deliberately narrow: the stored, retrospectively selected news summaries did not improve this forecasting setup. This does not test a complete real-time article feed.

- [Summary.MD](Summary.MD) records my final result and decision.
- [Approach.md](Approach.md) explains the final news-aware methodology.
- [Limitations.md](Limitations.md) defines the boundaries of the evidence.
- [NEWS_EVENT_TRANSFORMATION.md](Scripts/project/docs/NEWS_EVENT_TRANSFORMATION.md) documents the exact news transformations and joins.

## Main Files

- [ResearchModelTrainingNews.ipynb](ResearchModelTrainingNews.ipynb) is my news-integrated notebook with input, prediction, dashboard, news-impact, and context views.
- `ResearchModelTraining.ipynb` is my original ensemble notebook, preserved for reproducibility.
- `data/centralData/arabica_ml_model_ready.csv` is the prepared modeling table.
- `Scripts/project/artifacts/models/research_paper_ensemble.joblib` is the selected model bundle.
- `Scripts/project/artifacts/models/arabica_all_inputs_news_model.joblib` is the corrected news experiment bundle.
- `Scripts/project/` contains my reusable scripts, tests, generated artifacts, and technical documentation.

## Notebook Workflow

With the project environment active, I open the companion notebook with:

```bash
python -m notebook ResearchModelTrainingNews.ipynb
```

The default `RETRAIN = False` loads my saved bundle. I set it to `True` only when I want to repeat training from the local prepared data.

The notebook supports two forecast modes:

- `Latest refit` uses the model refitted on all locally available labeled data.
- `Holdout replay` uses the saved pre-test weights and reproduces the committed holdout predictions.

I calculate the dashboard's news impact as the difference between separately fitted all-input and no-news models. I use it as a sensitivity diagnostic, not causal attribution. Forecast dates beyond the stored exchange calendar are weekday estimates, and the displayed uncertainty range is based on empirical holdout residuals.

To execute every notebook cell and verify its saved views, I run:

```bash
python Scripts/project/scripts/check_news_notebook.py
```

This check exercises all four view/model combinations, verifies the holdout replay against the saved CSV, and confirms that the dashboard images are nonblank.

## Run From A Fresh Clone

```bash
git clone https://github.com/TreavVasu/CoffeeFutures.git
cd CoffeeFutures
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r Scripts/project/requirements.txt
bash Scripts/fetch_missing_data.sh
jupyter notebook
```

I then open `ResearchModelTrainingNews.ipynb` and run all cells. The committed prepared data, summaries, and model bundles are sufficient for the saved model and dashboard workflow.

On Windows, I run the shell command from Git Bash or WSL.

## Data Checks And Rebuilds

For a normal local integrity check, I use:

```bash
bash Scripts/fetch_missing_data.sh
```

This verifies the committed Yahoo/COT base file, refreshes weather when required, forward-fills released COT values, and rebuilds the prepared model table if it is absent.

To force every supported refresh and rebuild:

```bash
bash Scripts/fetch_missing_data.sh --force
```

To regenerate the large ignored GDELT event file as well:

```bash
bash Scripts/fetch_missing_data.sh --with-events
```

The event rebuild is slow and produces files that exceed the intended GitHub footprint.

I can also run the stages individually:

```bash
python Scripts/project/scripts/refresh_weather_cache.py --force-all
python Scripts/project/scripts/fill_cot_forward_daily.py
python Scripts/project/scripts/prepare_ml_dataset.py
python Scripts/project/scripts/train_research_paper_ensemble.py --skip-events
python Scripts/project/scripts/train_all_inputs_news_model.py
```

## Files Included In The Repository

I commit the inputs and artifacts required to open the notebooks, replay predictions, and inspect the final evaluation:

- both research notebooks and their saved output states;
- the prepared Yahoo/COT/weather model table;
- daily and weekly news summaries;
- the production and news-experiment model bundles;
- model metrics, comparison tables, holdout predictions, and dashboard images.

The principal committed artifacts are:

```text
data/centralData/arabica_ml_model_ready.csv
data/centralData/yahoo_cot_full_outer_by_date.csv
data/centralData/yahoo_cot_full_outer_by_date_cot_ffill.csv
data/weather/open_meteo_coffee_regions_daily.csv
Scripts/project/artifacts/models/research_paper_ensemble.joblib
Scripts/project/artifacts/models/arabica_all_inputs_news_model.joblib
Scripts/project/artifacts/outputs/arabica_all_inputs_news_metrics.json
Scripts/project/artifacts/outputs/arabica_all_inputs_news_holdout_predictions.csv
Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_daily_summary.csv
Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_weekly_summary.csv
```

## Files I Intentionally Exclude

I do not commit the largest raw or generated event files:

```text
data/events/gdelt_coffee_events_2000_2026_filtered_scored.csv
data/events/gdelt_coffee_event_candidates_*_weekly_period_dump.csv
Scripts/EventsData/Zip/*.zip
Scripts/project/artifacts/outputs/gdelt_coffee_event_candidates_*.csv
```

These exclusions affect full raw-event reconstruction, but they do not prevent the saved notebook, model bundle, or dashboard from running. To repeat the original row-level event extraction, I regenerate the files locally with `--with-events` before retraining.

## Verification

I run the focused news tests with:

```bash
.venv/bin/python -m unittest discover -s Scripts/project/tests -p 'test_news_integration.py' -v
```

The suite checks news maturity, exchange-session alignment, boundary purging, exclusion of future-price-derived scores, text matching, baseline comparability, and repository path behavior.
