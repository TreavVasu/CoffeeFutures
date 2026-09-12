# Arabica Futures Research Workspace

Arabica Coffee Futures research workspace for building, evaluating, and running a 5-trading-day return model with Yahoo price data, COT positioning data, weather context, and optional GDELT/news event features.

![Arabica Coffee Futures Model Dashboard](Scripts/project/artifacts/plots/latest_research_paper_return_prediction_console.png)

## Start Here

- `ResearchModelTraining.ipynb` - main notebook for model training, prediction, threshold signal, and dashboard plots.
- `data/centralData/arabica_ml_model_ready.csv` - prepared model-ready dataset used by the notebook.
- `Scripts/project/artifacts/models/research_paper_ensemble.joblib` - saved trained model bundle used for local prediction/dashboard output.
- `Scripts/project/` - reusable Python scripts, source modules, tests, requirements, generated models, outputs, and plots.
- `Scripts/notebooks/` - earlier/supporting notebooks.
- `Scripts/researchpapers/` - research paper references used to document the modeling approach.

## Fresh GitHub Clone: Run Locally

These steps are for someone who only has the GitHub version of this repository.

1. Clone and enter the repo.

```bash
git clone https://github.com/TreavVasu/CoffeeFutures.git
cd CoffeeFutures
```

2. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the project dependencies.

```bash
pip install --upgrade pip
pip install -r Scripts/project/requirements.txt
```

4. Check or rebuild missing local data.

```bash
bash Scripts/fetch_missing_data.sh
```

This verifies the committed Yahoo/COT base file, refreshes weather if needed, forward-fills COT if needed, and rebuilds the prepared model dataset if it is missing.

On Windows, run the same command from Git Bash or WSL.

5. Start Jupyter.

```bash
python -m pip install notebook
jupyter notebook
```

6. Open and run:

```text
ResearchModelTraining.ipynb
```

The notebook can run from the GitHub version using the committed prepared dataset and saved model artifacts. For the cleanest first run, leave `Use local GDELT/news events` disabled if the large local event file is not present.

## One-Command Data Check

The helper script is the easiest way to repair a clone where derived data is missing:

```bash
bash Scripts/fetch_missing_data.sh
```

That script checks the committed base Yahoo/COT file, refreshes weather if needed, forward-fills COT data if needed, and rebuilds `data/centralData/arabica_ml_model_ready.csv` if needed.

Force all refresh/rebuild steps:

```bash
bash Scripts/fetch_missing_data.sh --force
```

To also regenerate the large ignored GDELT/news event file locally, run:

```bash
bash Scripts/fetch_missing_data.sh --with-events
```

The event rebuild can take a long time and creates large ignored files that should stay out of GitHub.

## What Is Included In GitHub

The GitHub version includes the files needed to open the main notebook and run the saved model/dashboard workflow:

- `ResearchModelTraining.ipynb`
- `data/centralData/arabica_ml_model_ready.csv`
- `data/centralData/yahoo_cot_full_outer_by_date.csv`
- `data/centralData/yahoo_cot_full_outer_by_date_cot_ffill.csv`
- `data/weather/open_meteo_coffee_regions_daily.csv`
- `Scripts/project/artifacts/models/research_paper_ensemble.joblib`
- `Scripts/project/artifacts/outputs/research_paper_ensemble_metrics.json`
- `Scripts/project/artifacts/outputs/research_paper_model_comparison.csv`
- `Scripts/project/artifacts/outputs/research_paper_model_cv_summary.csv`
- `Scripts/project/artifacts/outputs/research_paper_model_holdout_predictions.csv`

## What Is Not Included

Large raw/generated event files are intentionally ignored because they exceed GitHub's file-size limits. This mainly affects full raw-data reproduction and event-enriched retraining.

Ignored examples:

- `data/events/gdelt_coffee_events_2000_2026_filtered_scored.csv`
- `data/events/gdelt_coffee_event_candidates_*_weekly_period_dump.csv`
- `Scripts/EventsData/Zip/*.zip`
- `Scripts/project/artifacts/outputs/gdelt_coffee_event_candidates_*.csv`

The notebook handles the missing local GDELT file by running without event features. The saved model and prepared model-ready data are still available for local prediction and dashboard generation.

## Optional: Refresh Or Rebuild Data

Run the full local check/rebuild wrapper:

```bash
bash Scripts/fetch_missing_data.sh
```

Refresh weather cache:

```bash
python Scripts/project/scripts/refresh_weather_cache.py --force-all
```

Forward-fill released COT data:

```bash
python Scripts/project/scripts/fill_cot_forward_daily.py
```

Rebuild the prepared model dataset from committed central data and weather cache:

```bash
python Scripts/project/scripts/prepare_ml_dataset.py
```

Retrain the research ensemble without the large local event file:

```bash
python Scripts/project/scripts/train_research_paper_ensemble.py --skip-events
```

Retrain with local GDELT/news event features only after regenerating or restoring the ignored event file locally:

```bash
python Scripts/project/scripts/train_research_paper_ensemble.py
```

## Latest Dashboard Artifacts

- `Scripts/project/artifacts/outputs/latest_research_paper_return_predictions.csv`
- `Scripts/project/artifacts/plots/latest_research_paper_return_prediction_console.png`

## Notes

- The model target is a 5-trading-day forward Arabica futures return.
- Positive/negative indications are based on the notebook's decision threshold.
- The shaded forecast range in the dashboard is empirical model error from holdout residuals, not a guarantee.
- To fully reproduce the original event-enriched raw pipeline, regenerate the ignored GDELT/news data locally before retraining with events enabled.
