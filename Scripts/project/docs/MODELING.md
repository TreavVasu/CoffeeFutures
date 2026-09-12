# Arabica Futures Return Model

This project trains a 5-trading-day Arabica Coffee C futures return model from:

- Yahoo OHLCV price history in `data/centralData/yahoo_cot_full_outer_by_date.csv`
- CFTC COT positioning data from the same central file
- Optional Open-Meteo historical weather features for Brazil Minas Gerais, Colombia Huila, and Vietnam Dak Lak

The training pipeline uses a chronological train/validation/test split. COT features are shifted by a 3-calendar-day release lag before they are carried forward, so the model does not use report data before it would have been available.

Source/archive/code/date metadata and raw price level columns are intentionally excluded from the feature set. The model keeps market-state features such as returns, volatility, relative volume, normalized price-vs-average measures, COT positioning, COT percent-of-open-interest, COT trader counts, concentration, and weather features.

The end-to-end workflow is documented in `notebooks/Arabica_Futures_Model_Workflow.ipynb`.

For spreadsheet-style daily analysis, a forward-filled COT copy is available at `data/centralData/yahoo_cot_full_outer_by_date_cot_ffill.csv`. It carries the latest COT report values into the daily Yahoo rows between reports while leaving Yahoo price, return, and future-target columns unchanged. Rebuild it with:

```bash
.venv/bin/python Scripts/project/scripts/fill_cot_forward_daily.py
```

## Train

```bash
.venv/bin/python Scripts/project/scripts/train_arabica_model.py
```

To run the slower experimental XGBoost candidates, add `--include-xgboost`.

Outputs:

- `Scripts/project/artifacts/models/arabica_returns_model.joblib`
- `Scripts/project/artifacts/outputs/arabica_model_metrics.json`
- `Scripts/project/artifacts/outputs/arabica_holdout_predictions.csv`
- `Scripts/project/artifacts/plots/regression_metrics.png`
- `Scripts/project/artifacts/plots/classification_accuracy_metrics.png`
- `Scripts/project/artifacts/plots/roc_curve.png`
- `Scripts/project/artifacts/plots/actual_vs_predicted_returns.png`
- `Scripts/project/artifacts/plots/real_vs_predicted_price.png`
- `Scripts/project/artifacts/plots/feature_importance.png`
- `Scripts/project/artifacts/plots/prediction_error_drift.png`
- `Scripts/project/artifacts/plots/rolling_direction_accuracy.png`
- `Scripts/project/artifacts/plots/feature_drift_top.png`
- `Scripts/project/artifacts/plots/cot_target_correlation.png`
- `Scripts/project/artifacts/plots/cot_error_correlation.png`
- `Scripts/project/artifacts/plots/cot_drift_vs_error_correlation.png`
- `Scripts/project/artifacts/outputs/feature_drift_cot_correlation_report.csv`
- `Scripts/project/artifacts/outputs/worst_error_dates_with_drift_context.csv`
- `Scripts/project/artifacts/outputs/news_context_by_error_date.csv`
- `Scripts/project/artifacts/outputs/news_articles_by_error_date.csv`
- `Scripts/project/artifacts/plots/news_impact_vs_return.png`
- `Scripts/project/artifacts/plots/news_strength_vs_model_error.png`
- `Scripts/project/artifacts/plots/news_article_count_by_error_date.png`

## Predict

```bash
.venv/bin/python Scripts/project/scripts/predict_arabica_returns.py --latest-rows 10
```

This loads the saved model and writes `Scripts/project/artifacts/outputs/latest_arabica_predictions.csv`.

## Notes

Open-Meteo may rate-limit long historical pulls. The trainer caches successful weather data in `data/weather/` and records any region-level fetch errors in the metadata JSON. The default model uses the cached weather when present.

To retry missing weather regions without overwriting good cached data:

```bash
.venv/bin/python Scripts/project/scripts/refresh_weather_cache.py
```

The refresh utility saves per-region files and backs up the combined cache before writing an updated `data/weather/open_meteo_coffee_regions_daily.csv`.

The news context layer is intentionally separate from the base model. Run it with:

```bash
.venv/bin/python Scripts/project/scripts/news_context_analysis.py --top-n 25 --max-articles 8
```

It uses GDELT first and DuckDuckGo HTML fallback, then scores whether dated news looked bullish/bearish enough to plausibly affect the next week of Arabica prices. Ollama is optional and only used when `--use-ollama` is passed.
