from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "project" / "src"))

from arabica_modeling import (  # noqa: E402
    build_modeling_frame,
    feature_frame_for_model,
    latest_prediction_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Arabica Coffee C futures 5D return.")
    parser.add_argument("--central-data", default="data/centralData/yahoo_cot_full_outer_by_date.csv")
    parser.add_argument("--weather-cache", default="data/weather/open_meteo_coffee_regions_daily.csv")
    parser.add_argument("--model", default="project/artifacts/models/arabica_returns_model.joblib")
    parser.add_argument("--output", default="project/artifacts/outputs/latest_arabica_predictions.csv")
    parser.add_argument("--latest-rows", type=int, default=10)
    parser.add_argument("--no-weather", action="store_true", help="Do not merge cached weather features.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ROOT
    model_path = root / args.model
    central_path = root / args.central_data
    weather_path = None if args.no_weather else root / args.weather_cache
    if weather_path is not None and not weather_path.exists():
        weather_path = None

    bundle = joblib.load(model_path)
    feature_set = build_modeling_frame(central_path, weather_path=weather_path, prediction_mode=True)
    rows = latest_prediction_rows(feature_set, n_rows=args.latest_rows).replace([np.inf, -np.inf], np.nan)

    X = feature_frame_for_model(rows, bundle["numeric_features"])
    pred_return = bundle["regressor"].predict(X)
    pred_prob = bundle["classifier"].predict_proba(X)[:, 1]

    out = rows[["Date", "Close"]].copy()
    out["predicted_return_5d"] = pred_return
    out["predicted_direction_probability"] = pred_prob
    out["predicted_direction"] = pred_prob >= 0.5
    out["predicted_close_5d"] = out["Close"] * (1.0 + out["predicted_return_5d"])
    out["model_training_end_date"] = bundle.get("training_end_date")

    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(out.to_string(index=False))
    print(f"Saved predictions: {output_path}")


if __name__ == "__main__":
    main()
