"""Rebuild experiment predictors for cached-origin forecasts or replay saved OOS rows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "MonthFu/experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP / "src"))
import joblib
import pandas as pd
from direction_features import (build_experiment_frame, cot_feature_columns,
                                cot_schema_hash, cot_source_policy)
from experiment_models import predict_recipe


def validate_source_schema(bundle, frame, metadata=None):
    policy = bundle.get("cot_policy", "benchmark")
    if policy not in {"all", "benchmark"}:
        raise ValueError("Saved model has an unsupported COT source policy")
    if policy == "all":
        columns = cot_feature_columns(frame)
        required = set(bundle.get("required_cot_features", []))
        if (not required or required != set(columns)
                or bundle.get("cot_feature_schema_sha256") != cot_schema_hash(columns)):
            raise ValueError("Rebuilt COT source schema differs from the all-COT model; retrain for schema changes")
        if metadata is None or metadata.get("cot_policy") != "all":
            raise ValueError("All-COT prediction requires verified all-COT source-policy metadata")
        if bundle.get("cot_source_policy") != cot_source_policy(metadata):
            raise ValueError("COT availability/source policy differs from the fitted model")
        for member in bundle["members"].values():
            if (not member["spec"].get("require_all_cot", False)
                    or set(member.get("required_cot_features", [])) != required
                    or not required.issubset(member["features"])):
                raise ValueError("A fitted direction/return member omits required COT source fields")


def forecast(bundle, frame, metadata=None):
    validate_source_schema(bundle, frame, metadata)
    row = frame.tail(1)
    origin = row.Date.iloc[0]
    if origin < pd.Timestamp(bundle["fitted_as_of"]):
        raise ValueError("The final fitted bundle cannot forecast historical origins. Use --replay-date.")
    target = origin + pd.Timedelta(days=bundle["horizon"])
    if target.weekday() >= 5:
        target += pd.offsets.BDay(0)
    rows = []
    for name, recipe in bundle["recipes"].items():
        value = float(predict_recipe(recipe, bundle["members"], row)[0])
        classified = recipe["task"] == "class"
        up = value >= (recipe["threshold"] if classified else 0)
        band = bundle["abstention"]
        selective = ("Down" if value < band["lower"] else "Up" if value >= band["upper"] else "Uncertain") if name == "selected_direction" else None
        rows.append({"model": name, "as_of_date": str(origin.date()),
            "model_fitted_as_of": bundle["fitted_as_of"], "horizon_calendar_days": bundle["horizon"],
            "target_date_estimate": str(target.date()), "cached_close": float(row.Close.iloc[0]),
            "probability_up": value if classified else None,
            "probability_note": "Class-weighted classifier output; no independent probability calibration" if classified else None,
            "predicted_return": None if classified else value,
            "implied_close": None if classified else float(row.Close.iloc[0])*(1+value),
            "direction": "Up" if up else "Down", "selective_direction": selective,
            "threshold": recipe["threshold"] if classified else 0,
            "cot_policy": bundle.get("cot_policy", "benchmark"),
            "required_cot_feature_count": len(bundle.get("required_cot_features", [])),
            "data_status": "Forecast at latest cached close; weekday target estimate does not project exchange holidays or missing observations"})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", choices=[28, 30], type=int, default=30)
    parser.add_argument("--replay-date", help="Read an original saved holdout forecast without applying later fitted weights")
    parser.add_argument("--cot-policy", choices=["all", "benchmark"], default="all",
                        help="all reads new required-COT models; benchmark explicitly reads the preserved original experiment")
    parser.add_argument("--bundle", type=Path, help="Explicit model.joblib path, including a separately retrained benchmark")
    args = parser.parse_args()
    out = (args.bundle.parent if args.bundle else
           EXP/"artifacts"/("all_cot/direction" if args.cot_policy == "all" else "direction")/f"{args.horizon}calendar")
    if args.replay_date:
        rows = pd.read_csv(out/"holdout_predictions.csv", parse_dates=["Date"])
        selected = rows.loc[rows.Date.eq(pd.Timestamp(args.replay_date))]
        if len(selected) != 1:
            raise ValueError("Replay date must match a saved out-of-sample forecast origin")
        print(selected.to_json(orient="records", date_format="iso", indent=2))
        return
    bundle = joblib.load(args.bundle or out/"model.joblib")
    if bundle.get("cot_policy", "benchmark") != args.cot_policy:
        raise ValueError("Requested COT source policy does not match the saved model; choose its explicit --cot-policy")
    horizon = bundle["horizon"]
    frame, _, metadata = build_experiment_frame(ROOT, horizon, bundle["unit"], cot_policy=args.cot_policy)
    print(json.dumps(forecast(bundle, frame, metadata), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
