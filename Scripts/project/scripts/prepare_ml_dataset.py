from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "Scripts" / "project" / "src"))

from arabica_modeling import (  # noqa: E402
    TARGET_CLOSE,
    TARGET_DIRECTION,
    TARGET_RETURN,
    _is_model_relevant_feature,
    build_modeling_frame,
)


CORE_PRICE_COLUMNS = ["Date", "Close", "High", "Low", "Open", "Volume"]
TARGET_COLUMNS = [
    "target_close_1d",
    "target_return_1d",
    "target_direction_1d",
    TARGET_CLOSE,
    TARGET_RETURN,
    TARGET_DIRECTION,
]
ALWAYS_KEEP = CORE_PRICE_COLUMNS + TARGET_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a clean Arabica ML training dataset.")
    parser.add_argument(
        "--central-data",
        default="data/centralData/yahoo_cot_full_outer_by_date_cot_ffill.csv",
        help="Yahoo/COT outer join input CSV.",
    )
    parser.add_argument(
        "--weather-data",
        default="data/weather/open_meteo_coffee_regions_daily.csv",
        help="Daily weather feature CSV aligned by Date.",
    )
    parser.add_argument(
        "--output",
        default="data/centralData/arabica_ml_model_ready.csv",
        help="Cleaned model-ready output CSV.",
    )
    parser.add_argument(
        "--report",
        default="data/centralData/arabica_ml_model_ready.report.json",
        help="Cleanup report JSON path.",
    )
    parser.add_argument(
        "--max-missing-pct",
        type=float,
        default=0.20,
        help="Drop non-core feature columns missing more than this share of rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    central_path = ROOT / args.central_data
    weather_path = ROOT / args.weather_data
    output_path = ROOT / args.output
    report_path = ROOT / args.report

    feature_set = build_modeling_frame(
        central_path=central_path,
        weather_path=weather_path if weather_path.exists() else None,
        prediction_mode=True,
    )
    frame = feature_set.frame.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    yahoo_date_count_before = len(frame)
    frame = frame.loc[frame[["Close", "High", "Low", "Open", "Volume"]].notna().all(axis=1)].copy()
    yahoo_date_count_after = len(frame)

    frame["target_close_1d"] = frame["Close"].shift(-1)
    frame["target_return_1d"] = frame["target_close_1d"] / frame["Close"] - 1.0
    frame["target_direction_1d"] = frame["target_return_1d"] > 0

    candidate_columns = []
    for column in frame.columns:
        if column in ALWAYS_KEEP:
            candidate_columns.append(column)
            continue
        if column in {"cot_report_date", "cot_release_date"}:
            continue
        if column.startswith("weather_"):
            candidate_columns.append(column)
            continue
        if column in feature_set.numeric_features and _is_model_relevant_feature(column):
            candidate_columns.append(column)

    selected_columns = []
    dropped_missing = {}
    dropped_non_numeric = []
    missing_pct = frame[candidate_columns].isna().mean()
    for column in candidate_columns:
        if column in ALWAYS_KEEP:
            selected_columns.append(column)
            continue
        if missing_pct[column] > args.max_missing_pct:
            dropped_missing[column] = float(missing_pct[column])
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not pd.api.types.is_numeric_dtype(frame[column]):
            dropped_non_numeric.append(column)
            continue
        if frame[column].nunique(dropna=True) <= 1:
            dropped_non_numeric.append(column)
            continue
        selected_columns.append(column)

    selected_columns = list(dict.fromkeys(selected_columns))
    selected_columns = [
        *[column for column in CORE_PRICE_COLUMNS if column in selected_columns],
        *[
            column
            for column in selected_columns
            if column not in set(CORE_PRICE_COLUMNS) | set(TARGET_COLUMNS)
        ],
        *[column for column in TARGET_COLUMNS if column in selected_columns],
    ]

    clean = frame[selected_columns].replace([np.inf, -np.inf], np.nan).copy()
    clean["Date"] = pd.to_datetime(clean["Date"]).dt.date.astype(str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)

    weather_columns = [column for column in selected_columns if column.startswith("weather_")]
    cot_columns = [
        column
        for column in selected_columns
        if column not in ALWAYS_KEEP and not column.startswith("weather_") and column != "Date"
    ]
    report = {
        "input_central_data": str(central_path.relative_to(ROOT)),
        "input_weather_data": str(weather_path.relative_to(ROOT)) if weather_path.exists() else None,
        "output": str(output_path.relative_to(ROOT)),
        "rows": int(len(clean)),
        "columns": int(len(clean.columns)),
        "date_start": clean["Date"].min(),
        "date_end": clean["Date"].max(),
        "yahoo_price_dates_before_ohlcv_filter": int(yahoo_date_count_before),
        "yahoo_price_dates_after_ohlcv_filter": int(yahoo_date_count_after),
        "max_missing_pct": args.max_missing_pct,
        "core_price_columns": CORE_PRICE_COLUMNS,
        "target_columns": [column for column in TARGET_COLUMNS if column in clean.columns],
        "weather_feature_count": len(weather_columns),
        "cot_and_calendar_feature_count": len(cot_columns),
        "dropped_for_missing_count": len(dropped_missing),
        "dropped_for_missing": dict(sorted(dropped_missing.items())),
        "dropped_non_numeric_or_constant_count": len(dropped_non_numeric),
        "dropped_non_numeric_or_constant": sorted(dropped_non_numeric),
        "dropped_by_model_builder_count": len(feature_set.dropped_features),
        "dropped_by_model_builder": feature_set.dropped_features,
    }
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Saved cleaned dataset: {output_path}")
    print(f"Saved cleanup report: {report_path}")
    print(f"Rows: {len(clean):,}; columns: {len(clean.columns):,}")
    print(f"Weather features: {len(weather_columns):,}; COT/calendar features: {len(cot_columns):,}")
    print(f"Date range: {report['date_start']} to {report['date_end']}")


if __name__ == "__main__":
    main()
