from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


PRICE_AND_TARGET_COLUMNS = {
    "Date",
    "date_match_status",
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "return_1d",
    "return_5d",
    "range_pct",
    "close_to_open_pct",
    "volume_change_pct",
    "moving_avg_5",
    "moving_avg_20",
    "close_vs_ma_20",
    "future_close_1d",
    "future_close_5d",
    "future_return_1d",
    "future_return_5d",
    "future_direction_5d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forward-fill weekly COT values onto daily Yahoo Coffee C rows without filling price/target columns."
    )
    parser.add_argument("--input", default="data/centralData/yahoo_cot_full_outer_by_date.csv")
    parser.add_argument("--output", default="data/centralData/yahoo_cot_full_outer_by_date_cot_ffill.csv")
    parser.add_argument("--max-days", type=int, default=10, help="Maximum days after a COT report to keep forward-filled values.")
    parser.add_argument("--overwrite-input", action="store_true", help="Also replace the input file after writing a backup.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)

    df = pd.read_csv(input_path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["cot_report_date"] = pd.to_datetime(df["cot_report_date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)

    original_cot_present = df["cot_report_date"].notna()
    cot_fill_columns = [col for col in df.columns if col not in PRICE_AND_TARGET_COLUMNS]

    filled = df.copy()
    filled[cot_fill_columns] = filled[cot_fill_columns].ffill()
    filled["cot_days_since_report"] = (filled["Date"] - filled["cot_report_date"]).dt.days

    stale_or_early = filled["cot_report_date"].isna() | filled["cot_days_since_report"].gt(args.max_days)
    filled.loc[stale_or_early, cot_fill_columns] = pd.NA
    filled.loc[stale_or_early, "cot_days_since_report"] = pd.NA

    filled["cot_fill_status"] = "no_cot_available"
    filled.loc[filled["cot_report_date"].notna() & original_cot_present, "cot_fill_status"] = "original_cot_row"
    filled.loc[filled["cot_report_date"].notna() & ~original_cot_present, "cot_fill_status"] = "forward_filled_from_previous_cot"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filled.to_csv(output_path, index=False)

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "rows": int(len(df)),
        "cot_fill_columns": cot_fill_columns,
        "max_days": args.max_days,
        "original_cot_rows": int(original_cot_present.sum()),
        "forward_filled_rows": int(filled["cot_fill_status"].eq("forward_filled_from_previous_cot").sum()),
        "no_cot_available_rows": int(filled["cot_fill_status"].eq("no_cot_available").sum()),
        "open_interest_missing_before": int(df["Open_Interest_All"].isna().sum()) if "Open_Interest_All" in df else None,
        "open_interest_missing_after": int(filled["Open_Interest_All"].isna().sum()) if "Open_Interest_All" in filled else None,
    }
    report_path = output_path.with_suffix(".fill_report.json")
    report_path.write_text(json.dumps(report, indent=2))

    if args.overwrite_input:
        backup_path = input_path.with_suffix(".backup_before_cot_ffill.csv")
        if not backup_path.exists():
            input_path.replace(backup_path)
        filled.to_csv(input_path, index=False)
        report["backup"] = str(backup_path)
        report["input_overwritten"] = True
        report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
