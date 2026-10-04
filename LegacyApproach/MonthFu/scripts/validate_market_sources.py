"""Validate the Yahoo price history and Open-Meteo weather cache.

Both sources feed every model, so a silent change in either would move every
published number without any code edit. This script records what each file
actually contains and asserts the properties the pipeline depends on.

The critical check is the **label drift** test: the forward return is built from
``target_end_date`` and ``Close``. If a refreshed price file changes either, every
saved target, every frozen selection and every holdout metric becomes stale. The
script therefore rebuilds the targets from the committed prices and compares them
against the labels stored in the direction-run artifacts.

Read-only with respect to the data: it writes a single JSON report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def find_project_root(start: Path) -> Path:
    """Return the directory that contains the ``MonthFu`` package.

    The repository was reorganised, so the package is not always a fixed number
    of levels below this script.
    """
    for candidate in [start, *start.parents]:
        if (candidate / "MonthFu" / "src").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find the MonthFu package above {start}")


ROOT = find_project_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(ROOT))

from MonthFu.src.features import load_prices  # noqa: E402
from MonthFu.src.modeling import add_targets  # noqa: E402


def find_data_dir() -> Path:
    """Locate the shared DATA tree, which may be beside or inside the project root.

    DATA holds the COT, price and weather caches. The project root was moved
    during a repository reorganisation, so DATA is now its sibling; probe both
    layouts rather than assuming one.
    """
    roots = [ROOT, ROOT.parent, *ROOT.parents]
    for root in roots:
        for candidate in (root / "DATA", root / "data"):
            if (candidate / "COT").is_dir() and (candidate / "yahoo").is_dir():
                return candidate
    raise FileNotFoundError(
        f"Could not locate the DATA directory near {ROOT}. Looked for DATA/COT and DATA/yahoo.")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def check_prices(data: Path) -> dict:
    path = data / "yahoo/arabica_coffee_futures_history.csv"
    frame, audit = load_prices(path)
    closes = frame.Close.to_numpy(dtype=float)
    gaps = frame.Date.diff().dt.days.dropna()
    years = (frame.Date.max() - frame.Date.min()).days / 365.25
    return {
        "path": str(path), "sha256": sha256(path),
        "rows": int(len(frame)), "rejected_rows": int(audit.get("rejected_rows", 0)),
        "first_session": str(frame.Date.min().date()),
        "last_session": str(frame.Date.max().date()),
        "unique_sessions": bool(frame.Date.is_unique),
        "strictly_increasing": bool(frame.Date.is_monotonic_increasing),
        "positive_closes_only": bool((closes > 0).all()),
        "nan_closes": int(np.isnan(closes).sum()),
        "duplicate_dates": int(frame.Date.duplicated().sum()),
        "invalid_ohlc_rows": (int(frame.price_invalid_ohlc.sum())
                              if "price_invalid_ohlc" in frame else None),
        "low_volume_rows": (int(frame.price_low_volume.sum())
                            if "price_low_volume" in frame else None),
        "max_calendar_gap_days": int(gaps.max()) if len(gaps) else 0,
        "implied_sessions_per_year": round(float(len(frame) / years), 1) if years else None,
    }


def check_weather(data: Path) -> dict:
    path = data / "weather/open_meteo_coffee_regions_daily.csv"
    weather = pd.read_csv(path)
    date_column = next((c for c in weather.columns
                        if str(c).lower() in {"date", "time", "datetime"}), None)
    report: dict = {"path": str(path), "sha256": sha256(path),
                    "rows": int(len(weather)), "columns": int(len(weather.columns)),
                    "date_column": date_column}
    if date_column is None:
        report["error"] = "No obvious date column; alignment cannot be verified."
        return report
    dates = pd.to_datetime(weather[date_column], errors="coerce")
    report.update({
        "first_date": str(dates.min().date()),
        "last_date": str(dates.max().date()),
        "unique_dates": bool(dates.is_unique),
        "null_dates": int(dates.isna().sum()),
    })
    regions = sorted({c.split("_", 2)[1] for c in weather.columns
                      if c.startswith("weather_") and c.count("_") >= 2})
    report["regions"] = regions
    coverage = {}
    for region in regions:
        block = [c for c in weather.columns if c.startswith(f"weather_{region}_")]
        if not block:
            continue
        coverage[region] = {
            "variables": len(block),
            "fully_observed": bool(weather[block].notna().all().all()),
            "observed_fraction": round(float(weather[block].notna().to_numpy().mean()), 4),
        }
    report["per_region"] = coverage
    report["all_regions_fully_observed"] = (
        all(v["fully_observed"] for v in coverage.values()) if coverage else None)
    return report


def check_label_drift(data: Path, horizon: int, artifacts: Path) -> dict:
    """Rebuild targets from committed prices and compare with saved run labels."""
    prices, _ = load_prices(data / "yahoo/arabica_coffee_futures_history.csv")
    rebuilt = add_targets(prices, horizon, "calendar")
    stored_path = artifacts / "features_and_targets.csv.gz"
    if not stored_path.exists():
        return {"status": "skipped", "reason": f"{stored_path} not found"}
    stored = pd.read_csv(stored_path, parse_dates=["Date", "target_end_date"])
    merged = stored[["Date", "target_end_date", "target_return"]].merge(
        rebuilt[["Date", "target_end_date", "target_return"]],
        on="Date", suffixes=("_stored", "_rebuilt"), validate="one_to_one")
    if len(merged) != len(stored):
        return {"status": "failed", "reason": "origin count differs",
                "stored_rows": int(len(stored)), "merged_rows": int(len(merged))}
    left_end = merged.target_end_date_stored
    right_end = merged.target_end_date_rebuilt
    # NaT != NaT in pandas, so compare the two null sets explicitly. Origins too
    # recent to have a matured label are legitimately NaT on both sides.
    both_missing = left_end.isna() & right_end.isna()
    end_mismatch = int((~(left_end.eq(right_end) | both_missing)).sum())
    stored_r = merged.target_return_stored.to_numpy(float)
    rebuilt_r = merged.target_return_rebuilt.to_numpy(float)
    has_stored = np.isfinite(stored_r)
    has_rebuilt = np.isfinite(rebuilt_r)
    # Count origins where exactly one side has a label. Origins unmatured on both
    # sides agree, so they are excluded rather than subtracted, which would let
    # the count go negative.
    disputed = has_stored != has_rebuilt
    presence_mismatch = int((disputed & (has_stored | has_rebuilt)).sum())
    both_labelled = has_stored & has_rebuilt
    value_mismatch = int((~np.isclose(stored_r[both_labelled],
                                      rebuilt_r[both_labelled],
                                      rtol=1e-9, atol=1e-12)).sum())
    return {
        "status": ("passed" if not end_mismatch and not value_mismatch
                   and not presence_mismatch else "failed"),
        "horizon_days": horizon, "origins": int(len(merged)),
        "target_end_date_mismatches": end_mismatch,
        "target_return_mismatches": value_mismatch,
        "label_presence_mismatches": presence_mismatch,
        "unmatured_origins": int(both_missing.sum()),
        "note": ("A non-zero count means the price file moved relative to the "
                 "frozen run and every saved metric must be regenerated."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", nargs="+", type=int, default=[30, 28])
    parser.add_argument("--artifacts-root", type=Path,
                        default=ROOT / "MonthFu/experiments/artifacts/all_cot/direction")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "MonthFu/artifacts/source_validation.json")
    args = parser.parse_args()

    data = find_data_dir()
    report = {"data_dir": str(data), "prices": check_prices(data),
              "weather": check_weather(data), "label_drift": {}}
    for horizon in args.horizons:
        report["label_drift"][str(horizon)] = check_label_drift(
            data, horizon, args.artifacts_root / f"{horizon}calendar")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    prices, weather = report["prices"], report["weather"]
    print(f"prices : {prices['rows']:,} rows  {prices['first_session']} -> "
          f"{prices['last_session']}  sha={prices['sha256'][:12]}")
    print(f"         unique={prices['unique_sessions']} "
          f"increasing={prices['strictly_increasing']} "
          f"positive_closes={prices['positive_closes_only']}")
    print(f"weather: {weather['rows']:,} rows  {weather.get('first_date')} -> "
          f"{weather.get('last_date')}  regions={weather.get('regions')}")
    print(f"         all_regions_fully_observed={weather.get('all_regions_fully_observed')}")
    for horizon, drift in report["label_drift"].items():
        print(f"labels {horizon}d: {drift['status']}  "
              f"end-date mismatches={drift.get('target_end_date_mismatches')}  "
              f"return mismatches={drift.get('target_return_mismatches')}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


