from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "Scripts" / "project" / "src"))

from arabica_modeling import COFFEE_REGIONS, EXTENDED_WEATHER_DAILY, CORE_WEATHER_DAILY  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh missing Open-Meteo coffee-region weather cache.")
    parser.add_argument("--central-data", default="data/centralData/yahoo_cot_full_outer_by_date.csv")
    parser.add_argument("--weather-cache", default="data/weather/open_meteo_coffee_regions_daily.csv")
    parser.add_argument("--regions", nargs="*", default=None, help="Region keys to fetch. Default: missing regions.")
    parser.add_argument("--force-all", action="store_true", help="Fetch all configured regions.")
    parser.add_argument("--sleep", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    central_path = ROOT / args.central_data
    cache_path = ROOT / args.weather_cache

    dates = pd.read_csv(central_path, usecols=["Date"])
    start_date = pd.to_datetime(dates["Date"]).min().date().isoformat()
    end_date = pd.to_datetime(dates["Date"]).max().date().isoformat()

    if cache_path.exists():
        combined = pd.read_csv(cache_path, parse_dates=["Date"])
    else:
        combined = pd.DataFrame({"Date": pd.date_range(start_date, end_date, freq="D")})

    present_regions = {
        region
        for region in COFFEE_REGIONS
        if any(col.startswith(f"weather_{region}_") for col in combined.columns)
    }
    if args.force_all:
        regions_to_fetch = list(COFFEE_REGIONS)
    elif args.regions:
        regions_to_fetch = args.regions
    else:
        regions_to_fetch = [region for region in COFFEE_REGIONS if region not in present_regions]

    print(f"Date range: {start_date} to {end_date}")
    print(f"Present regions: {sorted(present_regions)}")
    print(f"Fetching regions: {regions_to_fetch}")

    successes = {}
    errors = {}
    for region_name in regions_to_fetch:
        if region_name not in COFFEE_REGIONS:
            errors[region_name] = "unknown region"
            continue
        region = COFFEE_REGIONS[region_name]
        print(f"Fetching {region_name}...")
        try:
            part = fetch_region(region_name, region, start_date, end_date, EXTENDED_WEATHER_DAILY)
        except requests.HTTPError as exc:
            print(f"Extended variables failed for {region_name}: {exc}")
            try:
                time.sleep(args.sleep)
                part = fetch_region(region_name, region, start_date, end_date, CORE_WEATHER_DAILY)
            except requests.RequestException as core_exc:
                errors[region_name] = str(core_exc)
                print(f"Core variables failed for {region_name}: {core_exc}")
                time.sleep(args.sleep)
                continue
        except requests.RequestException as exc:
            errors[region_name] = str(exc)
            print(f"Fetch failed for {region_name}: {exc}")
            time.sleep(args.sleep)
            continue

        region_cols = [col for col in combined.columns if col.startswith(f"weather_{region_name}_")]
        combined = combined.drop(columns=region_cols, errors="ignore")
        combined = combined.merge(part, on="Date", how="outer")
        successes[region_name] = {
            "rows": int(len(part)),
            "columns": [col for col in part.columns if col != "Date"],
        }
        temp_path = cache_path.with_name(f"{cache_path.stem}__{region_name}.csv")
        part.to_csv(temp_path, index=False)
        print(f"Saved per-region cache: {temp_path}")
        time.sleep(args.sleep)

    if successes:
        combined = combined.sort_values("Date")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path = cache_path.with_suffix(".backup_before_refresh.csv")
        if cache_path.exists() and not backup_path.exists():
            cache_path.replace(backup_path)
            print(f"Backed up previous combined cache: {backup_path}")
        combined.to_csv(cache_path, index=False)
        metadata_path = cache_path.with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps(
                {
                    "source": "https://archive-api.open-meteo.com/v1/archive",
                    "docs": "https://open-meteo.com/en/docs/historical-weather-api",
                    "regions": COFFEE_REGIONS,
                    "start_date": start_date,
                    "end_date": end_date,
                    "variables": EXTENDED_WEATHER_DAILY,
                    "present_regions": sorted(
                        {
                            region
                            for region in COFFEE_REGIONS
                            if any(col.startswith(f"weather_{region}_") for col in combined.columns)
                        }
                    ),
                    "refresh_successes": successes,
                    "refresh_errors": errors,
                },
                indent=2,
            )
        )
        print(f"Updated combined cache: {cache_path}")
    else:
        print("No new region data fetched; combined cache left unchanged.")

    if errors:
        print("Errors:")
        for region, error in errors.items():
            print(f"- {region}: {error}")


def fetch_region(region_name: str, region: dict, start_date: str, end_date: str, variables: list[str]) -> pd.DataFrame:
    params = {
        "latitude": region["latitude"],
        "longitude": region["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(variables),
        "timezone": region["timezone"],
    }
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    daily = pd.DataFrame(payload["daily"])
    daily["Date"] = pd.to_datetime(daily["time"], errors="coerce")
    daily = daily.drop(columns=["time"])
    daily = daily.rename(columns={col: f"weather_{region_name}_{col}" for col in daily.columns if col != "Date"})
    return daily


if __name__ == "__main__":
    main()
