from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZIP = ROOT / "EventsData" / "Zip" / "2000.zip"
DEFAULT_PRICE_PATH = ROOT / "data" / "centralData" / "yahoo_cot_full_outer_by_date_cot_ffill.csv"
DEFAULT_DATE_SOURCE = ROOT / "project" / "artifacts" / "outputs" / "yahoo_price_drift_events.csv"
OUTPUT_DIR = ROOT / "project" / "artifacts" / "outputs"
PLOT_DIR = ROOT / "project" / "artifacts" / "plots"
FILTERED_EVENT_DIR = ROOT / "data" / "events"

GDELT_COLUMNS = [
    "GLOBALEVENTID",
    "SQLDATE",
    "MonthYear",
    "Year",
    "FractionDate",
    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1EthnicCode",
    "Actor1Religion1Code",
    "Actor1Religion2Code",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",
    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2EthnicCode",
    "Actor2Religion1Code",
    "Actor2Religion2Code",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_Type",
    "Actor1Geo_FullName",
    "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type",
    "Actor2Geo_FullName",
    "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED",
]

EVENT_USECOLS = [
    "GLOBALEVENTID",
    "SQLDATE",
    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1Type1Code",
    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2Type1Code",
    "IsRootEvent",
    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "Actor1Geo_FullName",
    "Actor1Geo_CountryCode",
    "Actor2Geo_FullName",
    "Actor2Geo_CountryCode",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "DATEADDED",
]

COFFEE_TERMS = [
    "coffee",
    "arabica",
    "robusta",
    "coffee c",
    "coffee futures",
    "coffee crop",
    "coffee exports",
    "coffee production",
]

COFFEE_COUNTRY_CODES = {
    "BRA",
    "BR",
    "COL",
    "CO",
    "VNM",
    "VM",
    "IDN",
    "ID",
    "ETH",
    "ET",
    "UGA",
    "UG",
    "HND",
    "HO",
    "PER",
    "PE",
    "MEX",
    "MX",
    "GTM",
    "GT",
    "IND",
    "IN",
    "CRI",
    "CS",
    "SLV",
    "ES",
    "NIC",
    "NU",
    "PNG",
    "PP",
    "CIV",
    "IV",
    "KEN",
    "KE",
    "TZA",
    "TZ",
    "ECU",
    "EC",
}

COFFEE_COUNTRY_NAMES = [
    "brazil",
    "colombia",
    "vietnam",
    "indonesia",
    "ethiopia",
    "uganda",
    "honduras",
    "peru",
    "mexico",
    "guatemala",
    "india",
    "costa rica",
    "el salvador",
    "nicaragua",
    "papua new guinea",
    "ivory coast",
    "cote d'ivoire",
    "kenya",
    "tanzania",
    "ecuador",
    "minas gerais",
]

SUPPLY_RISK_ROOTS = {"14", "17", "18", "19", "20"}
SUPPLY_SUPPORT_ROOTS = {"01", "02", "03", "04", "05", "06", "07", "08"}
COT_CONTEXT_COLUMNS = [
    "Open_Interest_All",
    "managed_money_net",
    "commercial_net",
    "noncommercial_net",
    "managed_money_weekly_net_change",
    "commercial_weekly_net_change",
    "noncommercial_weekly_net_change",
    "open_interest_change_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract GDELT events from a yearly zip and score Coffee C futures impact by date."
    )
    parser.add_argument("--zip-path", default=str(DEFAULT_ZIP), help="GDELT yearly zip, e.g. EventsData/Zip/2000.zip")
    parser.add_argument("--price-path", default=str(DEFAULT_PRICE_PATH))
    parser.add_argument("--date-source", default=str(DEFAULT_DATE_SOURCE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--plots-dir", default=str(PLOT_DIR))
    parser.add_argument("--filtered-dir", default=str(FILTERED_EVENT_DIR))
    parser.add_argument("--year", type=int, default=None, help="Override year inferred from the zip filename.")
    parser.add_argument("--target-window-before", type=int, default=5)
    parser.add_argument("--target-window-after", type=int, default=7)
    parser.add_argument("--scan-full-year", action="store_true", help="Scan all dates instead of only price-event windows.")
    parser.add_argument("--max-date-lag-days", type=int, default=5, help="Map non-trading event dates to next trading day.")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--top-events-per-target", type=int, default=30)
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama to rescore the highest-ranked event rows.")
    parser.add_argument("--ollama-model", default="llama3.2:latest")
    parser.add_argument("--max-ollama-events", type=int, default=12)
    parser.add_argument("--ollama-sleep", type=float, default=0.2)
    parser.add_argument("--ollama-timeout", type=float, default=30.0)
    parser.add_argument("--ollama-num-predict", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    plots_dir = resolve_path(args.plots_dir)
    filtered_dir = resolve_path(args.filtered_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    zip_path = resolve_path(args.zip_path)
    year = args.year or infer_year(zip_path)
    load_env_files([ROOT / ".env", ROOT / "data" / "centralData" / ".env"])

    price = load_price_data(resolve_path(args.price_path), year)
    target_dates = load_target_dates(
        resolve_path(args.date_source),
        year,
        before_days=args.target_window_before,
        after_days=args.target_window_after,
    )
    date_filter = None if args.scan_full_year else target_dates["scan_dates_yyyymmdd"]

    print(f"Streaming {zip_path} for {year} events", flush=True)
    if date_filter is None:
        print("Date filter: full year", flush=True)
    else:
        print(f"Date filter: {len(date_filter)} calendar days around {len(target_dates['target_dates'])} price point(s)", flush=True)

    raw_events = stream_relevant_events(zip_path, date_filter, args.chunksize)
    if raw_events.empty:
        write_empty_outputs(year, output_dir, filtered_dir, plots_dir, target_dates)
        print("No coffee-relevant GDELT events found for the selected dates.", flush=True)
        return

    all_events = attach_price_context(raw_events, price, args.max_date_lag_days)
    all_events = score_events(all_events, price)
    all_events = apply_ollama_if_requested(all_events, args)
    events = select_top_events(all_events, target_dates, args.top_events_per_target)

    daily = build_daily_summary(all_events)
    points = build_price_point_summary(all_events, target_dates, price)

    detail_path = output_dir / f"gdelt_coffee_event_impact_{year}.csv"
    candidates_path = output_dir / f"gdelt_coffee_event_candidates_{year}.csv"
    daily_path = output_dir / f"gdelt_coffee_event_daily_impact_{year}.csv"
    points_path = output_dir / f"gdelt_coffee_event_price_points_{year}.csv"
    filtered_extract_path = filtered_dir / f"gdelt_coffee_events_{year}_filtered.csv"
    metadata_path = output_dir / f"gdelt_coffee_event_impact_metadata_{year}.json"

    events.to_csv(detail_path, index=False)
    all_events.to_csv(candidates_path, index=False)
    all_events.to_csv(filtered_extract_path, index=False)
    daily.to_csv(daily_path, index=False)
    points.to_csv(points_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "year": year,
                "zip_path": str(zip_path),
                "price_path": str(resolve_path(args.price_path)),
                "date_source": str(resolve_path(args.date_source)),
                "scan_full_year": args.scan_full_year,
                "target_window_before": args.target_window_before,
                "target_window_after": args.target_window_after,
                "rows_after_filtering": int(len(all_events)),
                "selected_event_rows": int(len(events)),
                "daily_rows": int(len(daily)),
                "price_point_rows": int(len(points)),
                "ollama_requested": bool(args.use_ollama),
                "ollama_model": args.ollama_model if args.use_ollama else None,
                "ollama_scored_rows": int(all_events["ollama_impact_score_0_1"].notna().sum())
                if "ollama_impact_score_0_1" in all_events
                else 0,
                "ollama_status_counts": all_events["ollama_status"].value_counts(dropna=False).to_dict()
                if "ollama_status" in all_events
                else {},
                "score_meaning": "final_impact_score_0_1 is 0=no plausible Coffee C impact and 1=highest likely one-week Coffee C futures impact.",
                "outputs": {
                    "selected_event_detail": str(detail_path),
                    "all_scored_candidates": str(candidates_path),
                    "filtered_extract_all_scored_candidates": str(filtered_extract_path),
                    "daily_summary": str(daily_path),
                    "price_point_summary": str(points_path),
                },
            },
            indent=2,
        )
    )

    write_plots(events, daily, price, target_dates, plots_dir, year)

    print(f"Saved event detail: {detail_path}")
    print(f"Saved all scored candidates: {candidates_path}")
    print(f"Saved filtered extraction: {filtered_extract_path}")
    print(f"Saved daily summary: {daily_path}")
    print(f"Saved price-point summary: {points_path}")
    print(f"Saved metadata: {metadata_path}")
    print("Top scored events:")
    top_cols = [
        "event_date",
        "trading_date",
        "final_impact_score_0_1",
        "impact_direction",
        "future_return_5d",
        "event_summary",
    ]
    print(events.sort_values("final_impact_score_0_1", ascending=False)[top_cols].head(12).to_string(index=False))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def infer_year(zip_path: Path) -> int:
    match = re.search(r"(19|20)\d{2}", zip_path.stem)
    if not match:
        raise ValueError(f"Could not infer year from {zip_path}; pass --year.")
    return int(match.group(0))


def load_env_files(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    if os.getenv("OLLAMA_KEY") and not os.getenv("OLLAMA_API_KEY"):
        os.environ["OLLAMA_API_KEY"] = os.getenv("OLLAMA_KEY", "")


def load_price_data(price_path: Path, year: int) -> pd.DataFrame:
    price = pd.read_csv(price_path, parse_dates=["Date"], low_memory=False)
    price = price.sort_values("Date").reset_index(drop=True)
    for column in ["Close", "Open", "High", "Low", "Volume", *COT_CONTEXT_COLUMNS]:
        if column in price.columns:
            price[column] = pd.to_numeric(price[column], errors="coerce")
    price["return_1d"] = price["Close"].pct_change()
    for horizon in [1, 5, 10, 20]:
        price[f"future_close_{horizon}d"] = price["Close"].shift(-horizon)
        price[f"future_return_{horizon}d"] = price[f"future_close_{horizon}d"] / price["Close"] - 1
    price["realized_vol_20d"] = price["return_1d"].rolling(20, min_periods=10).std() * np.sqrt(252)
    price["abs_future_return_5d"] = price["future_return_5d"].abs()
    price["abs_return_5d_p95"] = price["abs_future_return_5d"].rolling(252, min_periods=80).quantile(0.95)
    fallback = price.loc[price["Date"].dt.year.eq(year), "abs_future_return_5d"].quantile(0.95)
    if pd.isna(fallback) or fallback <= 0:
        fallback = price["abs_future_return_5d"].quantile(0.95)
    price["abs_return_5d_p95"] = price["abs_return_5d_p95"].fillna(fallback)
    return price


def load_target_dates(date_source: Path, year: int, before_days: int, after_days: int) -> dict:
    if not date_source.exists():
        dates = []
        source_frame = pd.DataFrame()
    else:
        source_frame = pd.read_csv(date_source)
        date_col = next((col for col in ["peak_date", "Date", "event_date", "date"] if col in source_frame.columns), None)
        if date_col is None:
            raise ValueError(f"{date_source} must contain one of peak_date, Date, event_date, or date.")
        source_frame[date_col] = pd.to_datetime(source_frame[date_col], errors="coerce")
        source_frame = source_frame.loc[source_frame[date_col].dt.year.eq(year)].copy()
        dates = sorted(source_frame[date_col].dropna().dt.normalize().unique())

    scan_dates = set()
    for date in dates:
        for day in pd.date_range(date - pd.Timedelta(days=before_days), date + pd.Timedelta(days=after_days), freq="D"):
            scan_dates.add(day.strftime("%Y%m%d"))
    return {
        "source_rows": source_frame,
        "target_dates": list(dates),
        "scan_dates_yyyymmdd": scan_dates,
        "before_days": before_days,
        "after_days": after_days,
    }


def stream_relevant_events(zip_path: Path, date_filter: set[str] | None, chunksize: int) -> pd.DataFrame:
    parts = []
    processed = 0
    kept = 0
    with ZipFile(zip_path) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_members:
            raise FileNotFoundError(f"No CSV member found inside {zip_path}")
        member = csv_members[0]
        with archive.open(member) as handle:
            reader = pd.read_csv(
                handle,
                sep="\t",
                header=None,
                names=GDELT_COLUMNS,
                usecols=EVENT_USECOLS,
                dtype=str,
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk_number, chunk in enumerate(reader, start=1):
                processed += len(chunk)
                if date_filter is not None:
                    chunk = chunk.loc[chunk["SQLDATE"].isin(date_filter)].copy()
                if chunk.empty:
                    if chunk_number % 20 == 0:
                        print(f"  processed {processed:,} rows, kept {kept:,}", flush=True)
                    continue
                chunk = keep_coffee_relevant(chunk)
                if not chunk.empty:
                    kept += len(chunk)
                    parts.append(chunk)
                if chunk_number % 20 == 0:
                    print(f"  processed {processed:,} rows, kept {kept:,}", flush=True)
    if not parts:
        return pd.DataFrame(columns=EVENT_USECOLS)
    out = pd.concat(parts, ignore_index=True)
    print(f"Finished zip scan: processed {processed:,} rows, kept {len(out):,}", flush=True)
    return out


def keep_coffee_relevant(chunk: pd.DataFrame) -> pd.DataFrame:
    text_cols = [
        "Actor1Name",
        "Actor2Name",
        "Actor1Code",
        "Actor2Code",
        "Actor1Geo_FullName",
        "Actor2Geo_FullName",
        "ActionGeo_FullName",
    ]
    text = chunk[text_cols].fillna("").agg(" ".join, axis=1).str.lower()
    term_pattern = "|".join(re.escape(term) for term in COFFEE_TERMS)
    country_name_pattern = "|".join(re.escape(term) for term in COFFEE_COUNTRY_NAMES)
    term_match = text.str.contains(term_pattern, regex=True, na=False)
    country_name_match = text.str.contains(country_name_pattern, regex=True, na=False)
    country_cols = [
        "Actor1CountryCode",
        "Actor2CountryCode",
        "Actor1Geo_CountryCode",
        "Actor2Geo_CountryCode",
        "ActionGeo_CountryCode",
    ]
    country_match = chunk[country_cols].fillna("").isin(COFFEE_COUNTRY_CODES).any(axis=1) | country_name_match
    out = chunk.loc[term_match | country_match].copy()
    out["keyword_relevant"] = term_match.loc[out.index].to_numpy()
    out["coffee_country_relevant"] = country_match.loc[out.index].to_numpy()
    out["coffee_country_name_relevant"] = country_name_match.loc[out.index].to_numpy()
    return out


def attach_price_context(events: pd.DataFrame, price: pd.DataFrame, max_lag_days: int) -> pd.DataFrame:
    events = events.copy()
    events["event_date"] = pd.to_datetime(events["SQLDATE"], format="%Y%m%d", errors="coerce")
    events = events.loc[events["event_date"].notna()].reset_index(drop=True)
    events["_event_row_id"] = np.arange(len(events))

    price_cols = [
        "Date",
        "Close",
        "Open",
        "High",
        "Low",
        "Volume",
        "return_1d",
        "future_close_1d",
        "future_close_5d",
        "future_close_10d",
        "future_close_20d",
        "future_return_1d",
        "future_return_5d",
        "future_return_10d",
        "future_return_20d",
        "realized_vol_20d",
        "abs_return_5d_p95",
    ]
    price_cols.extend([col for col in COT_CONTEXT_COLUMNS if col in price.columns])
    right = price[price_cols].rename(columns={"Date": "trading_date"}).sort_values("trading_date")
    mapped = pd.merge_asof(
        events.sort_values("event_date"),
        right,
        left_on="event_date",
        right_on="trading_date",
        direction="forward",
        tolerance=pd.Timedelta(days=max_lag_days),
    )
    mapped["event_to_trade_lag_days"] = (mapped["trading_date"] - mapped["event_date"]).dt.days
    return mapped.sort_values("_event_row_id").drop(columns=["_event_row_id"]).reset_index(drop=True)


def score_events(events: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    scored = events.copy()
    for column in ["GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone", "QuadClass"]:
        scored[column] = pd.to_numeric(scored[column], errors="coerce")

    scored["event_summary"] = scored.apply(build_event_summary, axis=1)
    scored["primary_country"] = scored.apply(primary_country, axis=1)
    scored["relevance_score_0_1"] = np.select(
        [
            scored["keyword_relevant"].astype(bool) & scored["coffee_country_relevant"].astype(bool),
            scored["keyword_relevant"].astype(bool),
            scored["coffee_country_relevant"].astype(bool),
        ],
        [1.0, 0.9, 0.65],
        default=0.2,
    )
    mentions = np.log1p(scored["NumMentions"].fillna(0)).clip(lower=0) / np.log1p(60)
    articles = np.log1p(scored["NumArticles"].fillna(0)).clip(lower=0) / np.log1p(30)
    sources = np.log1p(scored["NumSources"].fillna(0)).clip(lower=0) / np.log1p(15)
    tone = scored["AvgTone"].fillna(0).abs().clip(upper=12) / 12
    goldstein = scored["GoldsteinScale"].fillna(0).abs().clip(upper=10) / 10
    quad = scored["QuadClass"].map({1: 0.15, 2: 0.3, 3: 0.7, 4: 0.9}).fillna(0.2)
    scored["event_intensity_score_0_1"] = (
        0.25 * mentions.clip(upper=1)
        + 0.20 * articles.clip(upper=1)
        + 0.15 * sources.clip(upper=1)
        + 0.20 * tone
        + 0.20 * np.maximum(goldstein, quad)
    ).clip(0, 1)

    threshold = scored["abs_return_5d_p95"].replace(0, np.nan)
    fallback_threshold = price["abs_future_return_5d"].quantile(0.95)
    threshold = threshold.fillna(fallback_threshold if pd.notna(fallback_threshold) and fallback_threshold > 0 else 0.05)
    scored["price_response_score_0_1"] = (scored["future_return_5d"].abs() / threshold).clip(0, 1)
    scored["impact_direction"] = scored.apply(infer_impact_direction, axis=1)
    observed_direction = np.sign(pd.to_numeric(scored["future_return_5d"], errors="coerce").fillna(0))
    expected_direction = scored["impact_direction"].map({"bullish": 1, "bearish": -1}).fillna(0)
    scored["direction_alignment_score_0_1"] = np.where(
        expected_direction.eq(0),
        0.5,
        np.where(expected_direction.eq(observed_direction), 1.0, 0.0),
    )
    scored["rule_impact_score_0_1"] = (
        0.38 * scored["relevance_score_0_1"]
        + 0.24 * scored["event_intensity_score_0_1"]
        + 0.28 * scored["price_response_score_0_1"]
        + 0.10 * scored["direction_alignment_score_0_1"]
    ).clip(0, 1)
    scored["ollama_impact_score_0_1"] = np.nan
    scored["ollama_direction"] = ""
    scored["ollama_rationale"] = ""
    scored["ollama_status"] = "not_requested"
    scored["final_impact_score_0_1"] = scored["rule_impact_score_0_1"]
    scored["impact_bucket"] = pd.cut(
        scored["final_impact_score_0_1"],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001],
        labels=["minimal", "weak", "moderate", "strong", "severe"],
    ).astype(str)
    return scored


def build_event_summary(row: pd.Series) -> str:
    actor1 = clean_value(row.get("Actor1Name")) or clean_value(row.get("Actor1Code")) or "Unknown actor"
    actor2 = clean_value(row.get("Actor2Name")) or clean_value(row.get("Actor2Code")) or "unknown target"
    location = (
        clean_value(row.get("ActionGeo_FullName"))
        or clean_value(row.get("Actor1Geo_FullName"))
        or clean_value(row.get("Actor2Geo_FullName"))
        or "unknown location"
    )
    return (
        f"{actor1} -> {actor2}; event code {clean_value(row.get('EventCode'))}; "
        f"Goldstein {clean_value(row.get('GoldsteinScale'))}; tone {clean_value(row.get('AvgTone'))}; {location}"
    )


def clean_value(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def primary_country(row: pd.Series) -> str:
    for column in ["ActionGeo_CountryCode", "Actor1Geo_CountryCode", "Actor2Geo_CountryCode", "Actor1CountryCode", "Actor2CountryCode"]:
        value = clean_value(row.get(column))
        if value:
            return value
    return ""


def infer_impact_direction(row: pd.Series) -> str:
    root = str(clean_value(row.get("EventRootCode"))).zfill(2)
    goldstein = pd.to_numeric(row.get("GoldsteinScale"), errors="coerce")
    producer_event = bool(row.get("coffee_country_relevant"))
    if producer_event and (root in SUPPLY_RISK_ROOTS or row.get("QuadClass", 0) in [3, 4] or (pd.notna(goldstein) and goldstein < -2)):
        return "bullish"
    if producer_event and (root in SUPPLY_SUPPORT_ROOTS or (pd.notna(goldstein) and goldstein > 3)):
        return "bearish"
    return "neutral"


def apply_ollama_if_requested(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not args.use_ollama or events.empty:
        return events

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    api_key = os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_KEY") or ""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if not ollama_available(base_url, headers):
        print("Ollama endpoint was not reachable; keeping deterministic event scores.", file=sys.stderr, flush=True)
        out = events.copy()
        out["ollama_status"] = "endpoint_unreachable"
        return out

    out = events.copy()
    out["ollama_status"] = "not_selected_for_ollama"
    candidate_idx = out.sort_values("rule_impact_score_0_1", ascending=False).head(args.max_ollama_events).index
    for n, idx in enumerate(candidate_idx, start=1):
        row = out.loc[idx]
        prompt = build_ollama_prompt(row)
        out.loc[idx, "ollama_status"] = "parse_failed"
        try:
            response = requests.post(
                f"{base_url}/api/chat",
                headers=headers,
                json={
                    "model": args.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0, "num_predict": args.ollama_num_predict},
                },
                timeout=args.ollama_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            message = payload.get("message", {})
            content = message.get("content") or payload.get("response") or message.get("thinking") or ""
            parsed = parse_first_json(content)
            impact = extract_impact_score(parsed)
            if impact is not None:
                out.loc[idx, "ollama_impact_score_0_1"] = impact
                out.loc[idx, "ollama_direction"] = str(parsed.get("direction", "")).lower()[:40]
                out.loc[idx, "ollama_rationale"] = str(parsed.get("rationale", ""))[:600]
                out.loc[idx, "ollama_status"] = "scored"
                out.loc[idx, "final_impact_score_0_1"] = 0.6 * impact + 0.4 * out.loc[idx, "rule_impact_score_0_1"]
            else:
                out.loc[idx, "ollama_rationale"] = f"parse_failed: {content[:500]}"
            print(f"  Ollama classified {n}/{len(candidate_idx)} event rows", flush=True)
            time.sleep(args.ollama_sleep)
        except Exception as exc:
            out.loc[idx, "ollama_rationale"] = f"ollama_error: {exc}"
            out.loc[idx, "ollama_status"] = "error"

    out["impact_bucket"] = pd.cut(
        out["final_impact_score_0_1"],
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.001],
        labels=["minimal", "weak", "moderate", "strong", "severe"],
    ).astype(str)
    return out


def ollama_available(base_url: str, headers: dict) -> bool:
    try:
        response = requests.get(f"{base_url}/api/tags", headers=headers, timeout=5)
        return response.status_code < 500
    except Exception:
        return False


def build_ollama_prompt(row: pd.Series) -> str:
    return (
        "Score this GDELT event for likely one-week impact on ICE Coffee C Arabica futures. "
        "Return only JSON with keys impact_score_0_1, direction, rationale. "
        "The rationale must be a brief reason, not the word short. "
        "0=no impact, 1=severe plausible impact. "
        f"Event: {row.get('event_summary')}. "
        f"Country={row.get('primary_country')}; root={row.get('EventRootCode')}; quad={row.get('QuadClass')}; "
        f"mentions={row.get('NumMentions')}; articles={row.get('NumArticles')}; tone={row.get('AvgTone')}; "
        f"next_5d_coffee_return={row.get('future_return_5d')}; rule_score={row.get('rule_impact_score_0_1'):.3f}."
    )


def parse_first_json(text: str) -> dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def extract_impact_score(parsed: dict) -> float | None:
    for key in [
        "impact_score_0_1",
        "impact_score",
        "severity_score",
        "severity",
        "score",
        "coffee_c_impact_score",
        "coffee_impact_score",
    ]:
        if key in parsed:
            return clip_float(parsed.get(key), 0.0, 1.0)
    return None


def clip_float(value, low: float, high: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return float(min(high, max(low, number)))


def select_top_events(events: pd.DataFrame, target_dates: dict, top_per_target: int) -> pd.DataFrame:
    if not target_dates["target_dates"]:
        return events.sort_values("final_impact_score_0_1", ascending=False).reset_index(drop=True)

    selected_idx = set()
    if "ollama_status" in events.columns:
        selected_idx.update(
            events.loc[events["ollama_status"].isin(["scored", "parse_failed", "error"])].index.tolist()
        )
    for target_date in target_dates["target_dates"]:
        start = pd.Timestamp(target_date) - pd.Timedelta(days=target_dates["before_days"])
        end = pd.Timestamp(target_date) + pd.Timedelta(days=target_dates["after_days"])
        window = events.loc[events["event_date"].between(start, end)]
        selected_idx.update(window.sort_values("final_impact_score_0_1", ascending=False).head(top_per_target).index.tolist())
    return events.loc[sorted(selected_idx)].sort_values(["event_date", "final_impact_score_0_1"], ascending=[True, False]).reset_index(drop=True)


def build_daily_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    frame = events.copy()
    frame["signed_impact_score"] = frame["final_impact_score_0_1"] * frame["impact_direction"].map(
        {"bullish": 1, "bearish": -1, "neutral": 0}
    ).fillna(0)
    grouped = (
        frame.groupby("trading_date", dropna=False)
        .agg(
            event_count=("GLOBALEVENTID", "count"),
            max_impact_score_0_1=("final_impact_score_0_1", "max"),
            mean_impact_score_0_1=("final_impact_score_0_1", "mean"),
            signed_impact_score=("signed_impact_score", "sum"),
            bullish_events=("impact_direction", lambda x: int((x == "bullish").sum())),
            bearish_events=("impact_direction", lambda x: int((x == "bearish").sum())),
            neutral_events=("impact_direction", lambda x: int((x == "neutral").sum())),
            Close=("Close", "first"),
            future_return_5d=("future_return_5d", "first"),
            future_return_20d=("future_return_20d", "first"),
            top_event_summary=("event_summary", lambda x: " | ".join(x.head(3))),
        )
        .reset_index()
    )
    return grouped.sort_values("trading_date")


def build_price_point_summary(events: pd.DataFrame, target_dates: dict, price: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for target_date in target_dates["target_dates"]:
        start = pd.Timestamp(target_date) - pd.Timedelta(days=target_dates["before_days"])
        end = pd.Timestamp(target_date) + pd.Timedelta(days=target_dates["after_days"])
        if events.empty:
            window = pd.DataFrame()
        else:
            window = events.loc[events["event_date"].between(start, end)].sort_values("final_impact_score_0_1", ascending=False)
        price_row = target_price_row(price, pd.Timestamp(target_date)) if price is not None else {}
        rows.append(
            {
                "target_price_date": pd.Timestamp(target_date).date().isoformat(),
                **price_row,
                "window_start": start.date().isoformat(),
                "window_end": end.date().isoformat(),
                "event_count": int(len(window)),
                "max_impact_score_0_1": float(window["final_impact_score_0_1"].max()) if len(window) else 0.0,
                "mean_impact_score_0_1": float(window["final_impact_score_0_1"].mean()) if len(window) else 0.0,
                "top_event_date": window["event_date"].iloc[0].date().isoformat() if len(window) else "",
                "top_event_direction": window["impact_direction"].iloc[0] if len(window) else "",
                "top_event_summary": window["event_summary"].iloc[0] if len(window) else "",
            }
        )
    return pd.DataFrame(rows)


def target_price_row(price: pd.DataFrame, target_date: pd.Timestamp) -> dict:
    frame = price.loc[price["Date"].ge(target_date)].head(1)
    if frame.empty:
        return {
            "target_trading_date": "",
            "target_close": np.nan,
            "target_future_return_5d": np.nan,
            "target_future_return_20d": np.nan,
        }
    row = frame.iloc[0]
    out = {
        "target_trading_date": row["Date"].date().isoformat(),
        "target_close": float(row["Close"]) if pd.notna(row["Close"]) else np.nan,
        "target_future_return_5d": float(row["future_return_5d"]) if pd.notna(row["future_return_5d"]) else np.nan,
        "target_future_return_20d": float(row["future_return_20d"]) if pd.notna(row["future_return_20d"]) else np.nan,
    }
    for col in COT_CONTEXT_COLUMNS:
        if col in row.index:
            out[f"target_{col}"] = float(row[col]) if pd.notna(row[col]) else np.nan
    return out


def write_empty_outputs(year: int, output_dir: Path, filtered_dir: Path, plots_dir: Path, target_dates: dict) -> None:
    detail_path = output_dir / f"gdelt_coffee_event_impact_{year}.csv"
    candidates_path = output_dir / f"gdelt_coffee_event_candidates_{year}.csv"
    daily_path = output_dir / f"gdelt_coffee_event_daily_impact_{year}.csv"
    points_path = output_dir / f"gdelt_coffee_event_price_points_{year}.csv"
    filtered_extract_path = filtered_dir / f"gdelt_coffee_events_{year}_filtered.csv"
    metadata_path = output_dir / f"gdelt_coffee_event_impact_metadata_{year}.json"
    empty = pd.DataFrame()
    empty.to_csv(detail_path, index=False)
    empty.to_csv(candidates_path, index=False)
    empty.to_csv(daily_path, index=False)
    empty.to_csv(filtered_extract_path, index=False)
    build_price_point_summary(empty, target_dates).to_csv(points_path, index=False)
    metadata_path.write_text(json.dumps({"year": year, "rows_after_filtering": 0}, indent=2))


def write_plots(events: pd.DataFrame, daily: pd.DataFrame, price: pd.DataFrame, target_dates: dict, plots_dir: Path, year: int) -> None:
    if events.empty or daily.empty:
        return

    daily = daily.copy()
    daily["trading_date"] = pd.to_datetime(daily["trading_date"])
    price_year = price.loc[price["Date"].dt.year.eq(year)].copy()

    fig, ax1 = plt.subplots(figsize=(15, 6.5))
    ax1.plot(price_year["Date"], price_year["Close"], color="#264653", linewidth=1.8, label="Coffee C close")
    ax1.set_ylabel("Coffee C close")
    ax2 = ax1.twinx()
    ax2.bar(
        daily["trading_date"],
        daily["max_impact_score_0_1"],
        width=2.5,
        color="#e76f51",
        alpha=0.42,
        label="Max event impact score",
    )
    ax2.set_ylabel("Event impact score 0-1")
    for target_date in target_dates["target_dates"]:
        ax1.axvline(pd.Timestamp(target_date), color="#2a9d8f", linestyle="--", linewidth=1.2, alpha=0.75)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title(f"GDELT Coffee-Relevant Event Impact Vs Coffee C Price, {year}")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(plots_dir / f"gdelt_coffee_event_impact_timeline_{year}.png", dpi=180)
    plt.close(fig)

    plot_events = events.dropna(subset=["future_return_5d"]).copy()
    if not plot_events.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=plot_events,
            x="final_impact_score_0_1",
            y="future_return_5d",
            hue="impact_direction",
            size="NumMentions",
            sizes=(30, 180),
            alpha=0.72,
        )
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.title(f"Event Impact Score Vs Next 5D Coffee C Return, {year}")
        plt.xlabel("Final event impact score 0-1")
        plt.ylabel("Next 5 trading day return")
        plt.tight_layout()
        plt.savefig(plots_dir / f"gdelt_coffee_event_score_vs_5d_return_{year}.png", dpi=180)
        plt.close()

    top_daily = daily.sort_values("max_impact_score_0_1", ascending=False).head(15).copy()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_daily, x=top_daily["trading_date"].dt.strftime("%Y-%m-%d"), y="max_impact_score_0_1", color="#457b9d")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top Coffee-Relevant GDELT Event Dates, {year}")
    plt.xlabel("Mapped Coffee C trading date")
    plt.ylabel("Max impact score 0-1")
    plt.tight_layout()
    plt.savefig(plots_dir / f"gdelt_coffee_event_top_dates_{year}.png", dpi=180)
    plt.close()

    if target_dates["target_dates"]:
        windows = []
        for target_date in target_dates["target_dates"]:
            start = pd.Timestamp(target_date) - pd.Timedelta(days=target_dates["before_days"])
            end = pd.Timestamp(target_date) + pd.Timedelta(days=target_dates["after_days"])
            frame = daily.loc[daily["trading_date"].between(start, end)].copy()
            frame["target_price_date"] = pd.Timestamp(target_date).date().isoformat()
            windows.append(frame)
        if windows:
            window_frame = pd.concat(windows, ignore_index=True)
            if not window_frame.empty:
                plt.figure(figsize=(13, 6))
                sns.barplot(
                    data=window_frame,
                    x=window_frame["trading_date"].dt.strftime("%Y-%m-%d"),
                    y="max_impact_score_0_1",
                    hue="target_price_date",
                )
                plt.xticks(rotation=45, ha="right")
                plt.title(f"Event Impact Around Yahoo Price Drift Dates, {year}")
                plt.xlabel("Mapped trading date")
                plt.ylabel("Max impact score 0-1")
                plt.tight_layout()
                plt.savefig(plots_dir / f"gdelt_coffee_event_price_point_windows_{year}.png", dpi=180)
                plt.close()


if __name__ == "__main__":
    main()
