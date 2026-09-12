from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urljoin
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
sys.path.append(str(SCRIPT_DIR))

from gdelt_coffee_event_impact import (  # noqa: E402
    COFFEE_COUNTRY_CODES,
    DEFAULT_PRICE_PATH,
    EVENT_USECOLS,
    GDELT_COLUMNS,
    attach_price_context,
    build_daily_summary,
    build_event_summary,
    clean_value,
    infer_impact_direction,
    keep_coffee_relevant,
    load_price_data,
    score_events,
)
from gdelt_weekly_ollama_batch_score import (  # noqa: E402
    add_deterministic_period_score,
    add_period_columns,
    build_period_summary,
)

INDEX_URL = "https://data.gdeltproject.org/events/index.html"
OUTPUT_DIR = ROOT / "Scripts" / "project" / "artifacts" / "outputs"
EVENT_DIR = ROOT / "data" / "events"
TMP_DIR = Path("/private/tmp/arabica_gdelt_downloads")

FINAL_COLUMNS = [
    "GLOBALEVENTID",
    "SQLDATE",
    "event_date",
    "trading_date",
    "event_to_trade_lag_days",
    "Year",
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
    "keyword_relevant",
    "coffee_country_relevant",
    "coffee_country_name_relevant",
    "coffee_region_relevant",
    "primary_country",
    "relevance_mode",
    "event_summary",
    "impact_direction",
    "impact_bucket",
    "relevance_score_0_1",
    "event_intensity_score_0_1",
    "price_response_score_0_1",
    "direction_alignment_score_0_1",
    "rule_impact_score_0_1",
    "final_impact_score_0_1",
    "Close",
    "Volume",
    "return_1d",
    "future_return_1d",
    "future_return_5d",
    "future_return_10d",
    "future_return_20d",
    "realized_vol_20d",
    "Open_Interest_All",
    "managed_money_net",
    "commercial_net",
    "noncommercial_net",
    "managed_money_weekly_net_change",
    "commercial_weekly_net_change",
    "noncommercial_weekly_net_change",
    "open_interest_change_pct",
    "source_archive",
    "source_url",
]

NUMERIC_FOR_STRICT = [
    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",
    "QuadClass",
]

EVENT_USECOL_INDICES = [GDELT_COLUMNS.index(column) for column in EVENT_USECOLS]
COFFEE_REGION_TERMS = [
    "minas gerais",
    "espirito santo",
    "espírito santo",
    "sao paulo",
    "são paulo",
    "parana",
    "paraná",
    "cerrado mineiro",
    "mogiana",
    "sul de minas",
    "huila",
    "tolima",
    "antioquia",
    "caldas",
    "risaralda",
    "quindio",
    "quindío",
    "dak lak",
    "dak nong",
    "lam dong",
    "gia lai",
    "central highlands",
    "sidamo",
    "sidama",
    "yirgacheffe",
    "jimma",
    "kaffa",
    "mbale",
    "bugisu",
    "santa rosa",
    "huehuetenango",
    "tarrazu",
    "tarrazú",
    "matagalpa",
]


@dataclass
class ArchiveRef:
    name: str
    url: str
    kind: str
    start_date: str
    end_date: str
    size_text: str = ""
    md5: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download GDELT event archives one at a time, keep only Coffee C-relevant scored rows, "
            "delete each zip, and write all-year daily/weekly model feature tables."
        )
    )
    parser.add_argument("--index-url", default=INDEX_URL)
    parser.add_argument("--price-path", default=str(DEFAULT_PRICE_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--event-dir", default=str(EVENT_DIR))
    parser.add_argument("--tmp-dir", default=str(TMP_DIR))
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--max-date-lag-days", type=int, default=5)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--limit-archives", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of archives to download/process concurrently. The main thread still serializes "
            "CSV appends and manifest updates to keep resume state safe."
        ),
    )
    parser.add_argument("--filter-mode", choices=["strict", "broad"], default="strict")
    parser.add_argument(
        "--max-rows-per-trading-day",
        type=int,
        default=50,
        help="Storage guard: keep only the top N scored rows per mapped Coffee C trading day per archive. Use 0 to keep all.",
    )
    parser.add_argument("--include-master", action="store_true", help="Also process the 1979-2013 master archive.")
    parser.add_argument(
        "--skip-generated-historical",
        action="store_true",
        help="Do not add generated full historical yearly/monthly archives for 1979-2013.",
    )
    parser.add_argument("--only-daily", action="store_true", help="Skip non-daily archives even if --include-master is set.")
    parser.add_argument("--local-zip-dir", default=str(ROOT / "Scripts" / "EventsData" / "Zip"))
    parser.add_argument("--prefer-local-zips", action="store_true", help="Use matching local zips before downloading.")
    parser.add_argument("--keep-temp-zip", action="store_true", help="Debug only: do not delete downloaded temp zips.")
    parser.add_argument("--force", action="store_true", help="Reprocess archives already marked complete.")
    parser.add_argument("--rebuild-output", action="store_true", help="Remove prior filtered output and manifest before running.")
    parser.add_argument("--list-only", action="store_true", help="List selected archives and exit without downloading.")
    parser.add_argument("--user-agent", default="ArabicaFuturesCoffeeResearch/1.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    event_dir = resolve_path(args.event_dir)
    tmp_dir = resolve_path(args.tmp_dir)
    local_zip_dir = resolve_path(args.local_zip_dir)
    for path in [output_dir, event_dir, tmp_dir]:
        path.mkdir(parents=True, exist_ok=True)

    price_raw = pd.read_csv(resolve_path(args.price_path), usecols=["Date", "Close"], parse_dates=["Date"], low_memory=False)
    price_raw = price_raw.loc[price_raw["Close"].notna() & price_raw["Date"].notna()].sort_values("Date")
    if price_raw.empty:
        raise RuntimeError("No Yahoo price rows with Close were found; cannot map GDELT events to Coffee C prices.")

    start_year = args.start_year or int(price_raw["Date"].dt.year.min())
    end_year = args.end_year or int(price_raw["Date"].dt.year.max())
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = min(pd.Timestamp(f"{end_year}-12-31"), price_raw["Date"].max())

    price = load_price_data(resolve_path(args.price_path), start_year)
    price = price.loc[price["Date"].between(start_date, end_date + pd.Timedelta(days=args.max_date_lag_days))].copy()

    output_path = event_dir / f"gdelt_coffee_events_{start_year}_{end_year}_filtered_scored.csv"
    daily_path = output_dir / f"gdelt_coffee_events_{start_year}_{end_year}_daily_summary.csv"
    weekly_path = output_dir / f"gdelt_coffee_events_{start_year}_{end_year}_weekly_summary.csv"
    manifest_path = output_dir / f"gdelt_coffee_events_{start_year}_{end_year}_processing_manifest.json"

    if args.rebuild_output:
        for path in [output_path, daily_path, weekly_path, manifest_path]:
            if path.exists():
                path.unlink()

    manifest = load_manifest(manifest_path, start_year, end_year)
    seen_ids = load_seen_ids(output_path)

    archives = discover_archives(
        args.index_url,
        args.user_agent,
        include_generated_historical=not args.skip_generated_historical,
    )
    archives = select_archives(archives, start_date, end_date, include_master=args.include_master, only_daily=args.only_daily)
    if args.limit_archives:
        archives = archives[: args.limit_archives]

    print(
        f"Selected {len(archives):,} GDELT archive(s) for {start_date.date()} to {end_date.date()} "
        f"using {args.filter_mode} relevance.",
        flush=True,
    )
    print(f"Filtered scored table: {output_path}", flush=True)
    if args.list_only:
        if archives:
            print(f"First archive: {archives[0].name} ({archives[0].start_date})", flush=True)
            print(f"Last archive: {archives[-1].name} ({archives[-1].start_date})", flush=True)
            print(f"Historical master included: {any(item.kind == 'master' for item in archives)}", flush=True)
        return

    completed = {item["name"] for item in manifest.get("completed_archives", [])}
    pending_archives = [archive for archive in archives if args.force or archive.name not in completed]
    total_rows = process_archives(
        pending_archives=pending_archives,
        all_archive_count=len(archives),
        args=args,
        start_date=start_date,
        end_date=end_date,
        price=price,
        output_path=output_path,
        tmp_dir=tmp_dir,
        local_zip_dir=local_zip_dir,
        seen_ids=seen_ids,
        manifest=manifest,
        manifest_path=manifest_path,
        daily_path=daily_path,
        weekly_path=weekly_path,
    )

    build_summary_outputs(output_path, daily_path, weekly_path)
    manifest["final_rows"] = int(count_csv_rows(output_path))
    manifest["last_updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    write_json(manifest_path, manifest)

    print(f"Run added {total_rows:,} row(s).", flush=True)
    print(f"Saved filtered scored table: {output_path}", flush=True)
    print(f"Saved daily summary: {daily_path}", flush=True)
    print(f"Saved weekly summary: {weekly_path}", flush=True)
    print(f"Saved manifest: {manifest_path}", flush=True)


def process_archives(
    pending_archives: list[ArchiveRef],
    all_archive_count: int,
    args: argparse.Namespace,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    price: pd.DataFrame,
    output_path: Path,
    tmp_dir: Path,
    local_zip_dir: Path,
    seen_ids: set[str],
    manifest: dict,
    manifest_path: Path,
    daily_path: Path,
    weekly_path: Path,
) -> int:
    if not pending_archives:
        return 0

    workers = max(1, int(args.workers or 1))
    if workers == 1:
        return process_archives_sequential(
            pending_archives,
            all_archive_count,
            args,
            start_date,
            end_date,
            price,
            output_path,
            tmp_dir,
            local_zip_dir,
            seen_ids,
            manifest,
            manifest_path,
            daily_path,
            weekly_path,
        )

    total_rows = 0
    submitted = 0
    completed_count = 0
    archive_iter = iter(enumerate(pending_archives, start=1))
    futures = {}
    print(f"Processing with {workers} worker(s).", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            while len(futures) < workers:
                try:
                    archive_no, archive = next(archive_iter)
                except StopIteration:
                    break
                submitted += 1
                print(
                    f"[queued {archive_no:,}/{len(pending_archives):,}; selected {all_archive_count:,}] {archive.name}",
                    flush=True,
                )
                future = executor.submit(
                    process_archive,
                    archive=archive,
                    args=args,
                    start_date=start_date,
                    end_date=end_date,
                    price=price,
                    tmp_dir=tmp_dir,
                    local_zip_dir=local_zip_dir,
                )
                futures[future] = archive

            if not futures:
                break

            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                archive = futures.pop(future)
                completed_count += 1
                try:
                    result, output_rows = future.result()
                except Exception as exc:
                    result = {
                        **asdict(archive),
                        "status": "failed",
                        "error": str(exc)[:1000],
                        "processed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                    }
                    manifest.setdefault("failures", []).append(result)
                    manifest["last_updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
                    write_json(manifest_path, manifest)
                    print(f"[done {completed_count:,}/{len(pending_archives):,}] {archive.name} failed: {exc}", flush=True)
                    continue

                output_rows = drop_seen(output_rows, seen_ids)
                if not output_rows.empty:
                    append_csv(output_rows, output_path)
                result["rows_kept"] = int(len(output_rows))
                total_rows += result["rows_kept"]
                update_manifest_after_archive(
                    manifest,
                    manifest_path,
                    archive,
                    result,
                    output_path,
                    daily_path,
                    weekly_path,
                    args.filter_mode,
                )
                print(
                    f"[done {completed_count:,}/{len(pending_archives):,}] {archive.name}: "
                    f"kept {result['rows_kept']:,} row(s)",
                    flush=True,
                )
                time.sleep(args.sleep)

    print(f"Submitted {submitted:,} archive(s) to workers.", flush=True)
    return total_rows


def process_archives_sequential(
    pending_archives: list[ArchiveRef],
    all_archive_count: int,
    args: argparse.Namespace,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    price: pd.DataFrame,
    output_path: Path,
    tmp_dir: Path,
    local_zip_dir: Path,
    seen_ids: set[str],
    manifest: dict,
    manifest_path: Path,
    daily_path: Path,
    weekly_path: Path,
) -> int:
    total_rows = 0
    for archive_no, archive in enumerate(pending_archives, start=1):
        print(f"[{archive_no:,}/{len(pending_archives):,}; selected {all_archive_count:,}] {archive.name}", flush=True)
        try:
            result, output_rows = process_archive(
                archive=archive,
                args=args,
                start_date=start_date,
                end_date=end_date,
                price=price,
                tmp_dir=tmp_dir,
                local_zip_dir=local_zip_dir,
            )
        except Exception as exc:
            result = {
                **asdict(archive),
                "status": "failed",
                "error": str(exc)[:1000],
                "processed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            manifest.setdefault("failures", []).append(result)
            manifest["last_updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
            write_json(manifest_path, manifest)
            print(f"  failed: {exc}", flush=True)
            continue

        output_rows = drop_seen(output_rows, seen_ids)
        if not output_rows.empty:
            append_csv(output_rows, output_path)
        result["rows_kept"] = int(len(output_rows))
        total_rows += result["rows_kept"]
        update_manifest_after_archive(
            manifest,
            manifest_path,
            archive,
            result,
            output_path,
            daily_path,
            weekly_path,
            args.filter_mode,
        )
        time.sleep(args.sleep)
    return total_rows


def update_manifest_after_archive(
    manifest: dict,
    manifest_path: Path,
    archive: ArchiveRef,
    result: dict,
    output_path: Path,
    daily_path: Path,
    weekly_path: Path,
    filter_mode: str,
) -> None:
    manifest["completed_archives"] = [item for item in manifest.get("completed_archives", []) if item["name"] != archive.name]
    manifest["completed_archives"].append({**asdict(archive), **result})
    manifest["failures"] = [item for item in manifest.get("failures", []) if item.get("name") != archive.name]
    manifest["last_updated_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    manifest["output_path"] = str(output_path)
    manifest["daily_summary_path"] = str(daily_path)
    manifest["weekly_summary_path"] = str(weekly_path)
    manifest["filter_mode"] = filter_mode
    write_json(manifest_path, manifest)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_manifest(path: Path, start_year: int, end_year: int) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "source": INDEX_URL,
        "start_year": start_year,
        "end_year": end_year,
        "completed_archives": [],
        "failures": [],
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def load_seen_ids(output_path: Path) -> set[str]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    try:
        ids = pd.read_csv(output_path, usecols=["GLOBALEVENTID"], dtype=str, low_memory=False)["GLOBALEVENTID"]
    except Exception:
        return set()
    return set(ids.dropna().astype(str))


def discover_archives(index_url: str, user_agent: str, include_generated_historical: bool) -> list[ArchiveRef]:
    response = requests.get(index_url, headers={"User-Agent": user_agent}, timeout=60)
    response.raise_for_status()
    html = response.text
    archive_refs = []
    archive_pattern = r'href="([^"]+\.zip)"[^>]*>\s*([^<]+)</a>\s*(?:\(([^)]*)\))?\s*(?:\(MD5:\s*([0-9a-fA-F]+)\))?'
    for match in re.finditer(archive_pattern, html, flags=re.I):
        href, label, size_text, md5 = match.groups()
        name = Path(label.strip() or href).name
        ref = classify_archive(name=name, url=urljoin(index_url, href), size_text=size_text or "", md5=md5 or "")
        if ref is not None:
            archive_refs.append(ref)
    if include_generated_historical:
        archive_refs.extend(generate_full_historical_archives(index_url))
    archive_refs = dedupe_archives(archive_refs)
    archive_refs = sorted(archive_refs, key=lambda item: (item.start_date, item.name))
    return archive_refs


def generate_full_historical_archives(index_url: str) -> list[ArchiveRef]:
    refs = []
    for year in range(1979, 2006):
        name = f"{year}.zip"
        refs.append(
            ArchiveRef(
                name=name,
                url=urljoin(index_url, name),
                kind="yearly",
                start_date=f"{year}-01-01",
                end_date=f"{year}-12-31",
            )
        )
    for period in pd.period_range("2006-01", "2013-03", freq="M"):
        name = f"{period.year}{period.month:02d}.zip"
        refs.append(
            ArchiveRef(
                name=name,
                url=urljoin(index_url, name),
                kind="monthly",
                start_date=period.start_time.date().isoformat(),
                end_date=period.end_time.date().isoformat(),
            )
        )
    return refs


def dedupe_archives(archives: list[ArchiveRef]) -> list[ArchiveRef]:
    by_name = {}
    for archive in archives:
        if archive.name not in by_name:
            by_name[archive.name] = archive
    return list(by_name.values())


def classify_archive(name: str, url: str, size_text: str, md5: str) -> ArchiveRef | None:
    daily = re.match(r"(?P<date>\d{8})\.export\.CSV\.zip$", name)
    if daily:
        date_text = daily.group("date")
        date = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
        if pd.isna(date):
            return None
        iso = date.date().isoformat()
        return ArchiveRef(name=name, url=url, kind="daily", start_date=iso, end_date=iso, size_text=size_text, md5=md5)

    master = re.search(r"(?P<start>\d{4})-(?P<end>\d{4})", name)
    if "MASTER" in name.upper() and master:
        start_year = int(master.group("start"))
        end_year = int(master.group("end"))
        return ArchiveRef(
            name=name,
            url=url,
            kind="master",
            start_date=f"{start_year}-01-01",
            end_date=f"{end_year}-12-31",
            size_text=size_text,
            md5=md5,
        )

    yearly = re.match(r"(?P<year>\d{4})\.zip$", name)
    if yearly:
        year = int(yearly.group("year"))
        return ArchiveRef(
            name=name,
            url=url,
            kind="yearly",
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            size_text=size_text,
            md5=md5,
        )
    return None


def select_archives(
    archives: list[ArchiveRef],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    include_master: bool,
    only_daily: bool,
) -> list[ArchiveRef]:
    selected = []
    for archive in archives:
        archive_start = pd.Timestamp(archive.start_date)
        archive_end = pd.Timestamp(archive.end_date)
        if archive_end < start_date or archive_start > end_date:
            continue
        if archive.kind == "master" and (only_daily or not include_master):
            continue
        if archive.kind != "daily" and only_daily:
            continue
        if include_master and archive.kind == "daily" and archive_end.year <= 2013:
            continue
        selected.append(archive)
    return selected


def process_archive(
    archive: ArchiveRef,
    args: argparse.Namespace,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    price: pd.DataFrame,
    tmp_dir: Path,
    local_zip_dir: Path,
) -> tuple[dict, pd.DataFrame]:
    archive_path = local_zip_dir / archive.name if args.prefer_local_zips and (local_zip_dir / archive.name).exists() else None
    downloaded = False
    if archive_path is None:
        archive_path = tmp_dir / archive.name
        download_archive(archive.url, archive_path, args.user_agent, args.timeout)
        downloaded = True

    rows_processed = 0
    rows_relevant = 0
    rows_scored_before_cap = 0
    day_buffers: dict[str, pd.DataFrame] = {}
    try:
        with ZipFile(archive_path) as zip_file:
            members = [name for name in zip_file.namelist() if name.lower().endswith(".csv")]
            if not members:
                raise RuntimeError(f"No CSV member found in {archive_path}")
            for member in members:
                with zip_file.open(member) as handle:
                    reader = pd.read_csv(
                        handle,
                        sep="\t",
                        header=None,
                        names=EVENT_USECOLS,
                        usecols=EVENT_USECOL_INDICES,
                        dtype=str,
                        chunksize=args.chunksize,
                        low_memory=False,
                    )
                    for chunk in reader:
                        rows_processed += len(chunk)
                        chunk = filter_date_range(chunk, start_date, end_date)
                        if chunk.empty:
                            continue
                        chunk = keep_coffee_relevant(chunk)
                        if chunk.empty:
                            continue
                        chunk = apply_relevance_mode(chunk, args.filter_mode)
                        if chunk.empty:
                            continue
                        rows_relevant += len(chunk)
                        scored = attach_price_context(chunk, price, args.max_date_lag_days)
                        scored = scored.loc[scored["trading_date"].notna()].copy()
                        if scored.empty:
                            continue
                        scored = score_events(scored, price)
                        scored["source_archive"] = archive.name
                        scored["source_url"] = archive.url
                        scored["Year"] = pd.to_datetime(scored["event_date"], errors="coerce").dt.year
                        scored = finalize_columns(scored)
                        if scored.empty:
                            continue
                        rows_scored_before_cap += len(scored)
                        update_day_buffers(scored, day_buffers, args.max_rows_per_trading_day)
        output_rows = combine_day_buffers(day_buffers)
        return {
            "rows_processed": int(rows_processed),
            "rows_relevant_before_price_map": int(rows_relevant),
            "rows_scored_before_daily_cap": int(rows_scored_before_cap),
            "rows_kept_before_global_dedupe": int(len(output_rows)),
            "rows_kept": int(len(output_rows)),
            "max_rows_per_trading_day": int(args.max_rows_per_trading_day),
            "status": "complete",
            "processed_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }, output_rows
    finally:
        if downloaded and archive_path.exists() and not args.keep_temp_zip:
            archive_path.unlink()


def download_archive(url: str, path: Path, user_agent: str, timeout: float) -> None:
    part_path = path.with_suffix(path.suffix + ".part")
    if part_path.exists():
        part_path.unlink()
    with requests.get(url, headers={"User-Agent": user_agent}, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        next_report = 100 * 1024 * 1024
        if total:
            print(f"  downloading {path.name}: {total / (1024 * 1024):.1f} MB", flush=True)
        else:
            print(f"  downloading {path.name}", flush=True)
        with part_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_report:
                        if total:
                            print(
                                f"  downloaded {downloaded / (1024 * 1024):.0f}/{total / (1024 * 1024):.0f} MB",
                                flush=True,
                            )
                        else:
                            print(f"  downloaded {downloaded / (1024 * 1024):.0f} MB", flush=True)
                        next_report += 100 * 1024 * 1024
    part_path.replace(path)


def filter_date_range(chunk: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(chunk["SQLDATE"], format="%Y%m%d", errors="coerce")
    mask = dates.between(start_date, end_date)
    return chunk.loc[mask].copy()


def apply_relevance_mode(chunk: pd.DataFrame, filter_mode: str) -> pd.DataFrame:
    out = chunk.copy()
    out["coffee_region_relevant"] = coffee_region_match(out)
    out["relevance_mode"] = "broad"
    if filter_mode == "broad":
        return out

    for col in NUMERIC_FOR_STRICT:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    event_root = out["EventRootCode"].fillna("").astype(str).str.zfill(2)
    producer_geo_relevant = (
        out[["Actor1Geo_CountryCode", "Actor2Geo_CountryCode", "ActionGeo_CountryCode"]]
        .fillna("")
        .isin(COFFEE_COUNTRY_CODES)
        .any(axis=1)
    )
    severe_supply_risk = (
        event_root.isin({"14", "17", "18", "19", "20"})
        | out["GoldsteinScale"].le(-7.0)
    ) & (
        out["NumArticles"].fillna(0).ge(2)
        | out["NumMentions"].fillna(0).ge(5)
        | out["GoldsteinScale"].le(-6.0)
    )
    keep = (
        out["keyword_relevant"].astype(bool)
        | out["coffee_region_relevant"].astype(bool)
        | (producer_geo_relevant & severe_supply_risk)
    )
    out = out.loc[keep].copy()
    out["relevance_mode"] = "strict"
    return out


def coffee_region_match(frame: pd.DataFrame) -> pd.Series:
    text_cols = [
        "Actor1Name",
        "Actor2Name",
        "Actor1Code",
        "Actor2Code",
        "Actor1Geo_FullName",
        "Actor2Geo_FullName",
        "ActionGeo_FullName",
    ]
    text = frame[text_cols].fillna("").agg(" ".join, axis=1).str.lower()
    pattern = "|".join(re.escape(term) for term in COFFEE_REGION_TERMS)
    return text.str.contains(pattern, regex=True, na=False)


def finalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in FINAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    return out[FINAL_COLUMNS]


def update_day_buffers(frame: pd.DataFrame, day_buffers: dict[str, pd.DataFrame], max_rows: int) -> None:
    if frame.empty:
        return
    frame = frame.copy()
    frame["_trading_day_key"] = pd.to_datetime(frame["trading_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for day, group in frame.groupby("_trading_day_key", dropna=False):
        group = group.drop(columns=["_trading_day_key"])
        prior = day_buffers.get(str(day))
        combined = group if prior is None else pd.concat([prior, group], ignore_index=True)
        combined = combined.drop_duplicates("GLOBALEVENTID", keep="first")
        combined = combined.sort_values(
            ["keyword_relevant", "coffee_region_relevant", "final_impact_score_0_1", "NumArticles", "NumMentions"],
            ascending=[False, False, False, False, False],
        )
        if max_rows and max_rows > 0:
            combined = combined.head(max_rows)
        day_buffers[str(day)] = combined


def combine_day_buffers(day_buffers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not day_buffers:
        return pd.DataFrame(columns=FINAL_COLUMNS)
    return pd.concat(day_buffers.values(), ignore_index=True).sort_values(["trading_date", "final_impact_score_0_1"], ascending=[True, False])


def drop_seen(frame: pd.DataFrame, seen_ids: set[str]) -> pd.DataFrame:
    ids = frame["GLOBALEVENTID"].astype(str)
    mask = ~ids.isin(seen_ids)
    out = frame.loc[mask].copy()
    seen_ids.update(out["GLOBALEVENTID"].dropna().astype(str).tolist())
    return out


def append_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def build_summary_outputs(filtered_path: Path, daily_path: Path, weekly_path: Path) -> None:
    if not filtered_path.exists() or filtered_path.stat().st_size == 0:
        pd.DataFrame().to_csv(daily_path, index=False)
        pd.DataFrame().to_csv(weekly_path, index=False)
        return

    rows = pd.read_csv(filtered_path, low_memory=False, parse_dates=["event_date", "trading_date"])
    rows = coerce_summary_dtypes(rows)
    daily = build_daily_summary(rows)
    daily.to_csv(daily_path, index=False)

    weekly_rows = add_period_columns(rows, period="weekly", date_column="trading_date")
    weekly = build_period_summary(weekly_rows, top_events_per_period=8)
    weekly = add_deterministic_period_score(weekly)
    weekly.to_csv(weekly_path, index=False)


def coerce_summary_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    bool_cols = ["keyword_relevant", "coffee_country_relevant", "coffee_country_name_relevant", "coffee_region_relevant"]
    for col in bool_cols:
        if col in out:
            out[col] = out[col].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
    numeric_cols = [
        "GoldsteinScale",
        "NumMentions",
        "NumSources",
        "NumArticles",
        "AvgTone",
        "QuadClass",
        "Close",
        "Volume",
        "return_1d",
        "future_return_1d",
        "future_return_5d",
        "future_return_10d",
        "future_return_20d",
        "realized_vol_20d",
        "Open_Interest_All",
        "managed_money_net",
        "commercial_net",
        "noncommercial_net",
        "managed_money_weekly_net_change",
        "commercial_weekly_net_change",
        "noncommercial_weekly_net_change",
        "open_interest_change_pct",
        "relevance_score_0_1",
        "event_intensity_score_0_1",
        "price_response_score_0_1",
        "direction_alignment_score_0_1",
        "rule_impact_score_0_1",
        "final_impact_score_0_1",
    ]
    for col in numeric_cols:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["event_date", "trading_date"]:
        if col in out:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    if "event_summary" not in out:
        out["event_summary"] = out.apply(build_event_summary, axis=1)
    if "impact_direction" not in out:
        out["impact_direction"] = out.apply(infer_impact_direction, axis=1)
    return out


def count_csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    return max(0, line_count - 1)


if __name__ == "__main__":
    main()
