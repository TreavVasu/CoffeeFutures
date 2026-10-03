"""Rebuild Coffee C COT tables from the official CFTC Historical Compressed archives.

Discovery follows https://www.cftc.gov/MarketReports/CommitmentsofTraders/
HistoricalCompressed/index.htm, the page the supplied notebook used. For each
enabled report family the page is scraped for annual ZIP links, the archives are
downloaded, their tables extracted, and every row whose market name contains
``COFFEE C`` is retained.

``--verify``  Rebuild into scratch and compare against the committed tables in
              ``DATA/COT``: row count, numeric measurement fields and a value
              digest. Exits non-zero when they do not match. The committed
              cache is never modified.
``--write``   Replace ``DATA/COT`` with the rebuilt tables and write a manifest.

Publication-timing rules live in ``src/cot_release_features.py`` and are
independent of this script; only the source tables are rebuilt here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def find_data_dir(start: Path) -> Path:
    """Locate the shared DATA tree, which may sit beside or inside the repo root.

    The project was reorganised once, so the sibling ``../DATA`` layout that the
    original notebook assumed is not guaranteed. Probe both, plus one level up
    from the repository root, and fail loudly rather than writing anywhere wrong.
    """
    start = start.resolve()
    candidates = [start / "DATA", start / "data",
                  start.parent / "DATA", start.parent / "data",
                  start.parent.parent / "DATA", start.parent.parent / "data"]
    for candidate in candidates:
        if (candidate / "COT").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the DATA directory. Looked in: "
        + ", ".join(str(c) for c in candidates))


DATA = find_data_dir(ROOT)
COT_OUT = DATA / "COT"
RAW_DIR = DATA / "raw_cot_data"
ZIP_DIR = DATA / "cftc_cot"

CFTC_HISTORICAL_COMPRESSED_URL = (
    "https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm")
CFTC_HISTORICAL_VIEWABLE_URL = (
    "https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalViewable/index.htm")
MARKET_TOKENS = ("COFFEE C",)
USER_AGENT = "Mozilla/5.0 ArabicaFutures research downloader"
# The price history starts 2000-01-03, so reports before that cannot inform any
# modelled origin. The CFTC legacy family publishes two naming series; both are
# kept, but only from this year forward, matching the supplied notebook.
START_YEAR = 2000

REPORT_FAMILIES = {
    "legacy_futures_only": {
        "label": "Futures Only Reports - Legacy COT",
        "patterns": [r"/dea_fut_xls_(\d{4})\.zip$", r"/deafut_xls_(\d{4})\.zip$"],
        "bundles": [],
        "bundle_years": None,
    },
    "disaggregated_futures_only": {
        "label": "Disaggregated Futures Only",
        "patterns": [r"/fut_disagg_xls_(\d{4})\.zip$"],
        "bundles": [r"/fut_disagg_xls_hist_2006_2016\.zip$"],
        # The 2006-2016 bundle already contains those report dates. The CFTC page
        # also lists yearly archives for 2013-2016, which restate the same weeks;
        # taking both would double count them, so the bundle's range is skipped.
        "bundle_years": (2006, 2016),
    },
}


def fetch_links(page_url: str, session) -> list[str]:
    response = session.get(page_url, timeout=60, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    return sorted({urljoin(page_url, a["href"]) for a in soup.find_all("a", href=True)
                   if a["href"].endswith(".zip")})


def archive_name(url: str) -> str:
    return Path(urlparse(url).path).name


def archive_stem(name: str) -> str:
    """Match the committed naming: dea_fut_xls_2007.zip -> dea_fut_xls_2007."""
    for suffix in ("_annual_2007", "_annual"):
        if suffix in name:
            return name[: name.index(suffix)]
    return Path(name).stem


def select_archives(links: list[str], family: dict) -> dict[str, str]:
    """Map archive filename -> url, keeping yearly files and named history bundles.

    The CFTC page also lists pre-2000 legacy series, which cannot inform a
    modelled origin, so yearly archives are filtered to ``START_YEAR``. Any year
    already covered by a bundle in ``bundle_years`` is skipped, because the yearly
    archive restates weeks the bundle already carries.
    """
    covered = family.get("bundle_years")
    chosen: dict[str, str] = {}
    for url in links:
        path, name = urlparse(url).path, archive_name(url)
        if any(re.search(pattern, path) for pattern in family["bundles"]):
            chosen[name] = url
            continue
        for pattern in family["patterns"]:
            match = re.search(pattern, path)
            if not match:
                continue
            year = int(match.group(1))
            if year < START_YEAR:
                break
            if covered and covered[0] <= year <= covered[1]:
                break
            chosen[name] = url
            break
    return chosen


def ensure_archive(name: str, url: str, destination: Path, session) -> Path:
    if destination.exists() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {name}", flush=True)
    response = session.get(url, timeout=180, stream=True, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    with open(destination, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 16):
            if chunk:
                handle.write(chunk)
    return destination


def extract(name: str, archive: Path, family: str, target_root: Path) -> list[Path]:
    """Extract one archive. CFTC yearly members share a filename, so the archive
    name prefixes the directory to stop them overwriting each other.

    Some bundles, notably ``fut_disagg_xls_hist_2006_2016``, hold two tables for
    disjoint report windows. Return every table so neither half is dropped.
    """
    out = target_root / family / archive_stem(name)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(out)
    tables = sorted(p for p in out.rglob("*")
                    if p.suffix.lower() in {".xls", ".xlsx", ".csv", ".txt"})
    if not tables:
        raise ValueError(f"No readable table in {archive.name}")
    return tables


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".txt":
        return pd.read_csv(path, encoding="latin-1", low_memory=False)
    return pd.read_excel(path)


def is_coffee(frame: pd.DataFrame) -> pd.Series:
    """Boolean mask for rows whose market name identifies Coffee C.

    The CFTC tables carry the contract in ``Market_and_Exchange_Names`` (legacy)
    and ``market_and_exchange_names`` (disaggregated). Probe case-insensitively
    for the token rather than trusting a fixed column name, and refuse to guess
    when no such column exists.
    """
    for column in frame.columns:
        if "market" not in str(column).lower() or "name" not in str(column).lower():
            continue
        values = frame[column].astype(str).str.upper()
        mask = values.str.contains("COFFEE C", regex=False)
        if mask.any():
            return mask
    raise ValueError(
        "No market-and-exchange-name column found; refusing to guess the contract")


def rebuild(target_root: Path, session) -> pd.DataFrame:
    links = fetch_links(CFTC_HISTORICAL_COMPRESSED_URL, session)
    print(f"discovered {len(links)} archives on the CFTC Historical Compressed page", flush=True)
    records = []
    for family, spec in REPORT_FAMILIES.items():
        archives = select_archives(links, spec)
        print(f"{family}: {len(archives)} archives", flush=True)
        for name, url in sorted(archives.items()):
            zip_path = ensure_archive(name, url, ZIP_DIR / family / name, session)
            for table in extract(name, zip_path, family, target_root):
                frame = read_table(table)
                kept = frame.loc[is_coffee(frame)].copy()
                if kept.empty:
                    print(f"    {name}/{table.name}: no COFFEE C rows", flush=True)
                    continue
                kept["source_file"] = str(table.relative_to(target_root))
                kept["source_dataset"] = family
                kept["source_archive"] = archive_stem(name)
                records.append(kept)
                print(f"    {name}/{table.name}: {len(frame)} rows -> {len(kept)} COFFEE C",
                      flush=True)
            time.sleep(0.2)
    combined = pd.concat(records, ignore_index=True)
    return align_measurement_dtypes(combined)


def align_measurement_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Force the shared measurement columns to numeric.

    The legacy and disaggregated tables do not carry identical columns. After the
    union, a column present in only one family is all-NaN for the other, which
    pandas types as object. The committed table stores every measurement column
    as a real number, and the CFTC loader relies on that, so coerce the numeric
    measurement fields explicitly instead of letting the union decide.

    Only the text provenance columns are left alone. ``CFTC_Contract_Market_Code``
    is a code but is stored numerically in the committed table, so it is coerced
    with the measurements.
    """
    non_measurement = {"source_file", "source_dataset", "source_archive",
                       "Market_and_Exchange_Names", "Report_Date_as_MM_DD_YYYY"}
    for column in frame.columns:
        if column in non_measurement or column == "Date":
            continue
        coerced = pd.to_numeric(frame[column], errors="coerce")
        # Keep a column numeric only if it holds at least one real measurement.
        if coerced.notna().any():
            frame[column] = coerced
    return frame



def digest(frame: pd.DataFrame, numeric: list[str] | None = None) -> str:
    """Content hash over sorted measurement fields and their values."""
    if numeric is None:
        numeric = list(frame.select_dtypes("number").columns)
    hasher = hashlib.sha256()
    for column in sorted(numeric):
        hasher.update(column.encode())
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
        hasher.update(np.ascontiguousarray(values, dtype="float64").tobytes())
    return hasher.hexdigest()


def compare(rebuilt: pd.DataFrame, committed: pd.DataFrame) -> dict:
    """Compare the two tables on content, independent of row order.

    Archive members are concatenated in archive order while the committed table
    was sorted, so identical data arrives in a different row order. Sorting on
    the family and report date puts like with like before comparing.

    Fields are compared by explicit dtype only. Probing with ``to_numeric``
    would classify an identifier column that happens to look numeric in one read
    as a measurement field, and the two files would then disagree about the field
    count rather than about any value.
    """
    keys = ["source_dataset", "Report_Date_as_MM_DD_YYYY"]
    left = rebuilt.copy()
    right = committed.copy()
    # The rebuilt frame holds a parsed Timestamp; the committed CSV was read back
    # as text. Normalize both to an ISO string so the key sets are comparable.
    for frame in (left, right):
        frame["Report_Date_as_MM_DD_YYYY"] = pd.to_datetime(
            frame["Report_Date_as_MM_DD_YYYY"], errors="raise").dt.strftime("%Y-%m-%d")
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    report = {"rebuilt_rows": len(rebuilt), "committed_rows": len(committed)}
    for name, frame in (("rebuilt", left), ("committed", right)):
        numeric = sorted(c for c in frame.columns
                         if pd.api.types.is_numeric_dtype(frame[c]))
        report[f"{name}_numeric_fields"] = len(numeric)
        report[f"{name}_digest"] = digest(frame, numeric)
    report["key_set_match"] = (set(map(tuple, left[keys].values))
                               == set(map(tuple, right[keys].values)))
    report["row_count_match"] = report["rebuilt_rows"] == report["committed_rows"]
    report["field_count_match"] = (report["rebuilt_numeric_fields"]
                                   == report["committed_numeric_fields"])
    report["digest_match"] = report["rebuilt_digest"] == report["committed_digest"]
    report["matches"] = all(report[key] for key in
                            ("key_set_match", "row_count_match",
                             "field_count_match", "digest_match"))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true",
                        help="rebuild to scratch and compare with the committed tables")
    parser.add_argument("--write", action="store_true",
                        help="replace DATA/COT with the rebuilt tables and write a manifest")
    args = parser.parse_args()
    if args.verify == args.write:
        parser.error("choose exactly one of --verify or --write")

    import requests
    session = requests.Session()
    combined_path = COT_OUT / "coffee_c_all_cot_data.csv"
    provenance = {"compressed_page": CFTC_HISTORICAL_COMPRESSED_URL,
                  "viewable_page": CFTC_HISTORICAL_VIEWABLE_URL}

    if args.verify:
        import tempfile
        if not combined_path.exists():
            print("No committed table to compare against.")
            return 1
        committed = pd.read_csv(combined_path, low_memory=False)
        with tempfile.TemporaryDirectory() as scratch:
            rebuilt = rebuild(Path(scratch) / "raw", session)
            report = compare(rebuilt, committed)
        print(json.dumps(report, indent=2))
        provenance.update(report)
        provenance["committed_source_sha256"] = hashlib.sha256(
            combined_path.read_bytes()).hexdigest()
        (COT_OUT / "cftc_rebuild_verification.json").write_text(
            json.dumps(provenance, indent=2) + "\n")
        return 0 if report["matches"] else 2

    rebuilt = rebuild(RAW_DIR, session)
    COT_OUT.mkdir(parents=True, exist_ok=True)
    rebuilt.to_csv(combined_path, index=False)
    manifest = {**provenance,
                "families": {name: spec["label"] for name, spec in REPORT_FAMILIES.items()},
                "rows": len(rebuilt),
                "numeric_fields": len(rebuilt.select_dtypes("number").columns),
                "sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest()}
    (COT_OUT / "cftc_refresh_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
