"""Availability-aware weekly COT features for end-of-session return forecasts.

Report dates are position snapshots, not publication dates. Regular releases are
estimated with three US federal business days of processing, never before the
report week's Friday; known interruptions and published schedules take precedence.
A release at 15:30 America/New_York
is usable only on the next *observed* Coffee C session. No values are backfilled.
Historical archives can contain later corrections: this is a conservative release
alignment, not a claim that the raw file contains original unrevised vintages.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

SOURCES = {
    "regular": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm",
    "release_faq": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
    "special": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm",
    "shutdown_2013": "https://www.cftc.gov/PressRoom/PressReleases/6745-13",
    "shutdown_2019": "https://www.cftc.gov/PressRoom/PressReleases/7864-19",
    "shutdown_2025": "https://www.cftc.gov/PressRoom/PressReleases/9147-25",
    "disaggregated_launch": "https://www.cftc.gov/PressRoom/PressReleases/5710-09",
    "historical_backcast": "https://www.cftc.gov/PressRoom/PressReleases/5737-09",
    "coffee_first_disaggregated_report": "https://www.cftc.gov/sites/default/files/files/dea/cotarchives/2009/futures/ag_lf090109.htm",
    "coffee_hours": "https://www.ice.com/products/15/Coffee-C-Futures",
}

# The cache contains corrected archives rather than original publication vintages.
# When CFTC explicitly identified a correction affecting Coffee C futures, wait
# until that correction was available. This withholds the entire report rather
# than inventing its unavailable original fields; unknown corrections remain a
# documented limitation. The options-only Coffee correction in 2008 is irrelevant
# to our futures-only families.
KNOWN_REVISION_DATES = {
    ("legacy_futures_only", "2010-05-18"): "2010-05-27",
    ("disaggregated_futures_only", "2010-05-18"): "2010-05-27",
    ("legacy_futures_only", "2012-11-27"): "2012-12-05",
    ("disaggregated_futures_only", "2012-11-27"): "2012-12-05",
    ("legacy_futures_only", "2018-09-18"): "2018-09-26",
    ("legacy_futures_only", "2019-03-26"): "2019-04-03",
    ("disaggregated_futures_only", "2019-03-26"): "2019-04-03",
}

# These are published schedules/announcements, not inferred from the raw dates.
SPECIAL_RELEASES = {
    "2008-12-22": ("2008-12-29", "announced_2008_holiday", "special"),
    "2013-10-01": ("2013-10-25", "announced_2013_restart", "shutdown_2013"),
    "2014-12-23": ("2014-12-30", "announced_2014_holiday", "special"),
    "2020-12-21": ("2020-12-28", "confirmed_2020_holiday", "special"),
    "2021-06-15": ("2021-06-21", "announced_2021_juneteenth", "special"),
    "2023-01-31": ("2023-02-24", "confirmed_2023_ion", "special"),
    "2023-02-07": ("2023-03-03", "confirmed_2023_ion", "special"),
    "2023-02-14": ("2023-03-08", "confirmed_2023_ion", "special"),
    "2023-02-21": ("2023-03-10", "confirmed_2023_ion", "special"),
    "2023-02-28": ("2023-03-14", "confirmed_2023_ion", "special"),
    "2023-03-07": ("2023-03-16", "confirmed_2023_ion", "special"),
    "2023-03-14": ("2023-03-21", "confirmed_2023_ion", "special"),
    "2025-01-07": ("2025-01-13", "announced_mourning_delay", "special"),
}
for report, release in zip(
    ["2018-12-24", "2018-12-31", "2019-01-08", "2019-01-15", "2019-01-22",
     "2019-01-29", "2019-02-05", "2019-02-12", "2019-02-19", "2019-02-26"],
    ["2019-02-01", "2019-02-05", "2019-02-08", "2019-02-12", "2019-02-15",
     "2019-02-19", "2019-02-22", "2019-02-26", "2019-03-01", "2019-03-05"],
):
    SPECIAL_RELEASES[report] = (release, "announced_2019_catchup_schedule", "shutdown_2019")
for report, release in zip(
    ["2025-09-30", "2025-10-07", "2025-10-14", "2025-10-21", "2025-10-28",
     "2025-11-04", "2025-11-10", "2025-11-18", "2025-11-25", "2025-12-02",
     "2025-12-09", "2025-12-16", "2025-12-23"],
    ["2025-11-19", "2025-11-21", "2025-11-25", "2025-12-02", "2025-12-05",
     "2025-12-09", "2025-12-10", "2025-12-12", "2025-12-15", "2025-12-17",
     "2025-12-19", "2025-12-23", "2025-12-29"],
):
    SPECIAL_RELEASES[report] = (release, "announced_2025_revised_schedule", "shutdown_2025")

# Published 2026 exceptions; ordinary Friday releases need no individual override.
for report, release in {
    "2025-12-30": "2026-01-05", "2026-06-16": "2026-06-22",
    "2026-06-30": "2026-07-06", "2026-11-10": "2026-11-16",
    "2026-11-24": "2026-11-30", "2026-12-22": "2026-12-28",
}.items():
    SPECIAL_RELEASES[report] = (release, "announced_2026_calendar", "regular")

FAMILIES = {
    "legacy_futures_only": ("legacy", {
        "commercial": ("Comm_Positions_Long_All", "Comm_Positions_Short_All", None),
        "noncommercial": ("NonComm_Positions_Long_All", "NonComm_Positions_Short_All", "NonComm_Postions_Spread_All"),
        "nonreportable": ("NonRept_Positions_Long_All", "NonRept_Positions_Short_All", None),
    }),
    "disaggregated_futures_only": ("disagg", {
        "managed_money": ("M_Money_Positions_Long_ALL", "M_Money_Positions_Short_ALL", "M_Money_Positions_Spread_ALL"),
        "producer": ("Prod_Merc_Positions_Long_ALL", "Prod_Merc_Positions_Short_ALL", None),
        "swap": ("Swap_Positions_Long_All", "Swap__Positions_Short_All", "Swap__Positions_Spread_All"),
        "other_reportable": ("Other_Rept_Positions_Long_ALL", "Other_Rept_Positions_Short_ALL", "Other_Rept_Positions_Spread_ALL"),
    }),
}

DISAGGREGATED_FIRST_POSITION_DATE = pd.Timestamp("2009-09-01")
DISAGGREGATED_BACKCAST_PUBLICATION_DATE = pd.Timestamp("2009-10-20")


def _source_field_map(frame: pd.DataFrame, family: str) -> tuple[dict, dict]:
    """Cover every position/count/percentage field, keeping identifiers separate.

    The combined cache has the union of both report schemas. Family membership
    follows field names, never future nonmissing coverage or target correlations.
    This keeps the predictor schema invariant when later report rows are removed.
    """
    short = FAMILIES[family][0]
    included, excluded = {}, {}
    disagg_groups = ("prod_merc", "swap", "m_money", "other_rept")
    metadata_fields = {
        "source_file", "source_dataset", "source_archive", "market_and_exchange_names",
        "as_of_date_in_form_yymmdd", "report_date_as_mm_dd_yyyy", "contract_units",
        "futonly_or_combined",
    }
    derived_fields = {
        "report_date", "publication_date", "release_method", "release_source",
        "release_date_estimated", "release_date_status", "known_revision_date",
        "known_revision_source", "raw_value_available_date", "historical_backcast",
    }
    for name in frame.columns:
        lowered = name.lower()
        if lowered.startswith("cftc_") or lowered in metadata_fields:
            excluded[name] = "identifier_or_text_metadata"
            continue
        if lowered in derived_fields:
            continue
        if ((family == "legacy_futures_only" and any(group in lowered for group in disagg_groups))
                or (family == "disaggregated_futures_only" and ("noncomm" in lowered or "_comm_" in lowered or lowered.startswith("comm_")))):
            excluded[name] = "other_report_family_schema"
            continue
        if lowered.startswith(("pct_of_", "conc_")):
            transform = "percentage_divided_by_100"
        elif lowered.startswith("traders_"):
            transform = "nonnegative_log1p_trader_count"
        elif lowered.startswith("open_interest_"):
            transform = "nonnegative_log1p_open_interest"
        elif "positions_" in lowered or "postions_" in lowered or lowered.startswith("change_in_"):
            transform = "contracts_divided_by_current_all_open_interest"
        else:
            excluded[name] = "unrecognized_nonmeasurement_field"
            continue
        suffix = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
        included[name] = {"feature": f"cot_{short}_source_{suffix}", "transform": transform}
    if len({value["feature"] for value in included.values()}) != len(included):
        raise ValueError("COT source field names collide after normalization")
    return included, excluded


def _source_features(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    mapping, _ = _source_field_map(frame, family)
    oi = _number(frame, "Open_Interest_All").where(lambda value: value > 0)
    values = {}
    for source, specification in mapping.items():
        value = _number(frame, source)
        transform = specification["transform"]
        if transform == "percentage_divided_by_100":
            value = value / 100
        elif transform.startswith("nonnegative_log1p_"):
            value = np.log1p(value.where(value >= 0))
        else:
            value = value / oi
        values[specification["feature"]] = value
    return pd.DataFrame(values, index=frame.index)


def _number(frame: pd.DataFrame, name: str | None) -> pd.Series:
    if name is None or name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _zscore(series: pd.Series, window: int) -> pd.Series:
    # Trailing weekly reports, including the just-published report; no future rows.
    roll = series.rolling(window, min_periods=max(8, window // 2))
    return (series - roll.mean()) / roll.std(ddof=0).replace(0, np.nan)


def _weekly_features(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    short, groups = FAMILIES[family]
    prefix = f"cot_{short}_"
    oi = _number(frame, "Open_Interest_All").where(lambda value: value > 0)
    features = {
        prefix + "log_open_interest": np.log(oi),
        prefix + "oi_change_1w": oi.pct_change(fill_method=None),
        prefix + "oi_change_4w": oi.pct_change(4, fill_method=None),
        prefix + "oi_change_13w": oi.pct_change(13, fill_method=None),
        prefix + "oi_z_52w": _zscore(oi, 52),
        prefix + "report_gap_days": frame["report_date"].diff().dt.days,
        prefix + "old_crop_oi_share": _number(frame, "Open_Interest_Old") / oi,
        prefix + "trader_count_log": np.log1p(_number(frame, "Traders_Tot_All")),
        prefix + "contracts_per_trader": oi / _number(frame, "Traders_Tot_All").replace(0, np.nan),
    }
    for group, (long_col, short_col, spread_col) in groups.items():
        group_prefix = prefix + group + "_"
        long = _number(frame, long_col)
        short_position = _number(frame, short_col)
        net = (long - short_position) / oi
        features.update({
            group_prefix + "net_oi": net,
            group_prefix + "long_oi": long / oi,
            group_prefix + "short_oi": short_position / oi,
            group_prefix + "gross_oi": (long + short_position) / oi,
            group_prefix + "long_short_balance": (long - short_position) / (long + short_position).replace(0, np.nan),
            group_prefix + "net_change_1w": net.diff(),
            group_prefix + "net_change_4w": net.diff(4),
            group_prefix + "net_change_13w": net.diff(13),
            group_prefix + "net_acceleration_4w": net.diff(4) - net.diff(4).shift(4),
            group_prefix + "net_z_26w": _zscore(net, 26),
            group_prefix + "net_z_52w": _zscore(net, 52),
            group_prefix + "net_change_z_26w": _zscore(net.diff(), 26),
        })
        if spread_col is not None:
            features[group_prefix + "spread_oi"] = _number(frame, spread_col) / oi
        for window in (52, 156):
            roll = net.rolling(window, min_periods=max(13, window // 2))
            low, high = roll.min(), roll.max()
            features[group_prefix + f"index_{window}w"] = (net - low) / (high - low).replace(0, np.nan)
    for size in (4, 8):
        for kind in ("Gross", "Net"):
            long = _number(frame, f"Conc_{kind}_LE_{size}_TDR_Long_All")
            short_position = _number(frame, f"Conc_{kind}_LE_{size}_TDR_Short_All")
            features[prefix + f"concentration_{kind.lower()}_{size}_imbalance"] = (long - short_position) / 100
    return pd.concat([pd.DataFrame(features, index=frame.index), _source_features(frame, family)], axis=1)


def _release_calendar(report_dates: pd.Series, root: Path) -> pd.DataFrame:
    dates = pd.DatetimeIndex(report_dates)
    # Mourning closures supplement the recurring federal calendar. All remaining
    # ordinary dates remain explicitly labelled estimates, including older years.
    holidays = USFederalHolidayCalendar().holidays(dates.min() - pd.Timedelta(days=7), dates.max() + pd.Timedelta(days=30))
    holidays = holidays.union(pd.to_datetime(["2004-06-11", "2007-01-02", "2018-12-05", "2025-01-09"]))
    processing_day = CustomBusinessDay(3, holidays=holidays)
    next_business_day = CustomBusinessDay(holidays=holidays)
    result = pd.DataFrame({"report_date": dates})
    # Rare Monday/Wed/Fri snapshots occur in the raw archive. A Monday with no
    # midweek federal holiday must not mechanically acquire a Thursday release.
    # Friday snapshots conservatively wait for the following week's Friday.
    friday_floor = [date + pd.Timedelta(days=(4 - date.weekday()) % 7 or 7)
                    if date.weekday() >= 4 else date + pd.Timedelta(days=4 - date.weekday())
                    for date in dates]
    result["publication_date"] = [next_business_day.rollforward(max(date + processing_day, floor))
                                  for date, floor in zip(dates, friday_floor)]
    result["release_method"] = "estimated_three_federal_business_days_friday_floor"
    result["release_source"] = SOURCES["regular"]
    result["release_date_estimated"] = True
    result["release_date_status"] = "estimated"
    # The 2013 announcement gives restart/catch-up bounds, not exact intermediate
    # dates: withhold unverified backlog until its announced completion date.
    mask = result["report_date"].between("2013-10-08", "2013-10-29")
    result.loc[mask, "publication_date"] = pd.Timestamp("2013-11-08")
    result.loc[mask, "release_method"] = "conservative_2013_catchup_bound"
    result.loc[mask, "release_source"] = SOURCES["shutdown_2013"]
    result.loc[mask, "release_date_status"] = "conservative_announced_bound"
    for report, (publication, method, source) in SPECIAL_RELEASES.items():
        mask = result["report_date"].eq(pd.Timestamp(report))
        result.loc[mask, "publication_date"] = pd.Timestamp(publication)
        result.loc[mask, "release_method"] = method
        result.loc[mask, "release_source"] = SOURCES[source]
        # An intended schedule is distinguished from a same-day confirmation.
        confirmed = method.startswith("confirmed_")
        result.loc[mask, "release_date_estimated"] = not confirmed
        result.loc[mask, "release_date_status"] = "confirmed" if confirmed else "announced_schedule"
    # Project-wide overrides remain supported; a MonthFu-local override is applied
    # last so all new experiment-specific work can stay inside MonthFu.
    override_paths = [root / "data" / "COT" / "cot_release_overrides.csv",
                      root / "MonthFu" / "data" / "cot_release_overrides.csv"]
    for override_path in override_paths:
        if not override_path.exists():
            continue
        overrides = pd.read_csv(override_path)
        required = {"report_date", "publication_date", "source"}
        if not required.issubset(overrides):
            raise ValueError(f"{override_path} must contain {sorted(required)}")
        overrides["report_date"] = pd.to_datetime(overrides["report_date"], errors="raise").dt.normalize()
        overrides["publication_date"] = pd.to_datetime(overrides["publication_date"], errors="raise").dt.normalize()
        if (overrides["report_date"].duplicated().any()
                or overrides[list(required)].isna().any().any()
                or overrides["source"].astype(str).str.strip().eq("").any()):
            raise ValueError("COT release overrides must have unique report dates and nonmissing dates/source")
        for row in overrides.itertuples():
            mask = result["report_date"].eq(row.report_date)
            result.loc[mask, ["publication_date", "release_method", "release_source", "release_date_estimated"]] = [row.publication_date, "explicit_override", row.source, False]
            result.loc[mask, "release_date_status"] = "explicit_verified_override"
    if (result["publication_date"] < result["report_date"]).any():
        raise ValueError("COT publication cannot precede the position snapshot")
    return result


def build_cot_features(prices: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return Date + numeric cot_* features, a per-report release audit, metadata.

    ``prices.Date`` supplies the observed trading calendar and must be unique.
    Optional ``MonthFu/data/cot_release_overrides.csv`` (or project-wide
    ``data/COT/cot_release_overrides.csv``) columns are ``report_date``,
    ``publication_date``, ``source``. Dates are daily session labels, not UTC times.
    Missing COT data is left missing; no pre-release or backwards filling occurs.
    """
    root = Path(root)
    sessions = pd.to_datetime(prices["Date"], errors="raise").dt.normalize().astype("datetime64[ns]")
    if sessions.isna().any() or sessions.duplicated().any():
        raise ValueError("Price session dates must be nonmissing and unique")
    daily = pd.DataFrame({"Date": sessions}).sort_values("Date").reset_index(drop=True)
    if daily.empty:
        return daily, pd.DataFrame(), {"reports": 0, "features": 0, "sources": SOURCES}
    path = root / "data" / "COT" / "coffee_c_all_cot_data.csv"
    raw = pd.read_csv(path, low_memory=False).copy()
    raw["report_date"] = pd.to_datetime(raw["Report_Date_as_MM_DD_YYYY"], errors="raise").dt.normalize()
    if raw["report_date"].isna().any():
        raise ValueError("COT report dates must be nonmissing")
    raw = raw.loc[raw["source_dataset"].isin(FAMILIES)].copy()
    # Retain the historical archive, but its position dates are not publication
    # dates. Its classifications were backcast and first public on October 20.
    backcast = raw["source_dataset"].eq("disaggregated_futures_only") & raw["report_date"].lt(DISAGGREGATED_FIRST_POSITION_DATE)
    raw["historical_backcast"] = backcast
    retained_backcast_rows = int(backcast.sum())
    if raw.duplicated(["source_dataset", "report_date"]).any():
        raise ValueError("Duplicate report family/date keys in raw COT data")
    if raw.empty:
        raise ValueError("No supported COT reports")
    calendar = _release_calendar(pd.Series(raw["report_date"].unique()).sort_values(), root)
    raw = raw.merge(calendar, on="report_date", how="left", validate="many_to_one")
    archive_release = raw["historical_backcast"] & raw["publication_date"].lt(DISAGGREGATED_BACKCAST_PUBLICATION_DATE)
    raw.loc[archive_release, "publication_date"] = DISAGGREGATED_BACKCAST_PUBLICATION_DATE
    raw.loc[archive_release, "release_method"] = "announced_2009_disaggregated_backcast_archive"
    raw.loc[archive_release, "release_source"] = SOURCES["historical_backcast"]
    raw.loc[archive_release, "release_date_estimated"] = False
    raw.loc[archive_release, "release_date_status"] = "announced_archive_release"
    raw["known_revision_date"] = pd.NaT
    raw["known_revision_source"] = ""
    for (family, report), revision_date in KNOWN_REVISION_DATES.items():
        mask = raw["source_dataset"].eq(family) & raw["report_date"].eq(pd.Timestamp(report))
        raw.loc[mask, "known_revision_date"] = pd.Timestamp(revision_date)
        raw.loc[mask, "known_revision_source"] = SOURCES["special"]
    raw["raw_value_available_date"] = raw[["publication_date", "known_revision_date"]].max(axis=1)
    audits = []
    field_coverage = {}
    session_array = daily["Date"].to_numpy(dtype="datetime64[ns]")
    for family, (short, _) in FAMILIES.items():
        weekly = raw.loc[raw["source_dataset"].eq(family)].sort_values("report_date").reset_index(drop=True)
        if weekly.empty:
            continue
        source_map, excluded_fields = _source_field_map(weekly, family)
        field_coverage[family] = {
            "included_source_fields": source_map,
            "excluded_source_fields": excluded_fields,
            "source_fields_with_observations": [name for name in source_map if _number(weekly, name).notna().any()],
        }
        # Before October 20, 2009, the live disaggregated series must use only
        # then-public reports. A second history adds the released archive at an
        # explicit publication event, recomputing windows for the latest current
        # position snapshot. Old backcasts never replace a newer live snapshot.
        current = weekly.loc[~weekly["historical_backcast"]].copy()
        current["feature_publication_date"] = current["raw_value_available_date"].cummax()
        features = _weekly_features(weekly, family)
        expanded_publication = weekly["raw_value_available_date"].cummax()
        if weekly["historical_backcast"].any():
            archive_ready = weekly.loc[weekly["historical_backcast"], "raw_value_available_date"].max()
            expanded_publication = expanded_publication.clip(lower=archive_ready)
        else:
            archive_ready = pd.NaT
        weekly["feature_publication_date"] = expanded_publication
        weekly.loc[current.index, "feature_publication_date"] = current["feature_publication_date"]
        weekly["expanded_history_publication_date"] = expanded_publication
        indices = np.searchsorted(session_array, weekly["feature_publication_date"].to_numpy(dtype="datetime64[ns]"), side="right")
        weekly["first_usable_session"] = pd.to_datetime([session_array[i] if i < len(session_array) else np.datetime64("NaT", "ns") for i in indices])
        audit = weekly[["source_dataset", "report_date", "historical_backcast", "publication_date", "known_revision_date", "known_revision_source", "raw_value_available_date", "feature_publication_date", "expanded_history_publication_date", "first_usable_session", "release_method", "release_source", "release_date_estimated", "release_date_status"]].copy()
        audit["publication_time_et"] = np.where(audit["historical_backcast"], "archive time unspecified; next observed session", "15:30 America/New_York")
        audit["release_lag_calendar_days"] = (audit["publication_date"] - audit["report_date"]).dt.days
        audits.append(audit)
        def make_payload(history, history_features, feature_dates, variant):
            payload = history[["report_date", "publication_date", "raw_value_available_date", "known_revision_date", "release_date_estimated"]].copy()
            payload["feature_publication_date"] = feature_dates
            usable_indices = np.searchsorted(session_array, feature_dates.to_numpy(dtype="datetime64[ns]"), side="right")
            payload["first_usable_session"] = pd.to_datetime([session_array[i] if i < len(session_array) else np.datetime64("NaT", "ns") for i in usable_indices])
            payload["history_variant"] = variant
            return pd.concat([payload, history_features], axis=1)

        if weekly["historical_backcast"].any():
            # Wait for every archived value required by this full-history variant,
            # including a later verified override/correction, before activation.
            baseline = make_payload(current, _weekly_features(current, family), current["feature_publication_date"], 0)
            baseline = baseline.loc[baseline["feature_publication_date"].lt(archive_ready)]
            expanded = make_payload(weekly, features, expanded_publication, 1)
            payload = pd.concat([baseline, expanded], ignore_index=True)
        else:
            payload = make_payload(weekly, features, expanded_publication, 0)
        payload = payload.dropna(subset=["first_usable_session"]).sort_values(["first_usable_session", "report_date"])
        # Latest position snapshot wins if several backlog releases become usable
        # on the same session; all earlier reports remain in the weekly windows.
        payload = payload.sort_values(["first_usable_session", "report_date", "history_variant"]).drop_duplicates("first_usable_session", keep="last")
        # A delayed old release, including an archive-only history, must never
        # replace a more recent position snapshot already available to the model.
        latest_date = payload["report_date"].cummax()
        payload = payload.loc[payload["report_date"].eq(latest_date)]
        joined = pd.merge_asof(daily[["Date"]], payload, left_on="Date", right_on="first_usable_session", direction="backward")
        prefix = f"cot_{short}_"
        extra = {
            prefix + "available": joined["report_date"].notna().astype(float),
            prefix + "snapshot_age_days": (joined["Date"] - joined["report_date"]).dt.days,
            prefix + "release_age_days": (joined["Date"] - joined["publication_date"]).dt.days,
            prefix + "release_lag_days": (joined["publication_date"] - joined["report_date"]).dt.days,
            prefix + "value_age_days": (joined["Date"] - joined["raw_value_available_date"]).dt.days,
            prefix + "dependency_delay_days": (joined["feature_publication_date"] - joined["publication_date"]).dt.days,
            prefix + "known_revised_archive": joined["known_revision_date"].notna().astype(float).where(joined["report_date"].notna()),
            prefix + "release_estimated": joined["release_date_estimated"].astype(float),
            prefix + "new_release_session": joined["Date"].eq(joined["first_usable_session"]).astype(float),
        }
        extra[prefix + "stale_14d"] = (extra[prefix + "snapshot_age_days"] > 14).astype(float).where(joined["report_date"].notna())
        if family == "disaggregated_futures_only":
            history_active = joined["history_variant"].eq(1)
            first_archive_index = np.searchsorted(session_array, np.datetime64(archive_ready, "ns"), side="right") if pd.notna(archive_ready) else len(session_array)
            first_archive_session = pd.Timestamp(session_array[first_archive_index]) if first_archive_index < len(session_array) else pd.NaT
            extra[prefix + "backcast_history_available"] = history_active.astype(float)
            extra[prefix + "backcast_history_age_days"] = (joined["Date"] - archive_ready).dt.days.where(history_active)
            extra[prefix + "backcast_history_new_session"] = joined["Date"].eq(first_archive_session).astype(float)
        daily = pd.concat([daily, joined[features.columns], pd.DataFrame(extra)], axis=1)
    if {"cot_disagg_managed_money_net_oi", "cot_legacy_noncommercial_net_oi"}.issubset(daily):
        daily["cot_managed_money_vs_noncommercial"] = daily["cot_disagg_managed_money_net_oi"] - daily["cot_legacy_noncommercial_net_oi"]
        daily["cot_producer_vs_commercial"] = daily["cot_disagg_producer_net_oi"] - daily["cot_legacy_commercial_net_oi"]
    audit_frame = pd.concat(audits, ignore_index=True).sort_values(["report_date", "source_dataset"]).reset_index(drop=True)
    metadata = {
        "source_file": str(path.relative_to(root)),
        "feature_schema_version": 2,
        "backcast_policy": "Retain prelaunch disaggregated archives; current-only history before 2009-10-20, full-history predictors after archive/dependency publication on the next observed session; never replace a newer positioning snapshot with an old archive report",
        "reports": int(len(audit_frame)), "features": int(daily.shape[1] - 1),
        "excluded_prelaunch_disaggregated_backcasts": 0,
        "retained_prelaunch_disaggregated_backcasts": retained_backcast_rows,
        "backcast_publication_date": DISAGGREGATED_BACKCAST_PUBLICATION_DATE.date().isoformat(),
        "source_field_coverage": field_coverage,
        "alignment": "Backward as-of on first observed trading session strictly after publication; no backfill",
        "normal_release_rule": "Estimated max(three federal business days after position date, report-week Friday); Friday snapshots use next Friday; 15:30 ET; dates labelled estimated",
        "override_schema": "MonthFu/data/cot_release_overrides.csv (preferred) or data/COT/cot_release_overrides.csv: report_date,publication_date,source",
        "weekly_windows": "Ordered weekly reports; pre-archive current-only windows, full archive history activates only after its publication/dependency dates; latest snapshot always wins",
        "release_method_counts": audit_frame["release_method"].value_counts().to_dict(),
        "known_corrected_archive_reports": int(audit_frame["known_revision_date"].notna().sum()),
        "sources": SOURCES,
        "limitations": [
            "Raw historical archives may contain subsequent revisions; original as-published data vintages are unavailable.",
            "Known Coffee futures corrections (2010-05-18, 2018-09-18 legacy concentration, 2019-03-26) and the non-market-specific 2012-11-27 update are withheld until their announced correction dates; unannounced/unidentified revisions remain possible.",
            "Ordinary historical release dates are estimates, not a complete verified historical publication calendar.",
            "2013 intermediate backlog reports are withheld until the announced completion bound; 2019 catch-up dates follow the announced twice-weekly schedule.",
            "Pre-September-2009 disaggregated backcasts are retained, but cannot seed live/rolling predictors before October 20, 2009; archive publication time is unspecified, so first use is the following observed session. Classification history was backcast rather than measured contemporaneously.",
            "Observed price dates provide the trading calendar; a missing price session can conservatively postpone availability.",
        ],
    }
    return daily.replace([np.inf, -np.inf], np.nan), audit_frame, metadata
