"""Availability-aware weekly COT features for end-of-session return forecasts.

Report dates are position snapshots, not publication dates. Regular releases are
estimated with three US federal business days of processing; known interruptions
and the published 2026 calendar take precedence. A release at 15:30 America/New_York
is usable only on the next *observed* Coffee C session. No values are backfilled.
Historical archives can contain later corrections: this is a conservative release
alignment, not a claim that the raw file contains original unrevised vintages.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

SOURCES = {
    "regular": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm",
    "special": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm",
    "shutdown_2013": "https://www.cftc.gov/PressRoom/PressReleases/6745-13",
    "shutdown_2019": "https://www.cftc.gov/PressRoom/PressReleases/7864-19",
    "shutdown_2025": "https://www.cftc.gov/PressRoom/PressReleases/9147-25",
    "disaggregated_launch": "https://www.cftc.gov/PressRoom/PressReleases/5710-09",
    "historical_backcast": "https://www.cftc.gov/PressRoom/PressReleases/5737-09",
    "coffee_hours": "https://www.ice.com/products/15/Coffee-C-Futures",
}

# These are published schedules/announcements, not inferred from the raw dates.
SPECIAL_RELEASES = {
    "2013-10-01": ("2013-10-25", "announced_2013_restart", "shutdown_2013"),
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
    return pd.DataFrame(features, index=frame.index)


def _release_calendar(report_dates: pd.Series, root: Path) -> pd.DataFrame:
    dates = pd.DatetimeIndex(report_dates)
    # Mourning closures supplement the recurring federal calendar. All remaining
    # ordinary dates remain explicitly labelled estimates, including older years.
    holidays = USFederalHolidayCalendar().holidays(dates.min() - pd.Timedelta(days=7), dates.max() + pd.Timedelta(days=30))
    holidays = holidays.union(pd.to_datetime(["2004-06-11", "2007-01-02", "2018-12-05", "2025-01-09"]))
    processing_day = CustomBusinessDay(3, holidays=holidays)
    result = pd.DataFrame({"report_date": dates})
    result["publication_date"] = [date + processing_day for date in dates]
    result["release_method"] = "estimated_three_federal_business_days"
    result["release_source"] = SOURCES["regular"]
    result["release_date_estimated"] = True
    # The 2013 announcement gives restart/catch-up bounds, not exact intermediate
    # dates: withhold unverified backlog until its announced completion date.
    mask = result["report_date"].between("2013-10-08", "2013-10-29")
    result.loc[mask, "publication_date"] = pd.Timestamp("2013-11-08")
    result.loc[mask, "release_method"] = "conservative_2013_catchup_bound"
    result.loc[mask, "release_source"] = SOURCES["shutdown_2013"]
    for report, (publication, method, source) in SPECIAL_RELEASES.items():
        mask = result["report_date"].eq(pd.Timestamp(report))
        result.loc[mask, "publication_date"] = pd.Timestamp(publication)
        result.loc[mask, "release_method"] = method
        result.loc[mask, "release_source"] = SOURCES[source]
        # A published catch-up schedule is distinguished from actual confirmation.
        result.loc[mask, "release_date_estimated"] = method.startswith("announced_2019")
    override_path = root / "data" / "COT" / "cot_release_overrides.csv"
    if override_path.exists():
        overrides = pd.read_csv(override_path)
        required = {"report_date", "publication_date", "source"}
        if not required.issubset(overrides):
            raise ValueError(f"{override_path} must contain {sorted(required)}")
        overrides["report_date"] = pd.to_datetime(overrides["report_date"], errors="raise").dt.normalize()
        overrides["publication_date"] = pd.to_datetime(overrides["publication_date"], errors="raise").dt.normalize()
        if overrides["report_date"].duplicated().any() or overrides[list(required)].isna().any().any():
            raise ValueError("COT release overrides must have unique report dates and nonmissing dates/source")
        for row in overrides.itertuples():
            mask = result["report_date"].eq(row.report_date)
            result.loc[mask, ["publication_date", "release_method", "release_source", "release_date_estimated"]] = [row.publication_date, "explicit_override", row.source, False]
    if (result["publication_date"] < result["report_date"]).any():
        raise ValueError("COT publication cannot precede the position snapshot")
    return result


def build_cot_features(prices: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return Date + numeric cot_* features, a per-report release audit, metadata.

    ``prices.Date`` supplies the observed trading calendar and must be unique.
    Optional ``data/COT/cot_release_overrides.csv`` columns are ``report_date``,
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
    # 2006-09 history was backcast and only published in October 2009. Excluding
    # it entirely also prevents rolling features from using it before release.
    backcast = raw["source_dataset"].eq("disaggregated_futures_only") & raw["report_date"].lt("2009-09-01")
    excluded_backcast_rows = int(backcast.sum())
    raw = raw.loc[~backcast].copy()
    if raw.duplicated(["source_dataset", "report_date"]).any():
        raise ValueError("Duplicate report family/date keys in raw COT data")
    if raw.empty:
        raise ValueError("No supported COT reports")
    calendar = _release_calendar(pd.Series(raw["report_date"].unique()).sort_values(), root)
    raw = raw.merge(calendar, on="report_date", how="left", validate="many_to_one")
    audits = []
    session_array = daily["Date"].to_numpy(dtype="datetime64[ns]")
    for family, (short, _) in FAMILIES.items():
        weekly = raw.loc[raw["source_dataset"].eq(family)].sort_values("report_date").reset_index(drop=True)
        if weekly.empty:
            continue
        features = _weekly_features(weekly, family)
        # If a manual correction produces out-of-order publication, rolling
        # calculations cannot use earlier reports until those too are published.
        weekly["feature_publication_date"] = weekly["publication_date"].cummax()
        indices = np.searchsorted(session_array, weekly["feature_publication_date"].to_numpy(dtype="datetime64[ns]"), side="right")
        weekly["first_usable_session"] = pd.to_datetime([session_array[i] if i < len(session_array) else np.datetime64("NaT") for i in indices])
        audit = weekly[["source_dataset", "report_date", "publication_date", "feature_publication_date", "first_usable_session", "release_method", "release_source", "release_date_estimated"]].copy()
        audit["publication_time_et"] = "15:30 America/New_York"
        audit["release_lag_calendar_days"] = (audit["publication_date"] - audit["report_date"]).dt.days
        audits.append(audit)
        payload = pd.concat([weekly[["report_date", "publication_date", "first_usable_session", "release_date_estimated"]], features], axis=1)
        payload = payload.dropna(subset=["first_usable_session"]).sort_values(["first_usable_session", "report_date"])
        # Latest position snapshot wins if several backlog releases become usable
        # on the same session; all earlier reports remain in the weekly windows.
        payload = payload.drop_duplicates("first_usable_session", keep="last")
        joined = pd.merge_asof(daily[["Date"]], payload, left_on="Date", right_on="first_usable_session", direction="backward")
        prefix = f"cot_{short}_"
        extra = {
            prefix + "available": joined["report_date"].notna().astype(float),
            prefix + "snapshot_age_days": (joined["Date"] - joined["report_date"]).dt.days,
            prefix + "release_age_days": (joined["Date"] - joined["publication_date"]).dt.days,
            prefix + "release_lag_days": (joined["publication_date"] - joined["report_date"]).dt.days,
            prefix + "release_estimated": joined["release_date_estimated"].astype(float),
            prefix + "new_release_session": joined["Date"].eq(joined["first_usable_session"]).astype(float),
        }
        extra[prefix + "stale_14d"] = (extra[prefix + "snapshot_age_days"] > 14).astype(float).where(joined["report_date"].notna())
        daily = pd.concat([daily, joined[features.columns], pd.DataFrame(extra)], axis=1)
    if {"cot_disagg_managed_money_net_oi", "cot_legacy_noncommercial_net_oi"}.issubset(daily):
        daily["cot_managed_money_vs_noncommercial"] = daily["cot_disagg_managed_money_net_oi"] - daily["cot_legacy_noncommercial_net_oi"]
        daily["cot_producer_vs_commercial"] = daily["cot_disagg_producer_net_oi"] - daily["cot_legacy_commercial_net_oi"]
    audit_frame = pd.concat(audits, ignore_index=True).sort_values(["report_date", "source_dataset"]).reset_index(drop=True)
    metadata = {
        "source_file": str(path.relative_to(root)),
        "reports": int(len(audit_frame)), "features": int(daily.shape[1] - 1),
        "excluded_prelaunch_disaggregated_backcasts": excluded_backcast_rows,
        "alignment": "Backward as-of on first observed trading session strictly after publication; no backfill",
        "normal_release_rule": "Estimated three federal business days after position date; 15:30 ET; dates labelled estimated",
        "override_schema": "data/COT/cot_release_overrides.csv: report_date,publication_date,source",
        "weekly_windows": "Changes/index/z-scores computed on ordered weekly reports before daily alignment",
        "release_method_counts": audit_frame["release_method"].value_counts().to_dict(),
        "sources": SOURCES,
        "limitations": [
            "Raw historical archives may contain subsequent revisions; original as-published data vintages are unavailable.",
            "Ordinary historical release dates are estimates, not a complete verified historical publication calendar.",
            "2013 intermediate backlog reports are withheld until the announced completion bound; 2019 catch-up dates follow the announced twice-weekly schedule.",
            "All pre-September-2009 disaggregated backcasts are excluded, including from rolling windows.",
            "Observed price dates provide the trading calendar; a missing price session can conservatively postpone availability.",
        ],
    }
    return daily.replace([np.inf, -np.inf], np.nan), audit_frame, metadata
