"""Lagged local weather and optional retrospective news features.

All dates are daily observation dates, and predictions are made after the market
close. Weather is historical reanalysis, not a publication-vintage archive: the
five-calendar-day lag is a conservative proxy, not proof of point-in-time data.
News remains exploratory because its upstream event selection used returns and
some full-history normalization. Delaying it cannot undo that selection.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_LAG_DAYS = 5
WEATHER_MAX_RELEASE_AGE_DAYS = 7
NEWS_LAG_SESSIONS = 6
NEWS_MAX_SOURCE_AGE_DAYS = 35
REGIONS = ("brazil_minas_gerais", "colombia_huila", "vietnam_dak_lak")
WEATHER_WINDOWS = (7, 14, 28, 30, 60, 90)
DAILY_NEWS_COLUMNS = ("trading_date", "event_count", "bullish_events", "bearish_events")
WEEKLY_NEWS_COLUMNS = (
    "period_end", "event_count", "bullish_event_count", "bearish_event_count",
    "explicit_coffee_event_count", "mean_event_intensity_score_0_1", "top_event_digest",
)


def _dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="raise").dt.normalize()


def _date_span(values: pd.Series) -> dict:
    valid = pd.to_datetime(values).dropna()
    return {"start": str(valid.min().date()) if len(valid) else None,
            "end": str(valid.max().date()) if len(valid) else None}


def _calendar(prices: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"Date": _dates(prices["Date"])})
    if out.Date.isna().any() or out.Date.duplicated().any():
        raise ValueError("Price calendar must contain unique, non-null observed session dates.")
    return out.sort_values("Date").reset_index(drop=True)


def _prior_seasonal_zscore(series: pd.Series) -> pd.Series:
    """Normalize against the previous five occurrences of this calendar month.

    The complete current month is never part of its baseline. Monthly moments
    permit a historical daily variance without re-reading any future rows.
    Two previous years of that month are required; missing months stay missing.
    """
    monthly = pd.DataFrame({"sum": series, "squares": series**2, "count": series.notna().astype(float)})
    monthly = monthly.groupby(series.index.to_period("M")).sum(min_count=1)
    prior = monthly.groupby(monthly.index.month, group_keys=False).transform(
        lambda group: group.shift(1).rolling(5, min_periods=2).sum()
    )
    count = prior["count"].where(prior["count"] >= 45)
    mean = prior["sum"] / count
    variance = (prior["squares"] / count - mean**2).clip(lower=0)
    baseline = pd.DataFrame({"mean": mean, "std": np.sqrt(variance)})
    aligned = baseline.reindex(series.index.to_period("M"))
    aligned.index = series.index
    return (series - aligned["mean"]) / aligned["std"].where(aligned["std"] > 1e-6)


def _dry_spell(rain: pd.Series) -> pd.Series:
    # An unknown observation interrupts the spell; it is never treated as dry.
    dry = rain.lt(1.0) & rain.notna()
    spells = dry.groupby((~dry).cumsum()).cumsum().astype(float)
    return spells.where(rain.notna())


def _region_weather(daily: pd.DataFrame, region: str) -> pd.DataFrame:
    prefix = f"weather_{region}_"

    def variable(name: str) -> pd.Series:
        values = daily.get(prefix + name, pd.Series(np.nan, index=daily.index))
        return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)

    temp = variable("temperature_2m_mean")
    high, low = variable("temperature_2m_max"), variable("temperature_2m_min")
    rain, soil = variable("precipitation_sum"), variable("soil_moisture_0_to_100cm_mean")
    vpd, et0 = variable("vapour_pressure_deficit_max"), variable("et0_fao_evapotranspiration")
    rain, soil, vpd, et0 = (value.where(value >= 0) for value in (rain, soil, vpd, et0))
    water_balance = rain - et0
    temp_z, rain_z = _prior_seasonal_zscore(temp), _prior_seasonal_zscore(rain)
    soil_z = _prior_seasonal_zscore(soil)
    values = {}
    for window in WEATHER_WINDOWS:
        minimum = max(3, int(np.ceil(window * 0.8)))
        for name, series, aggregation in (
            ("temp", temp, "mean"), ("temp_max", high, "mean"),
            ("rain", rain, "sum"), ("soil", soil, "mean"),
            ("vpd", vpd, "mean"), ("water_balance", water_balance, "sum"),
            ("temp_seasonal_z", temp_z, "mean"), ("rain_seasonal_z", rain_z, "mean"),
            ("soil_seasonal_z", soil_z, "mean"),
            ("dry_share", rain.lt(1).astype(float).where(rain.notna()), "mean"),
            ("heat_degrees", (high - 30).clip(lower=0), "mean"),
            ("cold_share", low.lt(5).astype(float).where(low.notna()), "mean"),
            ("heavy_rain_share", rain.gt(20).astype(float).where(rain.notna()), "mean"),
        ):
            values[f"{prefix}{name}_{window}d"] = getattr(series.rolling(window, min_periods=minimum), aggregation)()
    values[prefix + "dry_spell_days"] = _dry_spell(rain)
    values[prefix + "rain_acceleration"] = values[prefix + "rain_7d"] / 7 - values[prefix + "rain_30d"] / 30
    values[prefix + "soil_change_30d"] = values[prefix + "soil_7d"] - values[prefix + "soil_7d"].shift(30)
    values[prefix + "heat_vpd_stress_30d"] = values[prefix + "heat_degrees_30d"] * values[prefix + "vpd_30d"]
    values[prefix + "max_dry_spell_30d"] = values[prefix + "dry_spell_days"].rolling(30, min_periods=24).max()
    values[prefix + "dry_heat_stress_30d"] = values[prefix + "dry_share_30d"] * values[prefix + "heat_vpd_stress_30d"]
    values[prefix + "rain_trend_30_90d"] = values[prefix + "rain_30d"] / 30 - values[prefix + "rain_90d"] / 90
    values[prefix + "soil_trend_30_90d"] = values[prefix + "soil_30d"] - values[prefix + "soil_90d"]
    values[prefix + "water_deficit_30d"] = -values[prefix + "water_balance_30d"].clip(upper=0)
    values[prefix + "min_temperature_7d"] = low.rolling(7, min_periods=6).min()
    if region == "brazil_minas_gerais":
        # Broad calendar interactions are hypotheses, not parcel phenology.
        month = pd.Series(daily.index.month, index=daily.index)
        values[prefix + "flowering_water_deficit"] = -values[prefix + "water_balance_60d"].clip(upper=0) * month.isin([9, 10, 11])
        values[prefix + "harvest_rain"] = values[prefix + "rain_30d"] * month.isin([5, 6, 7, 8, 9])
        values[prefix + "winter_cold_risk"] = values[prefix + "cold_share_7d"] * month.isin([6, 7, 8])
    out = pd.DataFrame(values, index=daily.index)
    # Drop incomplete observation rows before the asof join; do not manufacture
    # a fresh release from an old rolling window during a cache gap.
    out = out.loc[temp.notna() & rain.notna()].copy()
    out["_source_date"] = out.index
    out["_available_date"] = out.index + pd.Timedelta(days=WEATHER_LAG_DAYS)
    return out.reset_index(drop=True)


def build_weather_features(weather: pd.DataFrame, calendar: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build daily weather features and join only observations already available."""
    weather = weather.copy()
    weather["Date"] = _dates(weather["Date"])
    if weather.Date.isna().any() or weather.Date.duplicated().any():
        raise ValueError("Weather observation dates must be unique and non-null.")
    weather = weather.sort_values("Date")
    if weather.empty:
        return calendar.copy(), {"rows": 0, "regions": {}}
    daily = weather.set_index("Date").reindex(pd.date_range(weather.Date.min(), weather.Date.max(), freq="D"))
    parts, regions = [calendar.copy()], {}
    for region in REGIONS:
        if not any(column.startswith(f"weather_{region}_") for column in weather):
            continue
        source = _region_weather(daily, region)
        if source.empty:
            continue
        features = [column for column in source if column.startswith("weather_")]
        aligned = pd.merge_asof(calendar, source, left_on="Date", right_on="_available_date", direction="backward")
        age = (aligned.Date - aligned._source_date).dt.days
        stale = age.isna() | (age > WEATHER_LAG_DAYS + WEATHER_MAX_RELEASE_AGE_DAYS)
        aligned.loc[stale, features] = np.nan
        part = aligned[features].copy()
        part[f"weather_{region}_source_age_days"] = age
        part[f"weather_{region}_stale"] = stale.astype(float)
        parts.append(part)
        regions[region] = {"observations": len(source), **_date_span(source._source_date),
                           "fresh_match_fraction": float((~stale).mean()),
                           "latest_used_source": str(aligned._source_date.iloc[-1].date()) if pd.notna(aligned._source_date.iloc[-1]) else None}
    out = pd.concat(parts, axis=1)
    # Equal-region means avoid asserting an unsupported production weighting.
    for variable in ("rain_seasonal_z_30d", "temp_seasonal_z_30d", "rain_seasonal_z_90d", "water_balance_60d"):
        brazil, colombia = (f"weather_{region}_{variable}" for region in REGIONS[:2])
        if brazil in out and colombia in out:
            out[f"weather_arabica_region_mean_{variable}"] = (out[brazil] + out[colombia]) / 2
            out[f"weather_arabica_region_difference_{variable}"] = out[brazil] - out[colombia]
    return out, {"rows": len(weather), "regions": regions}


def _text_news_features(text: pd.Series) -> pd.DataFrame:
    terms = {
        "disruption": ("frost", "drought", "flood", "wildfire", "strike", "conflict", "war", "sanctions", "blockade", "disease", "shortage", "export ban"),
        "support": ("bumper", "recovery", "surplus", "ceasefire", "agreement", "reopen"),
        "coffee": ("coffee", "arabica", "cafe", "caffeine", "roaster"),
    }
    text = text.fillna("").astype(str).str.lower()
    return pd.DataFrame({f"news_weekly_text_{key}": np.log1p(text.str.count(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b"))
                         for key, words in terms.items()}, index=text.index)


def build_daily_news_features(daily: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Whitelist summary counts, delay six actual sessions, then roll backward."""
    daily = daily[list(DAILY_NEWS_COLUMNS)].copy()
    daily["Date"] = _dates(daily.pop("trading_date"))
    if daily.Date.isna().any() or daily.Date.duplicated().any():
        raise ValueError("Daily news source dates must be unique and non-null.")
    index = pd.DatetimeIndex(calendar.Date)
    daily = daily.set_index("Date").reindex(index)
    counts = daily.apply(pd.to_numeric, errors="coerce")
    counts = counts.where(counts >= 0)
    observed = counts.event_count.notna()
    source_dates = pd.Series(index, index=index).where(observed).shift(NEWS_LAG_SESSIONS).ffill()
    delayed = counts.shift(NEWS_LAG_SESSIONS)
    share = (delayed.bullish_events - delayed.bearish_events) / delayed.event_count.replace(0, np.nan)
    age = (pd.Series(index, index=index) - source_dates).dt.days
    stale = age.isna() | age.gt(NEWS_MAX_SOURCE_AGE_DAYS)
    features = {}
    for window in (5, 20, 21, 28, 30, 60):
        for name, series in (("event_count_log", np.log1p(delayed.event_count)), ("bull_minus_bear_share", share)):
            features[f"news_daily_{name}_{window}s"] = series.rolling(window, min_periods=max(2, window // 2)).mean().where(~stale)
    features["news_daily_direction_change"] = features["news_daily_bull_minus_bear_share_5s"] - features["news_daily_bull_minus_bear_share_30s"]
    features["news_daily_volume_change"] = features["news_daily_event_count_log_5s"] - features["news_daily_event_count_log_30s"]
    features["news_daily_observed_fraction_30s"] = observed.astype(float).shift(NEWS_LAG_SESSIONS).rolling(30, min_periods=1).mean()
    features["news_daily_source_age_days"] = age
    features["news_daily_stale"] = stale.astype(float)
    return pd.DataFrame(features, index=index).rename_axis("Date").reset_index()


def build_weekly_news_features(weekly: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly[list(WEEKLY_NEWS_COLUMNS)].copy()
    ends = _dates(weekly["period_end"])
    if ends.isna().any() or ends.duplicated().any():
        raise ValueError("Weekly news period ends must be unique and non-null.")
    index = pd.DatetimeIndex(calendar.Date)
    last_session = index.searchsorted(ends, side="right") - 1
    available_session = last_session + NEWS_LAG_SESSIONS
    valid = (last_session >= 0) & (available_session < len(index)) & (ends <= index.max())
    weekly = weekly.loc[valid].copy()
    weekly["_source_date"] = ends.loc[valid]
    weekly["_available_date"] = index[available_session[valid]].to_numpy()
    for column in WEEKLY_NEWS_COLUMNS[1:-1]:
        weekly[column] = pd.to_numeric(weekly[column], errors="coerce")
        weekly[column] = weekly[column].where(weekly[column] >= 0)
    denominator = weekly.event_count.replace(0, np.nan)
    share = (weekly.bullish_event_count - weekly.bearish_event_count) / denominator
    features = pd.DataFrame({
        "_available_date": weekly._available_date, "_source_date": weekly._source_date,
        "news_weekly_event_count_log": np.log1p(weekly.event_count),
        "news_weekly_bull_minus_bear_share": share,
        "news_weekly_direction_concentration": share.abs(),
        "news_weekly_explicit_coffee_share": weekly.explicit_coffee_event_count / denominator,
        "news_weekly_mean_event_intensity": weekly.mean_event_intensity_score_0_1,
    })
    features = pd.concat([features, _text_news_features(weekly.top_event_digest)], axis=1)
    out = pd.merge_asof(calendar, features.sort_values("_available_date"), left_on="Date", right_on="_available_date", direction="backward")
    age = (out.Date - out._source_date).dt.days
    stale = age.isna() | age.gt(NEWS_MAX_SOURCE_AGE_DAYS)
    columns = [column for column in out if column.startswith("news_")]
    out.loc[stale, columns] = np.nan
    for window in (20, 21, 28, 30, 60):
        for name in ("event_count_log", "bull_minus_bear_share"):
            column = f"news_weekly_{name}"
            out[f"{column}_{window}s"] = out[column].rolling(window, min_periods=max(2, window // 2)).mean().where(~stale)
    out["news_weekly_source_age_days"] = age
    out["news_weekly_stale"] = stale.astype(float)
    return out.drop(columns=["_available_date", "_source_date"])


def build_external_features(prices: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict]:
    """Return Date + numeric weather_/news_ fields and a JSON-ready source audit.

    ``prices`` must contain the complete observed session calendar: news lags
    count those rows, not weekdays, generic business days, or calendar days.
    Missing values are retained for training-only imputation by the trainer.
    """
    root = Path(root)
    calendar = _calendar(prices)
    if calendar.empty:
        raise ValueError("At least one price session is required.")
    output_dir = root / "Scripts/project/artifacts/outputs"
    weather_path = root / "data/weather/open_meteo_coffee_regions_daily.csv"
    daily_path = output_dir / "gdelt_coffee_events_2000_2026_daily_summary.csv"
    weekly_path = output_dir / "gdelt_coffee_events_2000_2026_weekly_summary.csv"
    frame = calendar.copy()
    metadata = {
        "weather": {
            "path": str(weather_path.relative_to(root)), "available": weather_path.exists(),
            "availability_lag_calendar_days": WEATHER_LAG_DAYS,
            "maximum_carry_after_availability_calendar_days": WEATHER_MAX_RELEASE_AGE_DAYS,
            "alignment": "Daily calendar rollups; source Date + 5 calendar days; backward asof; stale weather values become NaN after 7 further days; no backward fill.",
            "climatology": "Previous five same-calendar-month occurrences only, excluding the entire current month/year; at least two prior months and 45 valid observations.",
            "vintage_limitation": "Local Open-Meteo historical reanalysis has no archived publication vintages. A 5-day availability proxy reduces timing leakage but cannot remove historical revisions. Results are conditional on this proxy.",
            "crop_interactions": "Broad Brazil flowering Sep-Nov, harvest May-Sep, winter Jun-Aug indicators are modeling hypotheses; not location-specific crop surveys. Vietnam is an indirect competing-origin proxy.",
        },
        "news": {
            "daily_path": str(daily_path.relative_to(root)), "weekly_path": str(weekly_path.relative_to(root)),
            "daily_available": daily_path.exists(), "weekly_available": weekly_path.exists(),
            "availability_lag_observed_sessions": NEWS_LAG_SESSIONS,
            "maximum_source_age_calendar_days": NEWS_MAX_SOURCE_AGE_DAYS,
            "daily_source_columns": list(DAILY_NEWS_COLUMNS), "weekly_source_columns": list(WEEKLY_NEWS_COLUMNS),
            "excluded": "All future-return, price-response, impact, deterministic-score and LLM-score columns. Embedded digest numbers are not parsed.",
            "status": "exploratory_retrospectively_selected_not_point_in_time_certified",
            "limitation": "Event retention/ranking used subsequent 5-session returns and year/full-history fallback return quantiles. Six observed sessions delay direct return maturity; it cannot reverse retrospective normalization, selection, revised data or missing publication snapshots. Keep news an optional exploratory ablation, excluded from the default deployable feature group.",
            "missingness": "Absent source rows remain missing, not zero-event days. Six-session lag uses the supplied complete price calendar. Weekly features expire after 35 calendar days from period end.",
        },
    }
    if weather_path.exists():
        frame, weather_audit = build_weather_features(pd.read_csv(weather_path), calendar)
        metadata["weather"].update(weather_audit)
    if daily_path.exists():
        daily = pd.read_csv(daily_path, usecols=list(DAILY_NEWS_COLUMNS))
        frame = frame.merge(build_daily_news_features(daily, calendar), on="Date", validate="one_to_one")
        metadata["news"]["daily_rows"] = len(daily)
        metadata["news"]["daily_span"] = _date_span(daily.trading_date)
    if weekly_path.exists():
        weekly = pd.read_csv(weekly_path, usecols=list(WEEKLY_NEWS_COLUMNS))
        frame = frame.merge(build_weekly_news_features(weekly, calendar), on="Date", validate="one_to_one")
        metadata["news"]["weekly_rows"] = len(weekly)
        metadata["news"]["weekly_span"] = _date_span(weekly.period_end)
    feature_columns = [column for column in frame if column != "Date"]
    frame[feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    metadata["feature_counts"] = {prefix: sum(column.startswith(prefix + "_") for column in feature_columns) for prefix in ("weather", "news")}
    metadata["feature_missing_fraction"] = frame[feature_columns].isna().mean().to_dict()
    return frame, metadata
