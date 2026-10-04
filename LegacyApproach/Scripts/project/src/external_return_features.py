"""Lagged local weather features.

All dates are daily observation dates, and predictions are made after the market
close. Weather is historical reanalysis, not a publication-vintage archive: the
five-calendar-day lag is a conservative proxy, not proof of point-in-time data.

The former news/GDELT layer was removed. Its upstream event selection used
subsequent returns and full-history normalization, so delaying it could not undo
that selection, and no primary candidate used it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WEATHER_LAG_DAYS = 5
WEATHER_MAX_RELEASE_AGE_DAYS = 7
REGIONS = ("brazil_minas_gerais", "colombia_huila", "vietnam_dak_lak")
WEATHER_WINDOWS = (7, 30, 60, 90)


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
    values = {}
    for window in WEATHER_WINDOWS:
        minimum = max(3, int(np.ceil(window * 0.8)))
        for name, series, aggregation in (
            ("temp", temp, "mean"), ("temp_max", high, "mean"),
            ("rain", rain, "sum"), ("soil", soil, "mean"),
            ("vpd", vpd, "mean"), ("water_balance", water_balance, "sum"),
            ("temp_seasonal_z", temp_z, "mean"), ("rain_seasonal_z", rain_z, "mean"),
            ("dry_share", rain.lt(1).astype(float).where(rain.notna()), "mean"),
            ("heat_degrees", (high - 30).clip(lower=0), "mean"),
            ("cold_share", low.lt(5).astype(float).where(low.notna()), "mean"),
        ):
            values[f"{prefix}{name}_{window}d"] = getattr(series.rolling(window, min_periods=minimum), aggregation)()
    values[prefix + "dry_spell_days"] = _dry_spell(rain)
    values[prefix + "rain_acceleration"] = values[prefix + "rain_7d"] / 7 - values[prefix + "rain_30d"] / 30
    values[prefix + "soil_change_30d"] = values[prefix + "soil_7d"] - values[prefix + "soil_7d"].shift(30)
    values[prefix + "heat_vpd_stress_30d"] = values[prefix + "heat_degrees_30d"] * values[prefix + "vpd_30d"]
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
    return out, {"rows": len(weather), "regions": regions}


def build_external_features(prices: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict]:
    """Return Date + numeric weather_/news_ fields and a JSON-ready source audit.

    ``prices`` must contain the complete observed session calendar so weather
    availability is expressed in real sessions, not weekdays or calendar days.
    Missing values are retained for training-only imputation by the trainer.
    """
    root = Path(root)
    calendar = _calendar(prices)
    if calendar.empty:
        raise ValueError("At least one price session is required.")
    weather_path = root / "../DATA/weather/open_meteo_coffee_regions_daily.csv"
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
        "excluded_sources": {
            "news": "The GDELT event layer was removed. Its retention/ranking used subsequent 5-session returns and full-history normalization, which a delay cannot reverse, and no primary candidate group used it.",
        },
    }
    if weather_path.exists():
        frame, weather_audit = build_weather_features(pd.read_csv(weather_path), calendar)
        metadata["weather"].update(weather_audit)
    feature_columns = [column for column in frame if column != "Date"]
    frame[feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    metadata["feature_counts"] = {"weather": sum(column.startswith("weather_") for column in feature_columns)}
    metadata["feature_missing_fraction"] = frame[feature_columns].isna().mean().to_dict()
    return frame, metadata
