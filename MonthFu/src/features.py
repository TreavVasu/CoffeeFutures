"""Trailing monthly return features built from the repository's original inputs.

Forecast origins are after the observed session close. This module never reads a
target and never estimates a statistic from validation or future observations.
The large feature bank is paired with fixed compact groups so model selection
can test useful engineering without forcing every candidate to use every field.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRICE_RETURN_WINDOWS = (1, 2, 3, 5, 7, 10, 14, 20, 21, 28, 30, 42, 60, 63, 90, 126, 252)
PRICE_ROLLING_WINDOWS = (5, 10, 20, 21, 28, 30, 60, 63, 90, 126, 252)


def load_prices(path: Path) -> tuple[pd.DataFrame, dict]:
    """Retain valid positive closes and mask inconsistent ancillary OHLCV.

    We preserve the actual market calendar. No missing sessions, closes or
    prices are interpolated, and no future row repairs a previous observation.
    """
    frame = pd.read_csv(path)
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(frame):
        raise ValueError(f"Prices must contain {sorted(required)}")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = frame.Date.notna() & np.isfinite(frame.Close) & frame.Close.gt(0)
    rejected = int((~valid).sum())
    frame = frame.loc[valid].sort_values("Date").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid positive-close price observations")
    if frame.Date.duplicated().any():
        raise ValueError("Duplicate price dates: resolve source conflicts before training.")
    bad_ohlc = (frame.High.lt(frame[["Open", "Close"]].max(axis=1))
                | frame.Low.gt(frame[["Open", "Close"]].min(axis=1))
                | frame[["Open", "High", "Low"]].le(0).any(axis=1)
                | ~np.isfinite(frame[["Open", "High", "Low"]]).all(axis=1))
    low_volume = frame.Volume.le(1) | ~np.isfinite(frame.Volume)
    frame["price_invalid_ohlc"] = bad_ohlc.astype(float)
    frame["price_low_volume"] = low_volume.astype(float)
    frame.loc[bad_ohlc, ["Open", "High", "Low"]] = np.nan
    frame.loc[low_volume, "Volume"] = np.nan
    # Do not accidentally admit source-added identifiers or future-return fields.
    frame = frame[["Date", "Open", "High", "Low", "Close", "Volume", "price_invalid_ohlc", "price_low_volume"]]
    audit = {
        "source": str(path), "rows": len(frame), "rejected_rows": rejected,
        "first_date": str(frame.Date.min().date()), "last_date": str(frame.Date.max().date()),
        "invalid_ohlc_rows_masked": int(bad_ohlc.sum()),
        "zero_or_one_volume_rows_masked": int(low_volume.sum()),
        "policy": "Positive-close observed sessions only; invalid OHLC and volume <= 1 are missing with quality flags. No interpolation, backwards fill, or certification of contract rolls.",
    }
    return frame, audit


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Construct fixed price/volume/calendar features using rows through t only."""
    close, volume = prices.Close, prices.Volume
    dates = pd.to_datetime(prices.Date)
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Price features require unique increasing session dates")
    ret, logret = close.pct_change(fill_method=None), np.log(close).diff()
    logprice = np.log(close)
    features = {
        "Date": dates,
        "price_invalid_ohlc": prices.get("price_invalid_ohlc", pd.Series(0.0, index=prices.index)),
        "price_low_volume": prices.get("price_low_volume", pd.Series(0.0, index=prices.index)),
    }
    for window in PRICE_RETURN_WINDOWS:
        features[f"price_return_{window}"] = close.pct_change(window, fill_method=None)
    for lag in [1, 2, 3, 5, 10, 21, 28, 30]:
        features[f"price_daily_return_lag_{lag}"] = ret.shift(lag)

    for window in PRICE_ROLLING_WINDOWS:
        minimum = max(4, window // 2)
        rolling = close.rolling(window, min_periods=minimum)
        vol = logret.rolling(window, min_periods=minimum).std()
        features[f"price_vol_{window}"] = vol
        features[f"price_vs_ma_{window}"] = _ratio(close, rolling.mean()) - 1
        features[f"price_drawdown_{window}"] = _ratio(close, rolling.max()) - 1
        features[f"price_rebound_{window}"] = _ratio(close, rolling.min()) - 1
        features[f"price_range_position_{window}"] = _ratio(close - rolling.min(), rolling.max() - rolling.min())
        features[f"price_volume_relative_{window}"] = _ratio(volume, volume.rolling(window, min_periods=minimum).median()) - 1
        features[f"price_trend_signal_{window}"] = _ratio(close.pct_change(window, fill_method=None), vol * np.sqrt(window))

    for window in [14, 21, 28, 30, 63]:
        minimum = max(7, window // 2)
        rolling = ret.rolling(window, min_periods=minimum)
        gain = ret.clip(lower=0).rolling(window, min_periods=minimum).mean()
        loss = (-ret.clip(upper=0)).rolling(window, min_periods=minimum).mean()
        features[f"price_rsi_{window}"] = _ratio(gain, gain + loss)
        downside = np.sqrt(ret.clip(upper=0).pow(2).rolling(window, min_periods=minimum).mean())
        upside = np.sqrt(ret.clip(lower=0).pow(2).rolling(window, min_periods=minimum).mean())
        features[f"price_downside_{window}"] = downside
        features[f"price_upside_{window}"] = upside
        features[f"price_vol_asymmetry_{window}"] = _ratio(upside - downside, upside + downside)
        features[f"price_skew_{window}"] = rolling.skew()
        features[f"price_kurtosis_{window}"] = rolling.kurt()
        features[f"price_positive_share_{window}"] = ret.gt(0).astype(float).where(ret.notna()).rolling(window, min_periods=minimum).mean()
        movement = close.diff().abs().rolling(window, min_periods=window).sum()
        features[f"price_efficiency_{window}"] = _ratio(close.diff(window).abs(), movement)
        features[f"price_directional_efficiency_{window}"] = _ratio(close.diff(window), movement)
        features[f"price_autocorrelation_{window}"] = rolling.corr(ret.shift(1))
        # Fixed linear slope on trailing log prices, scaled by realized volatility.
        centered_x = np.arange(window, dtype=float) - (window - 1) / 2
        x_ss = float(centered_x @ centered_x)
        slope = logprice.rolling(window, min_periods=window).apply(lambda y: float(centered_x @ y) / x_ss, raw=True)
        features[f"price_slope_signal_{window}"] = _ratio(slope * np.sqrt(window), logret.rolling(window, min_periods=minimum).std())

    gap = prices.Open / close.shift(1) - 1
    intraday = close / prices.Open - 1
    high_low_log = np.log(prices.High / prices.Low)
    close_open_log = np.log(close / prices.Open)
    true_range = pd.concat([prices.High - prices.Low, (prices.High - close.shift(1)).abs(),
                            (prices.Low - close.shift(1)).abs()], axis=1).max(axis=1) / close
    features["price_range_pct"] = (prices.High - prices.Low) / close
    features["price_intraday_return"] = intraday
    features["price_gap_return"] = gap
    features["price_close_location"] = _ratio(close - prices.Low, prices.High - prices.Low)
    for window in [5, 21, 28, 30, 63]:
        minimum = max(3, window // 2)
        features[f"price_atr_{window}"] = true_range.rolling(window, min_periods=minimum).mean()
        features[f"price_gap_mean_{window}"] = gap.rolling(window, min_periods=minimum).mean()
        features[f"price_gap_rms_{window}"] = np.sqrt(gap.pow(2).rolling(window, min_periods=minimum).mean())
        features[f"price_intraday_vol_{window}"] = intraday.rolling(window, min_periods=minimum).std()
        total_var = gap.pow(2) + intraday.pow(2)
        features[f"price_gap_variance_share_{window}"] = _ratio(gap.pow(2).rolling(window, min_periods=minimum).sum(), total_var.rolling(window, min_periods=minimum).sum())
        features[f"price_parkinson_vol_{window}"] = np.sqrt(high_low_log.pow(2).rolling(window, min_periods=minimum).mean() / (4 * np.log(2)))
        gk_var = (0.5 * high_low_log.pow(2) - (2 * np.log(2) - 1) * close_open_log.pow(2))
        features[f"price_garman_klass_vol_{window}"] = np.sqrt(gk_var.rolling(window, min_periods=minimum).mean().clip(lower=0))
        features[f"price_signed_volume_{window}"] = _ratio((np.sign(ret) * volume).rolling(window, min_periods=minimum).sum(), volume.rolling(window, min_periods=minimum).sum())
        features[f"price_illiquidity_{window}"] = (1e6 * ret.abs() / volume).rolling(window, min_periods=minimum).mean()
        features[f"price_volume_return_corr_{window}"] = ret.rolling(window, min_periods=minimum).corr(np.log1p(volume))
        features[f"price_abs_return_volume_corr_{window}"] = ret.abs().rolling(window, min_periods=minimum).corr(np.log1p(volume))

    for short, long in [(5, 21), (10, 63), (21, 63), (30, 126)]:
        features[f"price_vol_ratio_{short}_{long}"] = _ratio(features[f"price_vol_{short}"], features[f"price_vol_{long}"])
    for window in [21, 28, 30]:
        features[f"price_momentum_acceleration_{window}"] = features[f"price_return_{window}"] - features[f"price_return_{window}"].shift(window)
        features[f"price_momentum_change_5_{window}"] = features[f"price_return_5"] - features[f"price_return_{window}"] / (window / 5)
    features["price_momentum_acceleration"] = features["price_momentum_acceleration_30"]
    features["price_vol_ratio_10_60"] = _ratio(features["price_vol_10"], features["price_vol_60"])
    features["price_volume_return_30"] = (ret * np.log1p(volume)).rolling(30, min_periods=15).mean()
    features["price_ema_spread_21_63"] = _ratio(close.ewm(span=21, min_periods=21, adjust=False).mean(), close.ewm(span=63, min_periods=63, adjust=False).mean()) - 1
    features["price_ema_spread_5_21"] = _ratio(close.ewm(span=5, min_periods=5, adjust=False).mean(), close.ewm(span=21, min_periods=21, adjust=False).mean()) - 1
    features["price_vol_of_vol_30"] = pd.Series(features["price_vol_5"]).rolling(30, min_periods=15).std()
    features["price_volume_log_change_21"] = np.log1p(volume) - np.log1p(volume).shift(21)
    features["price_session_gap_days"] = dates.diff().dt.days.astype(float)
    features["price_month_progress"] = dates.dt.day / dates.dt.days_in_month
    features["price_days_to_month_end"] = (dates + pd.offsets.MonthEnd(0) - dates).dt.days.astype(float)
    features["price_quarter_end_month"] = dates.dt.month.isin([3, 6, 9, 12]).astype(float)
    features["price_weekday_sin"] = np.sin(2 * np.pi * dates.dt.dayofweek / 5)
    features["price_weekday_cos"] = np.cos(2 * np.pi * dates.dt.dayofweek / 5)
    for harmonic in [1, 2]:
        features[f"price_season_sin_{harmonic}"] = np.sin(2 * np.pi * harmonic * dates.dt.dayofyear / 365.25)
        features[f"price_season_cos_{harmonic}"] = np.cos(2 * np.pi * harmonic * dates.dt.dayofyear / 365.25)
    return pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)


def interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed economic interactions; no target correlations select these fields."""
    values = {}
    groups = ["disagg_managed_money", "disagg_producer", "legacy_noncommercial", "legacy_commercial"]
    for group in groups:
        position = f"cot_{group}_net_oi"
        flow = f"cot_{group}_net_change_4w"
        zscore = f"cot_{group}_net_z_52w"
        for horizon in [21, 28, 30]:
            if position in frame:
                values[f"interaction_{group}_position_trend_{horizon}"] = frame[position] * frame[f"price_return_{horizon}"]
            if flow in frame:
                values[f"interaction_{group}_flow_trend_{horizon}"] = frame[flow] * frame[f"price_trend_signal_{horizon}"]
        if zscore in frame:
            values[f"interaction_{group}_crowding_reversal"] = frame[zscore] * (frame.price_rsi_30 - 0.5)
        if position in frame and flow in frame:
            values[f"interaction_{group}_position_flow"] = frame[position] * frame[flow]
    for window in [21, 28, 30]:
        values[f"interaction_trend_vol_regime_{window}"] = frame[f"price_trend_signal_{window}"] * frame.price_vol_ratio_21_63
        values[f"interaction_volume_trend_{window}"] = frame[f"price_signed_volume_{window}"] * frame[f"price_return_{window}"]
        values[f"interaction_gap_trend_{window}"] = frame[f"price_gap_mean_{window}"] * frame[f"price_trend_signal_{window}"]
    for region in ["brazil_minas_gerais", "colombia_huila"]:
        for stress in ["dry_heat_stress_30d", "water_deficit_30d", "rain_seasonal_z_30d", "soil_seasonal_z_30d"]:
            column = f"weather_{region}_{stress}"
            if column in frame:
                values[f"interaction_{region}_{stress}_trend"] = frame[column] * frame.price_trend_signal_21
                values[f"interaction_{region}_{stress}_season"] = frame[column] * frame.price_season_sin_1
    return pd.DataFrame(values, index=frame.index).replace([np.inf, -np.inf], np.nan)


def _feature_groups(frame: pd.DataFrame, market: pd.DataFrame, cot: pd.DataFrame, external: pd.DataFrame,
                    interactions: pd.DataFrame) -> dict[str, list[str]]:
    price = [c for c in market if c != "Date"]
    cot_cols = [c for c in cot if c.startswith("cot_") and pd.api.types.is_numeric_dtype(cot[c])]
    weather = [c for c in external if c.startswith("weather_") and pd.api.types.is_numeric_dtype(external[c])]
    news = [c for c in external if c.startswith("news_") and pd.api.types.is_numeric_dtype(external[c])]
    compact_price = ["price_return_1", "price_return_5", "price_return_20", "price_vol_20",
                     "price_vs_ma_60", "price_range_pct", "price_intraday_return", "price_volume_relative_20"]
    compact_cot = [f"cot_{group}_net_oi" for group in ["legacy_noncommercial", "legacy_commercial", "disagg_managed_money", "disagg_producer"]]
    compact_weather = [f"weather_{region}_{name}_30d" for region in ["brazil_minas_gerais", "colombia_huila"] for name in ["rain", "temp", "dry_share"]]
    previous = ["Open", "High", "Low", "Close", "Volume"] + compact_price + [
        "price_vs_ma_20", "price_gap_return", "price_rsi_14", "price_season_sin_1", "price_season_cos_1"]
    core_price = [f"price_return_{w}" for w in [1, 5, 10, 21, 28, 30, 63, 126]] + [
        f"price_{signal}_{w}" for signal in ["vol", "vs_ma", "drawdown", "volume_relative", "trend_signal"] for w in [21, 30, 63]] + [
        "price_rsi_30", "price_skew_30", "price_downside_30", "price_efficiency_30", "price_signed_volume_30",
        "price_illiquidity_30", "price_gap_mean_30", "price_gap_variance_share_30", "price_parkinson_vol_30",
        "price_vol_ratio_21_63", "price_ema_spread_21_63", "price_vol_of_vol_30", "price_momentum_acceleration_30",
        "price_season_sin_1", "price_season_cos_1", "price_month_progress", "price_session_gap_days"]
    core_cot = [f"cot_{group}_{name}" for group in ["legacy_noncommercial", "legacy_commercial", "disagg_managed_money", "disagg_producer"]
                for name in ["net_oi", "net_change_1w", "net_change_4w", "net_z_52w", "index_52w"]]
    core_cot += [c for c in cot_cols if any(k in c for k in ["report_age", "release_age", "value_age", "dependency_delay", "known_revised", "stale", "oi_change_4w"])]
    core_weather = [f"weather_{region}_{name}" for region in ["brazil_minas_gerais", "colombia_huila"] for name in [
        "rain_seasonal_z_30d", "temp_seasonal_z_30d", "soil_seasonal_z_30d", "water_balance_60d", "dry_heat_stress_30d",
        "source_age_days", "stale"]]
    core_weather += ["weather_brazil_minas_gerais_flowering_water_deficit", "weather_brazil_minas_gerais_harvest_rain",
                     "weather_brazil_minas_gerais_winter_cold_risk"]
    core_interactions = [c for c in interactions if c.endswith("_trend_21") or c.endswith("crowding_reversal")]
    core_interactions += ["interaction_trend_vol_regime_21", "interaction_volume_trend_21"]
    groups = {
        "previous_technical": previous,
        "previous_history": [c for c in previous if c not in {"Open", "High", "Low", "Close", "Volume"}],
        "compact_legacy": compact_price + compact_cot + compact_weather,
        "price": price,
        "price_cot": price + cot_cols,
        "price_cot_weather": price + cot_cols + weather,
        "monthly_core": core_price + core_cot + core_weather + core_interactions,
        "engineered": price + cot_cols + weather + list(interactions),
        "experimental_news": price + cot_cols + weather + list(interactions) + news,
    }
    return {name: list(dict.fromkeys(c for c in columns if c in frame)) for name, columns in groups.items()}


def build_feature_frame(repo_root: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Return an origin-only frame, fixed feature manifest, and source audits.

    Frame contains Date, OHLCV and predictors. Manifest is JSON-ready and includes
    ``groups`` and per-column missingness. Audits contains a separate DataFrame
    under ``cot_release_audit`` for CSV export plus JSON-ready ``metadata``.
    """
    try:
        from .cot_release_features import build_cot_features
        from .external_return_features import build_external_features
    except ImportError:
        from cot_release_features import build_cot_features
        from external_return_features import build_external_features
    repo_root = Path(repo_root)
    prices, price_audit = load_prices(repo_root / "data/yahoo/arabica_coffee_futures_history.csv")
    cot, cot_audit, cot_meta = build_cot_features(prices, repo_root)
    external, external_meta = build_external_features(prices, repo_root)
    market = price_features(prices)
    frame = prices.drop(columns=["price_invalid_ohlc", "price_low_volume"])
    for part in [market, cot, external]:
        frame = frame.merge(part, on="Date", how="left", validate="one_to_one")
    interactions = interaction_features(frame)
    frame = pd.concat([frame, interactions], axis=1)
    groups = _feature_groups(frame, market, cot, external, interactions)
    columns = list(dict.fromkeys(c for group in groups.values() for c in group))
    frame[columns] = frame[columns].replace([np.inf, -np.inf], np.nan)
    assert not any(c.startswith("target") or "future_return" in c for c in columns)
    descriptions = []
    for column in columns:
        family = column.split("_")[0] if "_" in column else "raw_price"
        descriptions.append({"feature": column, "family": family,
                             "groups": [group for group, group_columns in groups.items() if column in group_columns],
                             "missing_fraction": float(frame[column].isna().mean()),
                             "experimental": column.startswith("news_")})
    manifest = {"groups": groups, "columns": descriptions, "feature_counts": {k: len(v) for k, v in groups.items()},
                "selection_policy": "Fixed groups; variance, availability, imputation and supervised screening must be fitted only inside each training fold."}
    metadata = {"price": price_audit, "cot": cot_meta, "external": external_meta,
                "feature_counts": manifest["feature_counts"],
                "signal_time": "After daily close; contemporaneous daily OHLCV known. COT publication is later than Coffee C close, hence next observed session. Weather five-day reanalysis proxy. News experimental.",
                "feature_policy": "Only trailing observations; no backwards fill, target-based full-history selection, or full-history scaling. OHLCV enters only previous_technical."}
    return frame, manifest, {"cot_release_audit": cot_audit, "metadata": metadata}


def build_frame(repo_root: Path, horizon: int = 30, unit: str = "calendar") -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    """Compatibility entry point for the training/evaluation module."""
    try:
        from .modeling import add_targets
    except ImportError:
        from modeling import add_targets
    frame, manifest, audits = build_feature_frame(repo_root)
    result = add_targets(frame, horizon, unit)
    metadata = audits["metadata"] | {"horizon": horizon, "horizon_unit": unit, "feature_manifest": manifest["columns"]}
    return result, manifest["groups"], audits["cot_release_audit"], metadata
