"""Fixed, causal directional feature ablations for the monthly-return experiment.

Feature definitions are chosen before validation.  Nothing in this module fits
a model, fills an unknown observation, or looks at a forward return.  Published
COT state is carried between its usable release sessions by an explicit as-of
join; this is report alignment, not general-purpose missing-value imputation.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from MonthFu.src.features import build_feature_frame
from MonthFu.src.modeling import add_targets


COT_GROUPS = (
    "legacy_noncommercial", "legacy_commercial", "disagg_managed_money", "disagg_producer",
)
WEATHER_REGIONS = ("brazil_minas_gerais", "colombia_huila")

# These are deliberately fixed lists, not feature rankings on the full dataset.
BASIC = [
    "price_return_1", "price_return_5", "price_return_21", "price_return_63",
    "price_vol_21", "price_vol_63", "price_vs_ma_21", "price_vs_ma_63",
    "price_rsi_30", "price_range_position_30", "price_drawdown_63",
    "price_signed_volume_21", "price_volume_relative_21", "price_gap_return",
    "price_intraday_return", "price_vol_ratio_21_63", "price_ema_spread_21_63",
    "price_season_sin_1", "price_season_cos_1", "price_low_volume",
]
PRICE_EXISTING = [
    f"price_{kind}_{window}" for window in (10, 30, 126)
    for kind in ("return", "vol", "trend_signal")
] + [
    "price_rsi_14", "price_skew_30", "price_vol_asymmetry_30",
    "price_directional_efficiency_30", "price_efficiency_30",
    "price_atr_21", "price_gap_variance_share_21", "price_parkinson_vol_21",
    "price_close_location", "price_ema_spread_5_21", "price_vol_of_vol_30",
    "price_month_progress", "price_days_to_month_end", "price_invalid_ohlc",
]
COT_EXISTING = [
    f"cot_{group}_{kind}" for group in COT_GROUPS
    for kind in ("net_oi", "net_change_1w", "net_change_4w", "net_z_52w", "index_52w")
] + [
    f"cot_{family}_{kind}" for family in ("legacy", "disagg")
    for kind in ("snapshot_age_days", "release_age_days", "dependency_delay_days",
                 "new_release_session", "stale_14d", "oi_change_4w")
]
WEATHER_EXISTING = [
    f"weather_{region}_{kind}" for region in WEATHER_REGIONS
    for kind in ("rain_seasonal_z_30d", "temp_seasonal_z_30d", "soil_seasonal_z_30d",
                 "dry_heat_stress_30d", "water_deficit_30d", "rain_acceleration",
                 "soil_change_30d", "source_age_days", "stale")
] + [
    "weather_brazil_minas_gerais_flowering_water_deficit",
    "weather_brazil_minas_gerais_harvest_rain",
    "weather_brazil_minas_gerais_winter_cold_risk",
]


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Absent optional inputs stay unknown, with the original row index."""
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _rank(series: pd.Series, window: int, minimum: int | None = None) -> pd.Series:
    """Trailing midrank of the current value, including only rows through t."""
    minimum = max(8, window // 2) if minimum is None else minimum

    def current_midrank(values: np.ndarray) -> float:
        if not np.isfinite(values[-1]):
            return np.nan
        finite = values[np.isfinite(values)]
        return float(((finite < values[-1]).sum() + .5 * (finite == values[-1]).sum()) / len(finite))

    return series.rolling(window, min_periods=minimum).apply(current_midrank, raw=True)


def _signed_streak(returns: pd.Series) -> pd.Series:
    """Up/down sessions in the current run; zero/missing interrupts a run."""
    signs = np.sign(returns.to_numpy(dtype=float))
    result = np.full(len(signs), np.nan)
    previous, length = 0.0, 0
    for i, sign in enumerate(signs):
        if not np.isfinite(sign):
            previous, length = 0.0, 0
            continue
        length = length + 1 if sign != 0 and sign == previous else int(sign != 0)
        result[i] = sign * length
        previous = sign
    return pd.Series(result, index=returns.index)


def _event_statistics(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Rank/persistence on observable release events, never duplicated weekdays.

    A catch-up release counts as the latest report actually available at that
    daily forecast origin.  Earlier reports not separately observed at a daily
    close are not invented.  The underlying monthly module continues to supply
    its original report-calendar weekly features.
    """
    family = group.split("_", 1)[0]
    net = _series(frame, f"cot_{group}_net_oi")
    flow = _series(frame, f"cot_{group}_net_change_1w")
    is_release = _series(frame, f"cot_{family}_new_release_session").eq(1)
    positions = np.flatnonzero(is_release.to_numpy())
    names = [f"direction_cot_{group}_{suffix}" for suffix in (
        "released_net_rank_13", "released_flow_rank_13", "released_flow_streak",
        "released_net_extreme_distance_26",
    )]
    if not len(positions):
        return pd.DataFrame(np.nan, index=frame.index, columns=names)
    events = pd.DataFrame({"Date": pd.to_datetime(frame.Date).iloc[positions].to_numpy(),
                           "net": net.iloc[positions].to_numpy(),
                           "flow": flow.iloc[positions].to_numpy()})
    roll = events.net.rolling(26, min_periods=13)
    midpoint = (roll.max() + roll.min()) / 2
    event_features = pd.DataFrame({
        "Date": events.Date,
        names[0]: _rank(events.net, 13, 7),
        names[1]: _rank(events.flow, 13, 7),
        names[2]: _signed_streak(events.flow),
        names[3]: _ratio(events.net - midpoint, (roll.max() - roll.min()) / 2),
    })
    aligned = pd.merge_asof(pd.DataFrame({"Date": pd.to_datetime(frame.Date).to_numpy()}),
                            event_features, on="Date", direction="backward")
    aligned.index = frame.index
    aligned = aligned[names]
    # Preserve unavailable/expired current source state.  Do not substitute a
    # previous release for an actual new release whose current values are NaN.
    aligned.loc[net.isna(), names] = np.nan
    aligned.loc[flow.isna(), names[1:3]] = np.nan
    return aligned


def add_direction_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Add causal direction features; discard inherited target/news contaminants.

    ``frame`` is the observed-session frame returned by build_feature_frame.
    No imputation occurs.  Returns engineered predictors and fixed family lists.
    Prefix rebuilding and future-input mutation must leave earlier values equal.
    """
    dates = pd.to_datetime(frame.Date)
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("Directional features require unique increasing session dates")
    forbidden = [c for c in frame if c.startswith(("target", "news_")) or "future_return" in c]
    frame = frame.drop(columns=forbidden).copy()
    close, volume = _series(frame, "Close"), _series(frame, "Volume")
    logprice = np.log(close.where(close > 0))
    ret = close.pct_change(fill_method=None)
    logret = logprice.diff()
    daily_sign = np.sign(ret)
    streak = _signed_streak(ret)
    atr = _series(frame, "price_atr_21")
    short_signal = _ratio(_series(frame, "price_return_5"), _series(frame, "price_vol_5") * np.sqrt(5))
    monthly_signal = _series(frame, "price_trend_signal_21")
    values: dict[str, pd.Series] = {}
    families: dict[str, list[str]] = {"price": [], "cot": [], "weather": []}

    def put(family: str, suffix: str, series: pd.Series) -> None:
        name = f"direction_{family}_{suffix}"
        values[name] = series.replace([np.inf, -np.inf], np.nan)
        families[family].append(name)

    put("price", "signed_streak_log", np.sign(streak) * np.log1p(streak.abs()))
    put("price", "streak_exhaustion", np.sign(streak) * streak.abs().clip(upper=10).pow(2) * ret.abs())
    put("price", "gap_intraday_disagreement", -np.sign(_series(frame, "price_gap_return")) * np.sign(_series(frame, "price_intraday_return")))
    put("price", "short_monthly_disagreement", -np.sign(short_signal) * np.sign(monthly_signal))
    put("price", "gap_intraday_balance", _ratio(_series(frame, "price_intraday_return") - _series(frame, "price_gap_return"), _series(frame, "price_vol_21")))
    put("price", "volume_pressure_change", _series(frame, "price_signed_volume_5") - _series(frame, "price_signed_volume_21"))
    put("price", "ema_acceleration", _series(frame, "price_ema_spread_5_21") - _series(frame, "price_ema_spread_5_21").shift(5))
    for window in (5, 21, 63):
        minimum = max(3, window // 2)
        direction = daily_sign.where(ret.notna())
        direction_mean = direction.rolling(window, min_periods=minimum).mean()
        up_var = ret.clip(lower=0).pow(2).rolling(window, min_periods=minimum).sum()
        down_var = ret.clip(upper=0).pow(2).rolling(window, min_periods=minimum).sum()
        put("price", f"signed_session_share_{window}", direction_mean)
        put("price", f"semivariance_balance_{window}", _ratio(up_var - down_var, up_var + down_var))
        reversals = daily_sign.ne(daily_sign.shift()).astype(float).where(ret.notna() & ret.shift().notna())
        put("price", f"reversal_share_{window}", reversals.rolling(window, min_periods=minimum).mean())
        put("price", f"direction_persistence_{window}", direction.rolling(window, min_periods=minimum).corr(direction.shift()))
        put("price", f"prior_high_distance_atr_{window}", _ratio(close / close.shift().rolling(window, min_periods=window).max() - 1, atr))
        put("price", f"prior_low_distance_atr_{window}", _ratio(close / close.shift().rolling(window, min_periods=window).min() - 1, atr))
        movement = logret.abs().rolling(window, min_periods=window).sum()
        put("price", f"signed_path_efficiency_{window}", _ratio(logprice.diff(window), movement))
        up_count = ret.gt(0).astype(float).where(ret.notna()).rolling(window, min_periods=minimum).sum()
        down_count = ret.lt(0).astype(float).where(ret.notna()).rolling(window, min_periods=minimum).sum()
        up_mean = _ratio(ret.clip(lower=0).rolling(window, min_periods=minimum).sum(), up_count)
        down_mean = _ratio(-ret.clip(upper=0).rolling(window, min_periods=minimum).sum(), down_count)
        put("price", f"up_down_magnitude_balance_{window}", _ratio(up_mean - down_mean, up_mean + down_mean))
        rolling_signed_volume = (np.sign(ret) * volume).rolling(window, min_periods=minimum).sum()
        rolling_volume = volume.rolling(window, min_periods=minimum).sum()
        put("price", f"volume_direction_divergence_{window}", _ratio(rolling_signed_volume, rolling_volume) - direction_mean)
    for momentum in (5, 21, 63):
        for history in (63, 252):
            put("price", f"momentum_{momentum}_rank_{history}", _rank(_series(frame, f"price_return_{momentum}"), history))
    for window in (21, 63):
        minimum = window // 2
        put("price", f"return_autocorrelation_lag5_{window}", ret.rolling(window, min_periods=minimum).corr(ret.shift(5)))
        local_vol = logret.rolling(window, min_periods=minimum).std()
        drift = logret.rolling(window, min_periods=minimum).mean()
        put("price", f"drift_tstat_{window}", _ratio(drift * np.sqrt(window), local_vol))
        prior_high = close.shift().rolling(window, min_periods=window).max()
        prior_low = close.shift().rolling(window, min_periods=window).min()
        breakout = close.gt(prior_high).astype(float) - close.lt(prior_low).astype(float)
        put("price", f"breakout_state_{window}", breakout.where(prior_high.notna() & prior_low.notna()))
        position = _ratio(close - prior_low, prior_high - prior_low)
        put("price", f"extreme_reversal_{window}", (position - .5).clip(-1, 1) * -short_signal)
        negative_after_up = ret.clip(upper=0).where(ret.shift().gt(0))
        positive_after_down = ret.clip(lower=0).where(ret.shift().lt(0))
        reversal_strength = (positive_after_down.rolling(window, min_periods=3).mean()
                             + negative_after_up.rolling(window, min_periods=3).mean())
        put("price", f"reversal_strength_{window}", _ratio(reversal_strength, local_vol))
    # Fixed calendar hypotheses; no season coefficients learned on full history.
    season_flags = {
        "brazil_flowering": dates.dt.month.isin([9, 10, 11]).astype(float),
        "brazil_harvest": dates.dt.month.isin([5, 6, 7, 8, 9]).astype(float),
        "brazil_winter": dates.dt.month.isin([6, 7, 8]).astype(float),
    }
    for season, flag in season_flags.items():
        put("price", f"{season}_trend", flag * monthly_signal)
        put("price", f"{season}_reversal", flag * -short_signal)

    event_parts = []
    for group in COT_GROUPS:
        family = group.split("_", 1)[0]
        index = _series(frame, f"cot_{group}_index_52w")
        flow1 = _series(frame, f"cot_{group}_net_change_1w")
        flow4 = _series(frame, f"cot_{group}_net_change_4w")
        age = _series(frame, f"cot_{family}_release_age_days")
        put("cot", f"{group}_crowded_long", (index - .8).clip(lower=0))
        put("cot", f"{group}_crowded_short", (.2 - index).clip(lower=0))
        put("cot", f"{group}_fresh_flow", flow1 * np.exp(-age.clip(lower=0) / 7))
        put("cot", f"{group}_flow_reversal", (-np.sign(flow1) * np.sign(flow4)).where(flow1.notna() & flow4.notna()))
        put("cot", f"{group}_price_flow_divergence", -np.sign(monthly_signal) * np.sign(flow4))
        put("cot", f"{group}_crowding_reversal", (index - .5) * -short_signal)
        put("cot", f"{group}_fresh_flow_trend", flow4 * monthly_signal * np.exp(-age.clip(lower=0) / 14))
        put("cot", f"{group}_flow_acceleration", flow1 - flow4 / 4)
        event = _event_statistics(frame, group)
        families["cot"].extend(event.columns.tolist())
        event_parts.append(event)
    for family in ("legacy", "disagg"):
        release_age = _series(frame, f"cot_{family}_release_age_days")
        snapshot_age = _series(frame, f"cot_{family}_snapshot_age_days")
        put("cot", f"{family}_freshness", np.exp(-release_age.clip(lower=0) / 7))
        put("cot", f"{family}_publication_delay", snapshot_age - release_age)

    severity_values = {}
    for region in WEATHER_REGIONS:
        prefix = f"weather_{region}_"
        fresh = _series(frame, prefix + "stale").eq(0)
        rain_z = _series(frame, prefix + "rain_seasonal_z_30d").where(fresh)
        temp_z = _series(frame, prefix + "temp_seasonal_z_30d").where(fresh)
        soil_z = _series(frame, prefix + "soil_seasonal_z_30d").where(fresh)
        # All three must be known; missing components are never silently zero.
        severity = (temp_z.clip(lower=0, upper=6) + (-rain_z).clip(lower=0, upper=6)
                    + (-soil_z).clip(lower=0, upper=6))
        severity_values[region] = severity
        put("weather", f"{region}_drought_heat_severity", severity)
        put("weather", f"{region}_excess_rain_severity", rain_z.clip(lower=0, upper=6))
        put("weather", f"{region}_dry_soil_compound", (-rain_z).clip(lower=0, upper=6) * (-soil_z).clip(lower=0, upper=6))
        put("weather", f"{region}_heat_soil_compound", temp_z.clip(lower=0, upper=6) * (-soil_z).clip(lower=0, upper=6))
        put("weather", f"{region}_water_deficit_log", np.log1p(_series(frame, prefix + "water_deficit_30d").clip(lower=0)).where(fresh))
        put("weather", f"{region}_cold_severity", ((5 - _series(frame, prefix + "min_temperature_7d")) / 5).clip(lower=0).where(fresh))
        put("weather", f"{region}_stress_rank_126", _rank(severity, 126).where(fresh))
        for lag in (5, 21):
            put("weather", f"{region}_stress_change_{lag}", (severity - severity.shift(lag)).where(fresh))
            put("weather", f"{region}_rain_z_change_{lag}", (rain_z - rain_z.shift(lag)).where(fresh))
        put("weather", f"{region}_stress_trend_divergence", severity * -monthly_signal)
        put("weather", f"{region}_stress_short_reversal", severity * -short_signal)
        if region == "brazil_minas_gerais":
            for season, flag in season_flags.items():
                put("weather", f"{region}_{season}_stress", severity * flag)
    brazil, colombia = [severity_values[region] for region in WEATHER_REGIONS]
    put("weather", "arabica_severity_mean", (brazil + colombia) / 2)
    put("weather", "arabica_severity_difference", brazil - colombia)
    put("weather", "arabica_stress_agreement", np.minimum(brazil, colombia))
    added = pd.concat([pd.DataFrame(values, index=frame.index), *event_parts], axis=1)
    result = pd.concat([frame, added], axis=1)
    return result, families


def cot_feature_columns(frame: pd.DataFrame) -> list[str]:
    """The complete source COT schema, including sparse/constant numeric fields."""
    return [name for name in frame if name.startswith("cot_")
            and pd.api.types.is_numeric_dtype(frame[name])]


def cot_schema_hash(columns: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(columns), separators=(",", ":")).encode()).hexdigest()


def cot_source_policy(metadata: dict) -> dict:
    """Stable timing/schema decisions; omit changing report counts and dates."""
    cot = metadata.get("cot", {})
    return {key: cot.get(key) for key in ("alignment", "normal_release_rule", "weekly_windows",
        "feature_schema_version", "backcast_policy", "feature_policy")}


def _groups(frame: pd.DataFrame, families: dict[str, list[str]],
            cot_policy: str = "benchmark") -> dict[str, list[str]]:
    if cot_policy not in {"all", "benchmark"}:
        raise ValueError("COT policy must be all or benchmark")
    core_price = [
        "signed_streak_log", "short_monthly_disagreement", "volume_pressure_change",
        "signed_session_share_21", "semivariance_balance_21", "reversal_share_21",
        "prior_high_distance_atr_21", "prior_low_distance_atr_21", "signed_path_efficiency_21",
        "momentum_5_rank_252", "momentum_21_rank_252", "momentum_63_rank_252",
        "drift_tstat_21", "drift_tstat_63", "extreme_reversal_21", "brazil_flowering_trend",
    ]
    core_cot = [
        f"direction_cot_{group}_{kind}" for group in ("legacy_noncommercial", "disagg_managed_money")
        for kind in ("fresh_flow", "flow_reversal", "price_flow_divergence", "crowding_reversal", "released_net_rank_13")
    ]
    core_weather = [
        f"direction_weather_{region}_{kind}" for region in WEATHER_REGIONS
        for kind in ("drought_heat_severity", "stress_change_5", "excess_rain_severity")
    ]
    groups = {
        "basic": BASIC,
        "direction_core": BASIC + [f"direction_price_{kind}" for kind in core_price]
                          + [f"cot_{group}_{kind}" for group in ("legacy_noncommercial", "disagg_managed_money")
                             for kind in ("net_oi", "net_change_4w", "index_52w")]
                          + core_cot + core_weather,
        "price_direction": BASIC + PRICE_EXISTING + families["price"],
        "cot_direction": BASIC + COT_EXISTING + families["cot"],
        "weather_direction": BASIC + WEATHER_EXISTING + families["weather"],
        "engineered_direction": BASIC + PRICE_EXISTING + COT_EXISTING + WEATHER_EXISTING
                                + families["price"] + families["cot"] + families["weather"],
    }
    result = {name: list(dict.fromkeys(c for c in columns if c in frame)) for name, columns in groups.items()}
    for columns in result.values():
        if any(c.startswith(("target", "news_")) or "future_return" in c for c in columns):
            raise AssertionError("Targets and retrospective news cannot enter directional feature groups")
    if len(result["engineered_direction"]) > 300:
        raise AssertionError("Fixed curated bank exceeds the experiment's 300-feature limit")
    if cot_policy == "all":
        cot_columns = cot_feature_columns(frame)
        if not cot_columns:
            raise ValueError("All-COT experiments require a nonempty numeric COT schema")
        # Three fixed contrasts bound the search while every contrast includes
        # the entire COT source bank.  The curated 300-column check above applies
        # to the original directional additions, not to required source fields.
        result = {name: list(dict.fromkeys(result[name] + cot_columns))
                  for name in ("basic", "direction_core", "engineered_direction")}
    return result


def build_experiment_frame(repo_root: Path, horizon: int = 30, unit: str = "calendar",
                           cot_policy: str = "benchmark") -> tuple[pd.DataFrame, dict[str, list[str]], dict]:
    """Return observed predictors + monthly targets, six groups and JSON metadata.

    Reuses the unchanged MonthFu publication-aware source pipeline.  Targets are
    attached after features/groups exist.  All model transforms and imputation
    must subsequently be fitted inside purged, label-mature training folds.
    """
    base, _, audits = build_feature_frame(Path(repo_root))
    frame, families = add_direction_features(base)
    groups = _groups(frame, families, cot_policy)
    selected = list(dict.fromkeys(["Date", "Open", "High", "Low", "Close", "Volume"]
                                 + groups["engineered_direction"] + groups["direction_core"]))
    frame = frame[[c for c in selected if c in frame]]
    result = add_targets(frame, horizon, unit)
    metadata = audits["metadata"] | {
        "horizon": horizon, "horizon_unit": unit,
        "cot_policy": cot_policy,
        "cot_feature_names": cot_feature_columns(frame),
        "cot_feature_schema_sha256": cot_schema_hash(cot_feature_columns(frame)),
        "cot_source_policy": "MonthFu availability-aware COT builder; metadata.cot records release/backcast decisions",
        "feature_counts": {name: len(columns) for name, columns in groups.items()},
        "new_feature_counts": {name: len(columns) for name, columns in families.items()},
        "feature_manifest": [
            {"feature": column, "groups": [name for name, columns in groups.items() if column in columns],
             "family": column.split("_")[1] if column.startswith("direction_") else column.split("_")[0],
             "missing_fraction": float(frame[column].isna().mean())}
            for column in groups["engineered_direction"]
        ],
        "experiment_feature_policy": "Fixed definitions and curated groups before model selection; trailing ranks/moments; targets attached last; no target-informed full-history screening, news, backward fill, or general forward-fill imputation.",
        "cot_event_policy": "New release statistics use first-usable observable session events, then backward as-of that specific release state. Missing current release values stay missing. Existing weekly/report features and publication/revision rules are reused unchanged.",
        "weather_policy": "Reuse existing five-calendar-day historical-reanalysis availability proxy and stale expiry; every new current stress value is masked if its current weather is stale. No assertion of original weather vintages.",
        "signal_time": "After daily observed session close; target starts at that close. COT published after the Coffee C close enters strictly at the next observed session.",
    }
    return result, groups, metadata
