"""Thirty-day return features and chronological evaluation primitives.

Signals are formed after the observation's daily close. Labels and their actual
end dates are separate from features; every fitting boundary uses label maturity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research_models import ExtremeLearningMachineRegressor


def load_prices(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    valid = frame.Date.notna() & np.isfinite(frame.Close) & frame.Close.gt(0)
    rejected = int((~valid).sum())
    frame = frame.loc[valid].sort_values("Date").reset_index(drop=True)
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
    return frame, {
        "source": str(path), "rows": len(frame), "rejected_rows": rejected,
        "first_date": str(frame.Date.min().date()), "last_date": str(frame.Date.max().date()),
        "invalid_ohlc_rows_masked": int(bad_ohlc.sum()),
        "zero_or_one_volume_rows_masked": int(low_volume.sum()),
        "policy": "Keep positive closes and observed session calendar; mask inconsistent OHLC and volume <= 1, retain quality flags. Close/roll quality is not certified.",
    }


def add_targets(frame: pd.DataFrame, horizon: int = 30, unit: str = "sessions") -> pd.DataFrame:
    if horizon < 1 or unit not in {"sessions", "calendar"}:
        raise ValueError("Use a positive horizon and sessions/calendar unit.")
    result = frame.copy()
    dates = pd.DatetimeIndex(result.Date)
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Targets require unique increasing observation dates.")
    if unit == "sessions":
        end = np.arange(len(result)) + horizon
    else:
        end = dates.searchsorted(dates + pd.Timedelta(days=horizon), side="left")
    valid = end < len(result)
    result["target_end_date"] = pd.NaT
    result["target_close"] = np.nan
    result.loc[valid, "target_end_date"] = dates.take(end[valid]).to_numpy()
    result.loc[valid, "target_close"] = result.Close.to_numpy()[end[valid]]
    result["target_return"] = result.target_close / result.Close - 1
    return result


def price_features(prices: pd.DataFrame) -> pd.DataFrame:
    close, volume = prices.Close, prices.Volume
    ret = close.pct_change(fill_method=None)
    logret = np.log(close).diff()
    features = {"Date": prices.Date, "price_invalid_ohlc": prices.price_invalid_ohlc,
                "price_low_volume": prices.price_low_volume}
    for lag in [1, 2, 5, 10, 20, 30, 60, 90, 126, 252]:
        features[f"price_return_{lag}"] = close.pct_change(lag, fill_method=None)
    for lag in [1, 2, 5, 10, 20, 30]:
        features[f"price_daily_return_lag_{lag}"] = ret.shift(lag)
    for window in [5, 10, 20, 30, 60, 90, 126, 252]:
        minimum = max(4, window // 2)
        rolling = close.rolling(window, min_periods=minimum)
        vol = logret.rolling(window, min_periods=minimum).std()
        features[f"price_vol_{window}"] = vol
        features[f"price_vs_ma_{window}"] = close / rolling.mean() - 1
        features[f"price_drawdown_{window}"] = close / rolling.max() - 1
        features[f"price_range_position_{window}"] = (close - rolling.min()) / (rolling.max() - rolling.min()).replace(0, np.nan)
        features[f"price_volume_relative_{window}"] = volume / volume.rolling(window, min_periods=minimum).median().replace(0, np.nan) - 1
        features[f"price_trend_signal_{window}"] = close.pct_change(window, fill_method=None) / (vol * np.sqrt(window)).replace(0, np.nan)
    for window in [14, 30, 60]:
        gain = ret.clip(lower=0).rolling(window, min_periods=window//2).mean()
        loss = (-ret.clip(upper=0)).rolling(window, min_periods=window//2).mean()
        features[f"price_rsi_{window}"] = gain / (gain + loss).replace(0, np.nan)
        features[f"price_downside_{window}"] = np.sqrt(ret.clip(upper=0).pow(2).rolling(window, min_periods=window//2).mean())
        features[f"price_skew_{window}"] = ret.rolling(window, min_periods=window//2).skew()
        features[f"price_efficiency_{window}"] = close.diff(window).abs() / close.diff().abs().rolling(window).sum().replace(0, np.nan)
    features["price_range_pct"] = (prices.High - prices.Low) / close
    features["price_intraday_return"] = close / prices.Open - 1
    features["price_gap_return"] = prices.Open / close.shift(1) - 1
    true_range = pd.concat([prices.High - prices.Low, (prices.High - close.shift(1)).abs(),
                            (prices.Low - close.shift(1)).abs()], axis=1).max(axis=1) / close
    features["price_atr_30"] = true_range.rolling(30, min_periods=15).mean()
    features["price_vol_ratio_10_60"] = features["price_vol_10"] / features["price_vol_60"].replace(0, np.nan)
    features["price_momentum_acceleration"] = features["price_return_30"] - features["price_return_30"].shift(30)
    features["price_volume_return_30"] = (ret * np.log1p(volume)).rolling(30, min_periods=15).mean()
    day = prices.Date.dt.dayofyear
    for harmonic in [1, 2]:
        features[f"price_season_sin_{harmonic}"] = np.sin(2 * np.pi * harmonic * day / 365.25)
        features[f"price_season_cos_{harmonic}"] = np.cos(2 * np.pi * harmonic * day / 365.25)
    return pd.DataFrame(features).replace([np.inf, -np.inf], np.nan)


def build_frame(root: Path, horizon: int = 30, unit: str = "sessions") -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    from cot_release_features import build_cot_features
    from external_return_features import build_external_features
    prices, price_audit = load_prices(root / "../DATA/yahoo/arabica_coffee_futures_history.csv")
    cot, cot_audit, cot_meta = build_cot_features(prices, root)
    external, external_meta = build_external_features(prices, root)
    frame = add_targets(prices, horizon, unit)
    market = price_features(prices)
    frame = frame.drop(columns=["price_invalid_ohlc", "price_low_volume"])
    for part in [market, cot, external]:
        frame = frame.merge(part, on="Date", how="left", validate="one_to_one")
    price_cols = [c for c in market if c != "Date"]
    cot_cols = [c for c in cot if c.startswith("cot_") and pd.api.types.is_numeric_dtype(cot[c])]
    weather_cols = [c for c in external if c.startswith("weather_") and pd.api.types.is_numeric_dtype(external[c])]
    news_cols = [c for c in external if c.startswith("news_") and pd.api.types.is_numeric_dtype(external[c])]
    interactions = {}
    # Fixed, interpretable interactions; no target-based full-sample feature screening.
    for col in [c for c in cot_cols if any(s in c for s in ["net_pct", "net_oi", "index_52", "z_52", "net_change_4"])][:16]:
        interactions[f"interaction_{col}_trend"] = frame[col] * frame.price_return_30
        interactions[f"interaction_{col}_vol"] = frame[col] * frame.price_vol_30
    for col in [c for c in weather_cols if "brazil" in c and any(s in c for s in ["dry", "deficit", "frost", "heat"])][:8]:
        interactions[f"interaction_{col}_season"] = frame[col] * frame.price_season_sin_1
    frame = pd.concat([frame, pd.DataFrame(interactions, index=frame.index)], axis=1)
    legacy_price = ["price_return_1", "price_return_5", "price_return_20", "price_vol_20",
                    "price_vs_ma_60", "price_range_pct", "price_intraday_return", "price_volume_relative_20"]
    legacy_cot = [c for c in cot_cols if "net" in c and not any(s in c for s in ["z_", "index", "change", "lag"])][:4]
    legacy_weather = [c for c in weather_cols if "30" in c][:6]
    groups = {
        "compact_legacy": legacy_price + legacy_cot + legacy_weather,
        "price": price_cols,
        "price_cot": price_cols + cot_cols,
        "price_cot_weather": price_cols + cot_cols + weather_cols,
        "engineered": price_cols + cot_cols + weather_cols + list(interactions),
    }
    feature_cols = groups["engineered"]
    frame[feature_cols] = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    metadata = {"price": price_audit, "cot": cot_meta, "external": external_meta,
                "feature_counts": {k: len(v) for k, v in groups.items()},
                "horizon": horizon, "horizon_unit": unit,
                "signal_time": "After daily close; evaluation forecasts supplied continuous Close returns, not executable trading P&L."}
    return frame, groups, cot_audit, metadata


@dataclass(frozen=True)
class Candidate:
    name: str
    group: str
    kind: str
    strength: float = 1.0
    leaf: int = 40
    train_years: int = 0


def candidates(quick: bool = False) -> list[Candidate]:
    result = [Candidate("zero", "price", "zero"), Candidate("historical_mean", "price", "mean"),
              Candidate("compact_legacy_30d", "compact_legacy", "hist", 10, 40)]
    for group in ["price", "price_cot", "price_cot_weather", "engineered"]:
        for alpha in ([1000] if quick else [100, 1000, 10000]):
            result.append(Candidate(f"{group}_ridge_{alpha}", group, "ridge", alpha))
        result.append(Candidate(f"{group}_hist_regularized", group, "hist", 50, 80))
        result.append(Candidate(f"{group}_extra_trees", group, "extra", 1, 60))
    if not quick:
        result.extend([
            Candidate("engineered_hist_recent10y", "engineered", "hist", 100, 100, 10),
            Candidate("engineered_ridge_recent10y", "engineered", "ridge", 10000, 40, 10),
            Candidate("engineered_elastic", "engineered", "elastic", .003),
            Candidate("engineered_random_forest", "engineered", "forest", 1, 80),
            Candidate("price_cot_elm", "price_cot", "elm", 100),
            Candidate("engineered_elm", "engineered", "elm", 1000),
        ])
    return result


def make_estimator(spec: Candidate, quick: bool = False) -> Pipeline:
    count = 80 if quick else 200
    estimators = {
        "zero": lambda: DummyRegressor(strategy="constant", constant=0),
        "mean": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: Ridge(alpha=spec.strength),
        "elastic": lambda: ElasticNet(alpha=spec.strength, l1_ratio=.1, max_iter=4000),
        # Disable internal random validation splitting on a time series.
        "hist": lambda: HistGradientBoostingRegressor(max_iter=count, learning_rate=.035,
                            max_leaf_nodes=7, min_samples_leaf=spec.leaf,
                            l2_regularization=spec.strength, early_stopping=False, random_state=42),
        "extra": lambda: ExtraTreesRegressor(n_estimators=count, max_depth=7,
                            min_samples_leaf=spec.leaf, max_features=.7, n_jobs=1, random_state=42),
        "forest": lambda: RandomForestRegressor(n_estimators=count, max_depth=6,
                            min_samples_leaf=spec.leaf, max_features=.7, n_jobs=1, random_state=42),
        "elm": lambda: ExtremeLearningMachineRegressor(n_hidden=128, alpha=spec.strength, random_state=42),
    }
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True)),
                     ("scale", StandardScaler()), ("model", estimators[spec.kind]())])


def mature_training_rows(frame: pd.DataFrame, cutoff: pd.Timestamp, train_years: int = 0) -> pd.DataFrame:
    training = frame.loc[frame.target_return.notna() & frame.target_end_date.lt(cutoff) & frame.Date.lt(cutoff)]
    if train_years:
        training = training.loc[training.Date.ge(cutoff - pd.DateOffset(years=train_years))]
    return training


def expanding_folds(frame: pd.DataFrame, holdout_start: pd.Timestamp, splits: int = 4,
                    test_size: int = 504, min_train: int = 1000) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    # Validation labels themselves must have matured before the final holdout.
    development = mature_training_rows(frame, holdout_start).reset_index(drop=True)
    first = len(development) - splits * test_size
    if first < min_train:
        raise ValueError("Not enough development history for the requested folds.")
    result = []
    for start in range(first, len(development), test_size):
        valid = development.iloc[start:start+test_size]
        train = mature_training_rows(frame, valid.Date.iloc[0])
        if len(train) < min_train:
            raise ValueError("Purging leaves too few training observations.")
        result.append((train, valid))
    return result


def fit_candidate(spec: Candidate, train: pd.DataFrame, columns: list[str], quick: bool = False) -> dict:
    # Both availability/variance screening and imputation are fitted inside each fold.
    usable = [c for c in columns if train[c].notna().mean() >= .20 and train[c].nunique(dropna=True) > 1]
    if not usable:
        raise ValueError(f"No train-time usable features for {spec.name}")
    estimator = make_estimator(spec, quick).fit(train[usable], train.target_return)
    return {"spec": asdict(spec), "features": usable, "estimator": estimator}


def predict_member(member: dict, frame: pd.DataFrame) -> np.ndarray:
    return member["estimator"].predict(frame[member["features"]])


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    return sum(member["weight"] * predict_member(member, frame) for member in bundle["members"])


def metrics(actual, predicted) -> dict:
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    mse = np.mean((y - p)**2)
    denom = np.mean(y**2)
    corr = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 1e-12 and np.std(y) > 0 else None
    return {"rows": len(y), "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y-p))),
            "direction_accuracy": float(np.mean(np.sign(y) == np.sign(p))),
            "correlation": corr, "r2_vs_zero": float(1-mse/denom) if denom else None,
            "mean_prediction": float(np.mean(p)), "mean_actual": float(np.mean(y)),
            "direction_note": "Zero forecasts are neutral and count correct only when actual return is zero."}


def block_comparison(y, selected, baseline, block: int = 60, draws: int = 1000) -> dict:
    y, selected, baseline = map(np.asarray, [y, selected, baseline])
    n = len(y)
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(draws):
        starts = rng.integers(0, max(1, n-block+1), size=int(np.ceil(n/block)))
        idx = np.concatenate([np.arange(s, min(s+block, n)) for s in starts])[:n]
        differences.append(np.sqrt(np.mean((y[idx]-selected[idx])**2)) - np.sqrt(np.mean((y[idx]-baseline[idx])**2)))
    return {"rmse_difference": float(np.sqrt(np.mean((y-selected)**2))-np.sqrt(np.mean((y-baseline)**2))),
            "rmse_difference_95pct_block_interval": np.quantile(differences, [.025, .975]).tolist(),
            "block_sessions": block, "bootstrap_draws": draws,
            "interpretation": "Negative favors selected; dependence-aware exploratory interval, not a significance guarantee."}
