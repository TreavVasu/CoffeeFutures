from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests


PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TARGET_RETURN = "target_return_5d"
TARGET_DIRECTION = "target_direction_5d"
TARGET_CLOSE = "target_close_5d"

CORE_WEATHER_DAILY = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

EXTENDED_WEATHER_DAILY = CORE_WEATHER_DAILY + [
    "soil_moisture_0_to_100cm_mean",
    "soil_temperature_0_to_100cm_mean",
    "vapour_pressure_deficit_max",
]

COFFEE_REGIONS = {
    "brazil_minas_gerais": {"latitude": -21.55, "longitude": -45.43, "timezone": "America/Sao_Paulo"},
    "colombia_huila": {"latitude": 2.93, "longitude": -75.28, "timezone": "America/Bogota"},
    "vietnam_dak_lak": {"latitude": 12.67, "longitude": 108.05, "timezone": "Asia/Ho_Chi_Minh"},
}


@dataclass(frozen=True)
class FeatureSet:
    frame: pd.DataFrame
    numeric_features: list[str]
    dropped_features: list[str]


def read_central_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "cot_report_date" in df.columns:
        df["cot_report_date"] = pd.to_datetime(df["cot_report_date"], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


def fetch_open_meteo_weather(
    start_date: str,
    end_date: str,
    output_path: str | Path,
    regions: dict[str, dict[str, float | str]] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return pd.read_csv(output_path, parse_dates=["Date"])

    regions = regions or COFFEE_REGIONS
    weather_parts = []
    fetch_errors = {}
    for region_name, region in regions.items():
        try:
            weather_parts.append(
                _fetch_one_region_weather(
                    region_name=region_name,
                    latitude=float(region["latitude"]),
                    longitude=float(region["longitude"]),
                    timezone=str(region["timezone"]),
                    start_date=start_date,
                    end_date=end_date,
                    variables=EXTENDED_WEATHER_DAILY,
                )
            )
        except requests.RequestException as exc:
            fetch_errors[region_name] = str(exc)
            print(f"Open-Meteo fetch skipped for {region_name}: {exc}")

    if not weather_parts:
        raise RuntimeError(f"Open-Meteo fetch failed for every region: {fetch_errors}")

    weather = weather_parts[0]
    for part in weather_parts[1:]:
        weather = weather.merge(part, on="Date", how="outer")
    weather = weather.sort_values("Date")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(output_path, index=False)
    meta_path = output_path.with_suffix(".metadata.json")
    meta_path.write_text(
        json.dumps(
            {
                "source": "https://archive-api.open-meteo.com/v1/archive",
                "docs": "https://open-meteo.com/en/docs/historical-weather-api",
                "regions": regions,
                "start_date": start_date,
                "end_date": end_date,
                "variables": EXTENDED_WEATHER_DAILY,
                "fetch_errors": fetch_errors,
                "fetched_region_count": len(weather_parts),
            },
            indent=2,
        )
    )
    return weather


def _fetch_one_region_weather(
    region_name: str,
    latitude: float,
    longitude: float,
    timezone: str,
    start_date: str,
    end_date: str,
    variables: list[str],
) -> pd.DataFrame:
    try:
        return _request_open_meteo_region(region_name, latitude, longitude, timezone, start_date, end_date, variables)
    except requests.HTTPError:
        return _request_open_meteo_region(region_name, latitude, longitude, timezone, start_date, end_date, CORE_WEATHER_DAILY)


def _request_open_meteo_region(
    region_name: str,
    latitude: float,
    longitude: float,
    timezone: str,
    start_date: str,
    end_date: str,
    variables: list[str],
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(variables),
        "timezone": timezone,
    }
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()
    daily = pd.DataFrame(payload["daily"])
    daily["Date"] = pd.to_datetime(daily["time"], errors="coerce")
    daily = daily.drop(columns=["time"])
    daily = daily.rename(columns={col: f"weather_{region_name}_{col}" for col in daily.columns if col != "Date"})
    return daily


def build_modeling_frame(
    central_path: str | Path,
    weather_path: str | Path | None = None,
    prediction_mode: bool = False,
) -> FeatureSet:
    raw = read_central_data(central_path)
    price = raw.loc[raw["Close"].notna(), ["Date", *PRICE_COLUMNS]].copy()
    price = price.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    price = _add_price_features(price)
    price[TARGET_CLOSE] = price["Close"].shift(-5)
    price[TARGET_RETURN] = price[TARGET_CLOSE] / price["Close"] - 1.0
    price[TARGET_DIRECTION] = price[TARGET_RETURN] > 0

    cot_history = _build_cot_history(raw)
    if not cot_history.empty:
        price = pd.merge_asof(
            price.sort_values("Date"),
            cot_history.sort_values("cot_release_date"),
            left_on="Date",
            right_on="cot_release_date",
            direction="backward",
        )
        price["days_since_cot_release"] = (price["Date"] - price["cot_release_date"]).dt.days

    if weather_path is not None and Path(weather_path).exists():
        weather = pd.read_csv(weather_path, parse_dates=["Date"])
        price = price.merge(weather, on="Date", how="left")
        price = _add_weather_rollups(price)

    price["month"] = price["Date"].dt.month
    price["quarter"] = price["Date"].dt.quarter
    price["day_of_week"] = price["Date"].dt.dayofweek
    price["year"] = price["Date"].dt.year
    price["month_sin"] = np.sin(2 * np.pi * price["month"] / 12.0)
    price["month_cos"] = np.cos(2 * np.pi * price["month"] / 12.0)

    excluded = {
        TARGET_RETURN,
        TARGET_DIRECTION,
        TARGET_CLOSE,
        "Date",
        "cot_release_date",
        "cot_report_date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "ma_5",
        "ma_10",
        "ma_20",
        "ma_60",
        "volume_ma_5",
        "volume_ma_10",
        "volume_ma_20",
        "volume_ma_60",
        "high_low_spread",
        "open_close_spread",
        "year",
    }
    numeric_candidates = [
        col
        for col in price.columns
        if (
            col not in excluded
            and pd.api.types.is_numeric_dtype(price[col])
            and not _is_useless_feature_name(col)
        )
    ]
    numeric_features = numeric_candidates
    low_signal = [
        col
        for col in numeric_features
        if price[col].notna().sum() < 50 or price[col].nunique(dropna=True) <= 1
    ]
    if low_signal:
        numeric_features = [col for col in numeric_features if col not in low_signal]
    dropped_features = sorted(set(excluded).intersection(price.columns) | set(low_signal))

    if not prediction_mode:
        price = price.loc[price[TARGET_RETURN].notna()].copy()

    return FeatureSet(frame=price.reset_index(drop=True), numeric_features=numeric_features, dropped_features=dropped_features)


def latest_prediction_rows(feature_set: FeatureSet, n_rows: int = 5) -> pd.DataFrame:
    frame = feature_set.frame.sort_values("Date")
    return frame.tail(n_rows).copy()


def _add_price_features(price: pd.DataFrame) -> pd.DataFrame:
    price = price.copy()
    price["return_1d"] = price["Close"].pct_change(1)
    price["return_2d"] = price["Close"].pct_change(2)
    price["return_5d"] = price["Close"].pct_change(5)
    price["return_10d"] = price["Close"].pct_change(10)
    price["return_20d"] = price["Close"].pct_change(20)
    price["range_pct"] = (price["High"] - price["Low"]) / price["Close"]
    price["close_to_open_pct"] = (price["Close"] - price["Open"]) / price["Open"]
    price["volume_change_pct"] = price["Volume"].pct_change(1)
    for window in [5, 10, 20, 60]:
        price[f"ma_{window}"] = price["Close"].rolling(window, min_periods=max(2, window // 2)).mean()
        price[f"close_vs_ma_{window}"] = price["Close"] / price[f"ma_{window}"] - 1.0
        price[f"volatility_{window}"] = price["return_1d"].rolling(window, min_periods=max(2, window // 2)).std()
        price[f"volume_ma_{window}"] = price["Volume"].rolling(window, min_periods=max(2, window // 2)).mean()
        price[f"volume_vs_ma_{window}"] = price["Volume"] / price[f"volume_ma_{window}"] - 1.0
    price["high_low_spread"] = price["High"] - price["Low"]
    price["open_close_spread"] = price["Close"] - price["Open"]
    return price


def _build_cot_history(raw: pd.DataFrame) -> pd.DataFrame:
    if "cot_report_date" not in raw.columns or raw["cot_report_date"].notna().sum() == 0:
        return pd.DataFrame()

    start_column = raw.columns.get_loc("Open_Interest_All") if "Open_Interest_All" in raw.columns else 0
    candidate_columns = list(raw.columns[start_column:])
    numeric_cot_columns = [
        col for col in candidate_columns if pd.api.types.is_numeric_dtype(raw[col]) and raw[col].notna().sum() > 0
    ]
    numeric_cot_columns = [col for col in numeric_cot_columns if not _is_useless_feature_name(col)]
    keep_columns = ["cot_report_date", *numeric_cot_columns]
    cot = raw.loc[raw["cot_report_date"].notna(), keep_columns].copy()
    cot = cot.sort_values("cot_report_date").drop_duplicates("cot_report_date", keep="last")
    cot["cot_release_date"] = cot["cot_report_date"] + pd.Timedelta(days=3)
    return cot


def _is_useless_feature_name(column: str) -> bool:
    lowered = column.lower()
    useless_tokens = [
        "source",
        "archive",
        "file",
        "market_and_exchange_names",
        "contract_units",
        "cftc_contract_market_code",
        "cftc_market_code",
        "cftc_region_code",
        "cftc_commodity_code",
        "cftc_subgroup_code",
        "as_of_date",
        "report_date",
        "futonly_or_combined",
    ]
    return any(token in lowered for token in useless_tokens)


def _is_model_relevant_feature(column: str) -> bool:
    if not _looks_like_cot_feature(column):
        return True
    allowed_exact = {
        "Open_Interest_All",
        "Change_in_Open_Interest_All",
        "open_interest_change_pct",
        "managed_money_net",
        "managed_money_net_pct_oi",
        "producer_merchant_net",
        "commercial_net",
        "noncommercial_net",
        "nonreportable_net",
        "swap_dealer_net",
        "other_reportable_net",
        "managed_money_weekly_net_change",
        "commercial_weekly_net_change",
        "noncommercial_weekly_net_change",
        "Conc_Gross_LE_4_TDR_Long_All",
        "Conc_Gross_LE_4_TDR_Short_All",
        "Conc_Gross_LE_8_TDR_Long_All",
        "Conc_Gross_LE_8_TDR_Short_All",
        "Conc_Net_LE_4_TDR_Long_All",
        "Conc_Net_LE_4_TDR_Short_All",
        "Conc_Net_LE_8_TDR_Long_All",
        "Conc_Net_LE_8_TDR_Short_All",
    }
    allowed_prefixes = (
        "M_Money_Positions_Long_ALL",
        "M_Money_Positions_Short_ALL",
        "Prod_Merc_Positions_Long_ALL",
        "Prod_Merc_Positions_Short_ALL",
        "Swap_Positions_Long_All",
        "Swap__Positions_Short_All",
        "NonRept_Positions_Long_All",
        "NonRept_Positions_Short_All",
        "NonComm_Positions_Long_All",
        "NonComm_Positions_Short_All",
        "Comm_Positions_Long_All",
        "Comm_Positions_Short_All",
        "Change_in_M_Money_Long_All",
        "Change_in_M_Money_Short_All",
        "Change_in_Prod_Merc_Long_All",
        "Change_in_Prod_Merc_Short_All",
        "Change_in_Swap_Long_All",
        "Change_in_Swap_Short_All",
        "Change_in_NonRept_Long_All",
        "Change_in_NonRept_Short_All",
        "Change_in_NonComm_Long_All",
        "Change_in_NonComm_Short_All",
        "Change_in_Comm_Long_All",
        "Change_in_Comm_Short_All",
        "Pct_of_OI_M_Money_Long_All",
        "Pct_of_OI_M_Money_Short_All",
        "Pct_of_OI_Prod_Merc_Long_All",
        "Pct_of_OI_Prod_Merc_Short_All",
        "Pct_of_OI_Swap_Long_All",
        "Pct_of_OI_Swap_Short_All",
        "Pct_of_OI_NonRept_Long_All",
        "Pct_of_OI_NonRept_Short_All",
        "Pct_of_OI_NonComm_Long_All",
        "Pct_of_OI_NonComm_Short_All",
        "Pct_of_OI_Comm_Long_All",
        "Pct_of_OI_Comm_Short_All",
    )
    return column in allowed_exact or column.startswith(allowed_prefixes)


def _looks_like_cot_feature(column: str) -> bool:
    tokens = [
        "Open_Interest",
        "Positions",
        "Pct_of_OI",
        "Traders",
        "Conc_",
        "managed_money",
        "producer_merchant",
        "commercial",
        "noncommercial",
        "nonreportable",
        "swap_dealer",
        "other_reportable",
        "Change_in_",
    ]
    return any(token in column for token in tokens)


def _add_weather_rollups(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("Date").copy()
    weather_columns = [col for col in frame.columns if col.startswith("weather_")]
    for column in weather_columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        for window in [5, 20]:
            frame[f"{column}_roll{window}"] = frame[column].rolling(window, min_periods=max(2, window // 2)).mean()
        frame[f"{column}_diff5"] = frame[column] - frame[column].shift(5)
    return frame


def chronological_split(frame: pd.DataFrame, train_size: float = 0.70, valid_size: float = 0.15):
    frame = frame.sort_values("Date").reset_index(drop=True)
    n_rows = len(frame)
    train_end = int(n_rows * train_size)
    valid_end = int(n_rows * (train_size + valid_size))
    return frame.iloc[:train_end].copy(), frame.iloc[train_end:valid_end].copy(), frame.iloc[valid_end:].copy()


def safe_json_dump(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_json_ready(payload), indent=2))


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def feature_frame_for_model(frame: pd.DataFrame, numeric_features: Iterable[str], text_column: str | None = None) -> pd.DataFrame:
    columns = list(numeric_features)
    return frame.loc[:, columns].copy()
