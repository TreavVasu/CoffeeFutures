from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "centralData" / "yahoo_cot_full_outer_by_date_cot_ffill.csv"
WEATHER_PATH = ROOT / "data" / "weather" / "open_meteo_coffee_regions_daily.csv"
OUTPUT_DIR = ROOT / "Scripts" / "project" / "artifacts" / "outputs"
PLOT_DIR = ROOT / "Scripts" / "project" / "artifacts" / "plots"
MODEL_DIR = ROOT / "Scripts" / "project" / "artifacts" / "models"

REGIONS = {
    "brazil_minas_gerais": "Brazil Minas Gerais",
    "colombia_huila": "Colombia Huila",
    "vietnam_dak_lak": "Vietnam Dak Lak",
}

BASE_WEATHER_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "soil_moisture_0_to_100cm_mean",
    "vapour_pressure_deficit_max",
]

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    data = load_model_data()
    top3_cot_features = select_top3_cot_features(data)
    weather_features, region_feature_map, region_drift = add_weather_features(data)

    yahoo_features = [
        "return_1d",
        "return_5d",
        "return_20d",
        "realized_vol_20d",
        "close_vs_ma_63",
        "range_pct",
        "close_to_open_pct",
        "volume_change_pct",
    ]
    model_features = yahoo_features + top3_cot_features + weather_features

    model_df = data.loc[data["target_return_5d"].notna()].replace([np.inf, -np.inf], np.nan).copy()
    train, valid, test = chronological_split(model_df)
    X_train, X_valid, X_test = train[model_features], valid[model_features], test[model_features]
    y_train = train["target_return_5d"].astype(float)
    y_valid = valid["target_return_5d"].astype(float)
    y_test = test["target_return_5d"].astype(float)

    candidates = build_candidates(model_features)
    validation_df = evaluate_candidates(candidates, X_train, y_train, X_valid, y_valid, model_group="base")
    ensemble_candidates = build_ensemble_candidates(candidates, validation_df)
    ensemble_validation_df = evaluate_candidates(
        ensemble_candidates,
        X_train,
        y_train,
        X_valid,
        y_valid,
        model_group="ensemble",
    )
    validation_df = (
        pd.concat([validation_df, ensemble_validation_df], ignore_index=True)
        .sort_values(["valid_rmse", "valid_mae", "valid_directional_accuracy"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    all_candidates = {**candidates, **ensemble_candidates}
    best_valid_rmse = float(validation_df.iloc[0]["valid_rmse"])
    selection_pool = validation_df.loc[validation_df["valid_rmse"].le(best_valid_rmse * 1.005)].copy()
    selected_row = selection_pool.sort_values(
        ["valid_directional_accuracy", "valid_rmse", "valid_mae"],
        ascending=[False, True, True],
    ).iloc[0]
    best_model_name = str(selected_row["model"])
    best_model = all_candidates[best_model_name]

    train_valid = pd.concat([train, valid], ignore_index=True)
    best_model.fit(train_valid[model_features], train_valid["target_return_5d"].astype(float))
    test_pred = best_model.predict(X_test)

    holdout_df = evaluate_holdout_models(
        all_candidates,
        validation_df["model"].tolist(),
        train_valid[model_features],
        train_valid["target_return_5d"].astype(float),
        X_test,
        y_test,
        max_models=12,
    )

    predictions = test[["Date", "Close", "target_close_5d", "target_return_5d", "target_direction_5d"]].copy()
    predictions["predicted_return_5d"] = test_pred
    predictions["predicted_close_5d"] = predictions["Close"] * (1 + predictions["predicted_return_5d"])
    predictions["predicted_direction_5d"] = predictions["predicted_return_5d"] > 0
    predictions["return_error"] = predictions["target_return_5d"] - predictions["predicted_return_5d"]
    predictions["price_error"] = predictions["target_close_5d"] - predictions["predicted_close_5d"]

    permutation = grouped_permutation_impact(
        best_model,
        X_test,
        y_test,
        region_feature_map,
        random_state=42,
    )
    feature_importance = extract_feature_importance(best_model, model_features)
    region_importance = aggregate_region_importance(feature_importance, region_feature_map)

    metrics = {
        "target": "5-trading-day future Arabica Coffee C return",
        "best_model": best_model_name,
        "features": model_features,
        "feature_counts": {
            "yahoo": len(yahoo_features),
            "top3_cot": len(top3_cot_features),
            "weather": len(weather_features),
            "total": len(model_features),
        },
        "regions": REGIONS,
        "top3_cot_features": top3_cot_features,
        "selection_policy": {
            "rule": "Choose the highest validation directional accuracy among models within 0.5% of the best validation RMSE.",
            "best_validation_rmse": best_valid_rmse,
            "selected_model_validation_rmse": float(selected_row["valid_rmse"]),
            "selected_model_validation_directional_accuracy": float(selected_row["valid_directional_accuracy"]),
        },
        "validation_results": validation_df.to_dict(orient="records"),
        "holdout_model_comparison": holdout_df.to_dict(orient="records"),
        "holdout": {
            "mae": float(mean_absolute_error(y_test, test_pred)),
            "rmse": float(root_mean_squared_error(y_test, test_pred)),
            "r2": float(r2_score(y_test, test_pred)),
            "directional_accuracy": float(accuracy_score((y_test > 0).astype(int), (test_pred > 0).astype(int))),
            "date_start": test["Date"].min().date().isoformat(),
            "date_end": test["Date"].max().date().isoformat(),
            "rows": int(len(test)),
        },
        "region_permutation_impact": permutation.to_dict(orient="records"),
        "region_feature_importance": region_importance.to_dict(orient="records"),
    }

    metrics_path = OUTPUT_DIR / "yahoo_cot_weather_return_model_metrics.json"
    predictions_path = OUTPUT_DIR / "yahoo_cot_weather_return_model_predictions.csv"
    comparison_path = OUTPUT_DIR / "yahoo_cot_weather_model_comparison.csv"
    region_drift_path = OUTPUT_DIR / "weather_regional_drift_scores.csv"
    region_impact_path = OUTPUT_DIR / "weather_region_impact.csv"
    model_path = MODEL_DIR / "yahoo_cot_weather_return_model.joblib"

    metrics_path.write_text(json.dumps(json_ready(metrics), indent=2))
    predictions.to_csv(predictions_path, index=False)
    validation_df.merge(holdout_df, on=["model", "model_group"], how="left").to_csv(comparison_path, index=False)
    region_drift.to_csv(region_drift_path, index=False)
    permutation.merge(region_importance, on=["region", "region_label"], how="outer").to_csv(region_impact_path, index=False)
    joblib.dump(
        {
            "model": best_model,
            "features": model_features,
            "yahoo_features": yahoo_features,
            "top3_cot_features": top3_cot_features,
            "weather_features": weather_features,
            "region_feature_map": region_feature_map,
            "metrics": metrics,
        },
        model_path,
    )

    write_prediction_plots(predictions)
    write_model_comparison_plot(validation_df, holdout_df)
    write_region_drift_plot(region_drift, predictions)
    write_region_impact_plot(permutation, region_importance)

    print("Selected model:", best_model_name)
    print("Holdout:", metrics["holdout"])
    print("Top model comparison:")
    print(validation_df.merge(holdout_df, on=["model", "model_group"], how="left").head(12).to_string(index=False))
    print("Region permutation impact:")
    print(permutation.to_string(index=False))
    print("Saved:", metrics_path)
    print("Saved:", predictions_path)
    print("Saved:", comparison_path)
    print("Saved:", region_drift_path)
    print("Saved:", region_impact_path)
    print("Saved:", model_path)


def load_model_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, low_memory=False, parse_dates=["Date"])
    data = data.loc[data["Close"].notna()].sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)

    data["return_1d"] = data["Close"].pct_change(1)
    data["return_5d"] = data["Close"].pct_change(5)
    data["return_20d"] = data["Close"].pct_change(20)
    data["realized_vol_20d"] = data["return_1d"].rolling(20, min_periods=10).std() * np.sqrt(252)
    data["ma_63"] = data["Close"].rolling(63, min_periods=30).mean()
    data["close_vs_ma_63"] = data["Close"] / data["ma_63"] - 1.0
    data["range_pct"] = (data["High"] - data["Low"]) / data["Close"]
    data["close_to_open_pct"] = (data["Close"] - data["Open"]) / data["Open"]
    data["volume_change_pct"] = data["Volume"].pct_change(1)
    data["target_return_5d"] = data["Close"].shift(-5) / data["Close"] - 1.0
    data["target_close_5d"] = data["Close"].shift(-5)
    data["target_direction_5d"] = data["target_return_5d"] > 0

    weather = pd.read_csv(WEATHER_PATH, parse_dates=["Date"])
    data = data.merge(weather, on="Date", how="left")
    return data


def select_top3_cot_features(data: pd.DataFrame) -> list[str]:
    cot_candidates = [
        "managed_money_weekly_net_change",
        "commercial_weekly_net_change",
        "noncommercial_weekly_net_change",
        "managed_money_net",
        "commercial_net",
        "noncommercial_net",
        "open_interest_change_pct",
        "Open_Interest_All",
    ]
    available = [col for col in cot_candidates if col in data.columns]
    corr_rows = []
    target = data["target_return_5d"]
    for col in available:
        corr_rows.append({"feature": col, "abs_corr": abs(pd.to_numeric(data[col], errors="coerce").corr(target))})
    corr = pd.DataFrame(corr_rows).sort_values("abs_corr", ascending=False)
    return corr["feature"].head(3).tolist()


def add_weather_features(data: pd.DataFrame) -> tuple[list[str], dict[str, list[str]], pd.DataFrame]:
    weather_features: list[str] = []
    region_feature_map: dict[str, list[str]] = {}
    drift_frames = []

    for region, region_label in REGIONS.items():
        region_features = []
        drift_inputs = []
        for variable in BASE_WEATHER_VARIABLES:
            base_col = f"weather_{region}_{variable}"
            if base_col not in data.columns:
                continue
            base = pd.to_numeric(data[base_col], errors="coerce")
            roll7 = f"{base_col}_roll7"
            roll30 = f"{base_col}_roll30"
            anomaly = f"{base_col}_roll30_vs_252"
            data[roll7] = base.rolling(7, min_periods=3).mean()
            data[roll30] = base.rolling(30, min_periods=10).mean()
            data[anomaly] = data[roll30] - data[roll30].rolling(252, min_periods=80).mean()
            region_features.extend([roll7, roll30, anomaly])
            if variable in {
                "temperature_2m_mean",
                "precipitation_sum",
                "soil_moisture_0_to_100cm_mean",
                "vapour_pressure_deficit_max",
            }:
                z_col = f"{base_col}_regional_drift_z"
                data[z_col] = rolling_zscore(data[roll30])
                drift_inputs.append(z_col)

        if region_features:
            stress_col = f"weather_{region}_stress_index"
            vpd_col = f"weather_{region}_vapour_pressure_deficit_max_roll30_vs_252"
            rain_col = f"weather_{region}_precipitation_sum_roll30_vs_252"
            soil_col = f"weather_{region}_soil_moisture_0_to_100cm_mean_roll30_vs_252"
            data[stress_col] = (
                pd.to_numeric(data.get(vpd_col), errors="coerce").fillna(0)
                - pd.to_numeric(data.get(rain_col), errors="coerce").fillna(0)
                - pd.to_numeric(data.get(soil_col), errors="coerce").fillna(0)
            )
            region_features.append(stress_col)

            drift_score = data[drift_inputs].abs().max(axis=1) if drift_inputs else pd.Series(np.nan, index=data.index)
            drift_frames.append(
                pd.DataFrame(
                    {
                        "Date": data["Date"],
                        "region": region,
                        "region_label": region_label,
                        "weather_drift_score": drift_score,
                    }
                )
            )

        region_feature_map[region] = region_features
        weather_features.extend(region_features)

    region_drift = pd.concat(drift_frames, ignore_index=True)
    return weather_features, region_feature_map, region_drift


def rolling_zscore(series: pd.Series, window: int = 252, min_periods: int = 80) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def chronological_split(frame: pd.DataFrame, train_size: float = 0.70, valid_size: float = 0.15):
    frame = frame.sort_values("Date").reset_index(drop=True)
    n_rows = len(frame)
    train_end = int(n_rows * train_size)
    valid_end = int(n_rows * (train_size + valid_size))
    return frame.iloc[:train_end].copy(), frame.iloc[train_end:valid_end].copy(), frame.iloc[valid_end:].copy()


def build_candidates(model_features: list[str]) -> dict[str, Pipeline]:
    def make_pipeline(model):
        return Pipeline(
            [
                (
                    "preprocess",
                    ColumnTransformer(
                        [
                            (
                                "num",
                                Pipeline(
                                    [
                                        ("impute", SimpleImputer(strategy="median")),
                                        ("scale", StandardScaler()),
                                    ]
                                ),
                                model_features,
                            )
                        ]
                    ),
                ),
                ("model", model),
            ]
        )

    candidates = {
        "ridge_alpha_1": make_pipeline(Ridge(alpha=1.0)),
        "ridge_alpha_3": make_pipeline(Ridge(alpha=3.0)),
        "bayesian_ridge": make_pipeline(BayesianRidge()),
        "elastic_net": make_pipeline(ElasticNet(alpha=0.0005, l1_ratio=0.15, max_iter=10000, random_state=42)),
        "huber": make_pipeline(HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=500)),
        "knn_25_distance": make_pipeline(KNeighborsRegressor(n_neighbors=25, weights="distance")),
        "decision_tree_shallow": make_pipeline(
            DecisionTreeRegressor(max_depth=5, min_samples_leaf=35, random_state=42)
        ),
        "bagged_trees": make_pipeline(
            BaggingRegressor(
                estimator=DecisionTreeRegressor(max_depth=5, min_samples_leaf=25, random_state=42),
                n_estimators=45,
                random_state=42,
                n_jobs=1,
            )
        ),
        "ada_boost_tree": make_pipeline(
            AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=4, min_samples_leaf=30, random_state=42),
                n_estimators=70,
                learning_rate=0.025,
                random_state=42,
            )
        ),
        "gradient_boosting": make_pipeline(
            GradientBoostingRegressor(
                n_estimators=140,
                learning_rate=0.025,
                max_depth=2,
                min_samples_leaf=25,
                subsample=0.75,
                random_state=42,
            )
        ),
        "extra_trees_depth_6": make_pipeline(
            ExtraTreesRegressor(n_estimators=180, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=1)
        ),
        "extra_trees_depth_8": make_pipeline(
            ExtraTreesRegressor(n_estimators=220, max_depth=8, min_samples_leaf=18, random_state=42, n_jobs=1)
        ),
        "random_forest_depth_6": make_pipeline(
            RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=22, random_state=42, n_jobs=1)
        ),
        "random_forest_depth_8": make_pipeline(
            RandomForestRegressor(n_estimators=180, max_depth=8, min_samples_leaf=18, random_state=42, n_jobs=1)
        ),
        "hist_gradient_boosting": make_pipeline(
            HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_iter=240,
                max_leaf_nodes=15,
                min_samples_leaf=28,
                l2_regularization=0.08,
                early_stopping=True,
                random_state=42,
            )
        ),
    }
    if XGBRegressor is not None:
        candidates["xgboost_conservative"] = make_pipeline(
            XGBRegressor(
                n_estimators=120,
                max_depth=2,
                learning_rate=0.025,
                subsample=0.75,
                colsample_bytree=0.75,
                reg_lambda=8.0,
                reg_alpha=0.2,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            )
        )
    return candidates


def evaluate_candidates(
    candidates: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    model_group: str,
) -> pd.DataFrame:
    rows = []
    for name, estimator in candidates.items():
        print(f"Fitting {model_group} model: {name}", flush=True)
        model = clone(estimator)
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        rows.append(
            {
                "model": name,
                "model_group": model_group,
                "valid_mae": float(mean_absolute_error(y_valid, pred)),
                "valid_rmse": float(root_mean_squared_error(y_valid, pred)),
                "valid_r2": float(r2_score(y_valid, pred)),
                "valid_directional_accuracy": float(
                    accuracy_score((y_valid > 0).astype(int), (pred > 0).astype(int))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["valid_rmse", "valid_mae"]).reset_index(drop=True)


def build_ensemble_candidates(
    base_candidates: dict[str, Pipeline],
    validation_df: pd.DataFrame,
) -> dict[str, object]:
    ranked = validation_df.sort_values(["valid_rmse", "valid_mae"]).reset_index(drop=True)
    top3 = ranked.head(3)

    def estimators_from(frame: pd.DataFrame):
        return [(row["model"], clone(base_candidates[row["model"]])) for _, row in frame.iterrows()]

    top3_estimators = estimators_from(top3)
    top3_weights = (1.0 / top3["valid_rmse"].clip(lower=1e-9)).to_numpy()

    ensembles: dict[str, object] = {
        "ensemble_top3_mean": VotingRegressor(estimators=top3_estimators, n_jobs=1),
        "ensemble_top3_weighted_rmse": VotingRegressor(estimators=top3_estimators, weights=top3_weights, n_jobs=1),
    }
    return ensembles


def evaluate_holdout_models(
    candidates: dict[str, object],
    ranked_model_names: list[str],
    X_train_valid: pd.DataFrame,
    y_train_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    max_models: int = 12,
) -> pd.DataFrame:
    rows = []
    selected_names = ranked_model_names[:max_models]
    for name in selected_names:
        print(f"Evaluating holdout model: {name}", flush=True)
        model = clone(candidates[name])
        model.fit(X_train_valid, y_train_valid)
        pred = model.predict(X_test)
        rows.append(
            {
                "model": name,
                "model_group": "ensemble" if name.startswith(("ensemble_", "stacking_")) else "base",
                "holdout_mae": float(mean_absolute_error(y_test, pred)),
                "holdout_rmse": float(root_mean_squared_error(y_test, pred)),
                "holdout_r2": float(r2_score(y_test, pred)),
                "holdout_directional_accuracy": float(
                    accuracy_score((y_test > 0).astype(int), (pred > 0).astype(int))
                ),
            }
        )
    return pd.DataFrame(rows)


def grouped_permutation_impact(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    region_feature_map: dict[str, list[str]],
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    baseline = root_mean_squared_error(y_test, model.predict(X_test))
    rows = []
    for region, features in region_feature_map.items():
        usable_features = [feature for feature in features if feature in X_test.columns]
        if not usable_features:
            continue
        permuted = X_test.copy()
        for feature in usable_features:
            values = permuted[feature].to_numpy(copy=True)
            rng.shuffle(values)
            permuted[feature] = values
        permuted_rmse = root_mean_squared_error(y_test, model.predict(permuted))
        rows.append(
            {
                "region": region,
                "region_label": REGIONS[region],
                "baseline_rmse": baseline,
                "permuted_rmse": permuted_rmse,
                "rmse_increase": permuted_rmse - baseline,
                "feature_count": len(usable_features),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_increase", ascending=False).reset_index(drop=True)


def extract_feature_importance(model: object, model_features: list[str]) -> pd.DataFrame:
    if not isinstance(model, Pipeline):
        return pd.DataFrame({"feature": model_features, "importance": np.zeros(len(model_features), dtype=float)})
    fitted = model.named_steps["model"]
    if hasattr(fitted, "feature_importances_"):
        values = np.asarray(fitted.feature_importances_, dtype=float)
    elif hasattr(fitted, "coef_"):
        values = np.abs(np.asarray(fitted.coef_, dtype=float)).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    values = values / values.sum() if values.sum() else values
    return pd.DataFrame({"feature": model_features, "importance": values}).sort_values("importance", ascending=False)


def aggregate_region_importance(
    feature_importance: pd.DataFrame,
    region_feature_map: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []
    for region, features in region_feature_map.items():
        mask = feature_importance["feature"].isin(features)
        rows.append(
            {
                "region": region,
                "region_label": REGIONS[region],
                "tree_importance_sum": float(feature_importance.loc[mask, "importance"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("tree_importance_sum", ascending=False).reset_index(drop=True)


def write_prediction_plots(predictions: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(predictions["Date"], predictions["target_return_5d"], label="Actual 5D return", linewidth=1.5)
    plt.plot(predictions["Date"], predictions["predicted_return_5d"], label="Predicted 5D return with weather", linewidth=1.5)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Yahoo + Top-3 COT + Weather Model: Actual vs Predicted 5D Returns")
    plt.ylabel("5D return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "yahoo_cot_weather_actual_vs_predicted_returns.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(predictions["Date"], predictions["target_close_5d"], label="Actual future close", linewidth=1.5)
    plt.plot(predictions["Date"], predictions["predicted_close_5d"], label="Predicted future close with weather", linewidth=1.5)
    plt.title("Yahoo + Top-3 COT + Weather Model: Actual vs Predicted Price")
    plt.ylabel("Coffee C close")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "yahoo_cot_weather_real_vs_predicted_price.png", dpi=180)
    plt.close()


def write_model_comparison_plot(validation_df: pd.DataFrame, holdout_df: pd.DataFrame) -> None:
    comparison = validation_df.merge(holdout_df, on=["model", "model_group"], how="left").head(12).copy()
    comparison = comparison.sort_values("valid_rmse", ascending=True)
    x = np.arange(len(comparison))

    fig, ax1 = plt.subplots(figsize=(15, 7))
    colors = np.where(comparison["model_group"].eq("ensemble"), "#2a9d8f", "#457b9d")
    ax1.bar(x - 0.18, comparison["valid_rmse"], width=0.36, color=colors, alpha=0.82, label="Validation RMSE")
    ax1.bar(
        x + 0.18,
        comparison["holdout_rmse"],
        width=0.36,
        color="#f4a261",
        alpha=0.72,
        label="Holdout RMSE",
    )
    ax1.set_ylabel("RMSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison["model"], rotation=35, ha="right")
    ax1.set_title("Weather Model Comparison: Base Models Vs Ensembles")
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        comparison["valid_directional_accuracy"],
        color="#1f2937",
        marker="o",
        linewidth=1.5,
        label="Validation direction accuracy",
    )
    ax2.plot(
        x,
        comparison["holdout_directional_accuracy"],
        color="#d1495b",
        marker="s",
        linewidth=1.5,
        label="Holdout direction accuracy",
    )
    ax2.set_ylabel("Directional accuracy")
    ax2.set_ylim(0.40, max(0.62, comparison["holdout_directional_accuracy"].max() + 0.03))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "yahoo_cot_weather_model_comparison.png", dpi=180)
    plt.close(fig)


def write_region_drift_plot(region_drift: pd.DataFrame, predictions: pd.DataFrame) -> None:
    drift = region_drift.copy()
    drift["Date"] = pd.to_datetime(drift["Date"])
    test_start = predictions["Date"].min()
    drift = drift.loc[drift["Date"].ge(test_start)].copy()
    top_error_dates = set(predictions.assign(abs_error=predictions["return_error"].abs()).nlargest(8, "abs_error")["Date"])

    plt.figure(figsize=(15, 7))
    for region_label, group in drift.groupby("region_label"):
        plt.plot(group["Date"], group["weather_drift_score"], linewidth=1.6, label=region_label)
    for date in sorted(top_error_dates):
        plt.axvline(date, color="#666666", alpha=0.18, linewidth=1)
    plt.axhline(2.0, color="#d1495b", linestyle="--", linewidth=1.1, label="High drift threshold")
    plt.title("Regional Weather Drift Scores During Weather-Model Holdout")
    plt.xlabel("Date")
    plt.ylabel("Max absolute regional weather z-score")
    plt.legend(loc="upper left")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "weather_regional_drift_scores.png", dpi=180)
    plt.close()


def write_region_impact_plot(permutation: pd.DataFrame, region_importance: pd.DataFrame) -> None:
    impact = permutation.merge(region_importance, on=["region", "region_label"], how="outer").fillna(0)
    impact = impact.sort_values("rmse_increase", ascending=False)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(impact))
    ax1.bar(x - 0.18, impact["rmse_increase"], width=0.36, color="#457b9d", label="RMSE increase when permuted")
    ax1.set_ylabel("Holdout RMSE increase")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, impact["tree_importance_sum"], width=0.36, color="#e9c46a", label="Model feature importance")
    ax2.set_ylabel("Summed weather feature importance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(impact["region_label"], rotation=20, ha="right")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("Regional Weather Impact On 5D Arabica Return Prediction")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "weather_region_impact.png", dpi=180)
    plt.close(fig)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
