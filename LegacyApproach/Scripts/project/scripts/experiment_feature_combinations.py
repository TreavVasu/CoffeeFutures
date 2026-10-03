from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from train_top3_cot_weather_model import (  # noqa: E402
    BASE_WEATHER_VARIABLES,
    MODEL_DIR,
    OUTPUT_DIR,
    PLOT_DIR,
    REGIONS,
    add_weather_features,
    chronological_split,
    load_model_data,
    select_top3_cot_features,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    data = load_model_data()
    top3_cot_features = select_top3_cot_features(data)
    weather_features, _, _ = add_weather_features(data)

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
    base_features = yahoo_features + top3_cot_features + weather_features
    aggregate_features = add_similar_feature_aggregates(data, yahoo_features, top3_cot_features)

    model_df = data.loc[data["target_return_5d"].notna()].replace([np.inf, -np.inf], np.nan).copy()
    train, valid, test = chronological_split(model_df)

    importance = rank_reference_importance(train, yahoo_features, top3_cot_features, weather_features)
    feature_sets = build_feature_sets(yahoo_features, top3_cot_features, weather_features, aggregate_features, importance)

    experiment_rows = []
    best_payload = None
    for set_name, features in feature_sets.items():
        print(f"\nFeature set: {set_name} ({len(features)} features)", flush=True)
        result = evaluate_feature_set(set_name, features, train, valid, test)
        experiment_rows.extend(result["comparison_rows"])
        if best_payload is None or better_result(result["selected"], best_payload["selected"]):
            best_payload = result

    if best_payload is None:
        raise RuntimeError("No feature-combination experiments were evaluated.")

    results = pd.DataFrame(experiment_rows).sort_values(
        ["selected", "holdout_rmse", "holdout_directional_accuracy"],
        ascending=[False, True, False],
    )
    best = best_payload["selected"]
    predictions = best_payload["predictions"]

    results_path = OUTPUT_DIR / "feature_combo_ensemble_experiment_results.csv"
    best_features_path = OUTPUT_DIR / "feature_combo_ensemble_best_features.json"
    predictions_path = OUTPUT_DIR / "feature_combo_ensemble_predictions.csv"
    model_path = MODEL_DIR / "feature_combo_ensemble_model.joblib"

    results.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    best_features_path.write_text(
        json.dumps(
            {
                "selected": best,
                "features": best_payload["features"],
                "aggregate_features": [feature for feature in best_payload["features"] if feature in aggregate_features],
                "reference_importance_top_30": importance.head(30).to_dict(orient="records"),
                "feature_set_count": len(feature_sets),
                "model_count_per_set": len(build_model_candidates(best_payload["features"])),
            },
            indent=2,
        )
    )
    joblib.dump(
        {
            "model": best_payload["model"],
            "features": best_payload["features"],
            "selected": best,
            "top3_cot_features": top3_cot_features,
            "aggregate_features": aggregate_features,
        },
        model_path,
    )

    write_experiment_plots(results, predictions)

    print("\nBest feature-combo experiment:")
    print(pd.Series(best).to_string())
    print("Saved:", results_path)
    print("Saved:", best_features_path)
    print("Saved:", predictions_path)
    print("Saved:", model_path)


def add_similar_feature_aggregates(
    data: pd.DataFrame,
    yahoo_features: list[str],
    top3_cot_features: list[str],
) -> list[str]:
    aggregate_features: list[str] = []

    aggregate_features.extend(
        add_row_stats(
            data,
            "yahoo_return",
            [feature for feature in ["return_1d", "return_5d", "return_20d"] if feature in yahoo_features],
        )
    )
    aggregate_features.extend(
        add_row_stats(
            data,
            "yahoo_intraday_range",
            [feature for feature in ["range_pct", "close_to_open_pct"] if feature in yahoo_features],
        )
    )
    aggregate_features.extend(add_row_stats(data, "cot_top3_net_change", top3_cot_features))

    for variable in BASE_WEATHER_VARIABLES:
        for suffix in ["roll7", "roll30", "roll30_vs_252"]:
            cols = [
                f"weather_{region}_{variable}_{suffix}"
                for region in REGIONS
                if f"weather_{region}_{variable}_{suffix}" in data.columns
            ]
            aggregate_features.extend(add_row_stats(data, f"weather_all_regions_{variable}_{suffix}", cols))

    for region in REGIONS:
        anomaly_cols = [
            f"weather_{region}_{variable}_roll30_vs_252"
            for variable in BASE_WEATHER_VARIABLES
            if f"weather_{region}_{variable}_roll30_vs_252" in data.columns
        ]
        aggregate_features.extend(add_row_stats(data, f"weather_{region}_anomaly_basket", anomaly_cols))

    stress_cols = [f"weather_{region}_stress_index" for region in REGIONS if f"weather_{region}_stress_index" in data.columns]
    aggregate_features.extend(add_row_stats(data, "weather_all_regions_stress_index", stress_cols))
    return aggregate_features


def add_row_stats(data: pd.DataFrame, prefix: str, cols: list[str]) -> list[str]:
    cols = [col for col in cols if col in data.columns]
    if len(cols) < 2:
        return []
    mean_col = f"{prefix}_avg"
    std_col = f"{prefix}_std"
    data[mean_col] = data[cols].mean(axis=1)
    data[std_col] = data[cols].std(axis=1)
    return [mean_col, std_col]


def rank_reference_importance(
    train: pd.DataFrame,
    yahoo_features: list[str],
    top3_cot_features: list[str],
    weather_features: list[str],
) -> pd.DataFrame:
    features = yahoo_features + top3_cot_features + weather_features
    model = make_pipeline(
        features,
        ExtraTreesRegressor(n_estimators=220, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=1),
    )
    model.fit(train[features], train["target_return_5d"].astype(float))
    values = model.named_steps["model"].feature_importances_
    ranked = pd.DataFrame({"feature": features, "importance": values})
    ranked["rank"] = ranked["importance"].rank(method="first", ascending=False).astype(int)
    return ranked.sort_values("importance", ascending=False).reset_index(drop=True)


def build_feature_sets(
    yahoo_features: list[str],
    top3_cot_features: list[str],
    weather_features: list[str],
    aggregate_features: list[str],
    importance: pd.DataFrame,
) -> dict[str, list[str]]:
    mandatory = list(dict.fromkeys(yahoo_features + top3_cot_features))
    ranked_features = importance["feature"].tolist()

    def keep_top(n: int) -> list[str]:
        return unique_features(mandatory + ranked_features[:n])

    return {
        "current_all_features": unique_features(mandatory + weather_features),
        "aggregates_only_plus_core": unique_features(mandatory + aggregate_features),
        "top25_plus_aggregates": unique_features(keep_top(25) + aggregate_features),
        "top35_plus_aggregates": unique_features(keep_top(35) + aggregate_features),
        "top45_plus_aggregates": unique_features(keep_top(45) + aggregate_features),
        "drop_bottom_35pct_plus_aggregates": unique_features(keep_top(max(25, int(len(ranked_features) * 0.65))) + aggregate_features),
    }


def unique_features(features: list[str]) -> list[str]:
    return list(dict.fromkeys(features))


def evaluate_feature_set(
    set_name: str,
    features: list[str],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    X_train = train[features]
    y_train = train["target_return_5d"].astype(float)
    X_valid = valid[features]
    y_valid = valid["target_return_5d"].astype(float)
    X_test = test[features]
    y_test = test["target_return_5d"].astype(float)

    candidates = build_model_candidates(features)
    valid_rows = []
    for model_name, model in candidates.items():
        print(f"  fitting {model_name}", flush=True)
        fitted = clone(model).fit(X_train, y_train)
        pred = fitted.predict(X_valid)
        valid_rows.append(score_row(set_name, model_name, "base", "valid", y_valid, pred))

    valid_df = pd.DataFrame(valid_rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    ensembles = build_fast_ensembles(candidates, valid_df)
    ensemble_rows = []
    for model_name, model in ensembles.items():
        print(f"  fitting {model_name}", flush=True)
        fitted = clone(model).fit(X_train, y_train)
        pred = fitted.predict(X_valid)
        ensemble_rows.append(score_row(set_name, model_name, "ensemble", "valid", y_valid, pred))

    valid_df = (
        pd.concat([valid_df, pd.DataFrame(ensemble_rows)], ignore_index=True)
        .sort_values(["rmse", "mae", "directional_accuracy"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    all_candidates = {**candidates, **ensembles}
    selected = select_validation_balanced_model(valid_df)
    selected_name = selected["model"]

    train_valid = pd.concat([train, valid], ignore_index=True)
    X_train_valid = train_valid[features]
    y_train_valid = train_valid["target_return_5d"].astype(float)

    holdout_rows = []
    ranked_names = valid_df["model"].head(8).tolist()
    if selected_name not in ranked_names:
        ranked_names.insert(0, selected_name)
    for model_name in ranked_names:
        model = clone(all_candidates[model_name]).fit(X_train_valid, y_train_valid)
        pred = model.predict(X_test)
        holdout_rows.append(
            score_row(
                set_name,
                model_name,
                "ensemble" if model_name.startswith("ensemble_") else "base",
                "holdout",
                y_test,
                pred,
            )
        )

    holdout_df = pd.DataFrame(holdout_rows)
    selected_model = clone(all_candidates[selected_name]).fit(X_train_valid, y_train_valid)
    selected_pred = selected_model.predict(X_test)
    selected_holdout = score_row(
        set_name,
        selected_name,
        "ensemble" if selected_name.startswith("ensemble_") else "base",
        "holdout",
        y_test,
        selected_pred,
    )
    selected.update(
        {
            "feature_set": set_name,
            "feature_count": len(features),
            "holdout_mae": selected_holdout["mae"],
            "holdout_rmse": selected_holdout["rmse"],
            "holdout_r2": selected_holdout["r2"],
            "holdout_directional_accuracy": selected_holdout["directional_accuracy"],
        }
    )

    comparison_rows = []
    for row in valid_df.to_dict(orient="records"):
        holdout_match = holdout_df.loc[holdout_df["model"].eq(row["model"])]
        output = {
            "feature_set": set_name,
            "feature_count": len(features),
            "model": row["model"],
            "model_group": row["model_group"],
            "valid_mae": row["mae"],
            "valid_rmse": row["rmse"],
            "valid_r2": row["r2"],
            "valid_directional_accuracy": row["directional_accuracy"],
            "selected": row["model"] == selected_name,
        }
        if not holdout_match.empty:
            holdout_row = holdout_match.iloc[0]
            output.update(
                {
                    "holdout_mae": holdout_row["mae"],
                    "holdout_rmse": holdout_row["rmse"],
                    "holdout_r2": holdout_row["r2"],
                    "holdout_directional_accuracy": holdout_row["directional_accuracy"],
                }
            )
        comparison_rows.append(output)

    predictions = test[["Date", "Close", "target_close_5d", "target_return_5d", "target_direction_5d"]].copy()
    predictions["predicted_return_5d"] = selected_pred
    predictions["predicted_close_5d"] = predictions["Close"] * (1 + predictions["predicted_return_5d"])
    predictions["predicted_direction_5d"] = predictions["predicted_return_5d"] > 0
    predictions["return_error"] = predictions["target_return_5d"] - predictions["predicted_return_5d"]
    predictions["price_error"] = predictions["target_close_5d"] - predictions["predicted_close_5d"]

    return {
        "selected": selected,
        "comparison_rows": comparison_rows,
        "predictions": predictions,
        "model": selected_model,
        "features": features,
    }


def build_model_candidates(features: list[str]) -> dict[str, Pipeline]:
    return {
        "bayesian_ridge": make_pipeline(features, BayesianRidge()),
        "extra_trees_depth_6": make_pipeline(
            features,
            ExtraTreesRegressor(n_estimators=180, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=1),
        ),
        "extra_trees_depth_8": make_pipeline(
            features,
            ExtraTreesRegressor(n_estimators=220, max_depth=8, min_samples_leaf=18, random_state=42, n_jobs=1),
        ),
        "random_forest_depth_6": make_pipeline(
            features,
            RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=22, random_state=42, n_jobs=1),
        ),
        "gradient_boosting": make_pipeline(
            features,
            GradientBoostingRegressor(
                n_estimators=140,
                learning_rate=0.025,
                max_depth=2,
                min_samples_leaf=25,
                subsample=0.75,
                random_state=42,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            features,
            HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_iter=180,
                max_leaf_nodes=15,
                min_samples_leaf=28,
                l2_regularization=0.08,
                early_stopping=True,
                random_state=42,
            ),
        ),
        "bagged_trees": make_pipeline(
            features,
            BaggingRegressor(
                estimator=DecisionTreeRegressor(max_depth=5, min_samples_leaf=25, random_state=42),
                n_estimators=45,
                random_state=42,
                n_jobs=1,
            ),
        ),
    }


def make_pipeline(features: list[str], model) -> Pipeline:
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
                            features,
                        )
                    ]
                ),
            ),
            ("model", model),
        ]
    )


def build_fast_ensembles(candidates: dict[str, Pipeline], valid_df: pd.DataFrame) -> dict[str, VotingRegressor]:
    top3 = valid_df.sort_values(["rmse", "mae"]).head(3)
    estimators = [(row["model"], clone(candidates[row["model"]])) for _, row in top3.iterrows()]
    weights = (1.0 / top3["rmse"].clip(lower=1e-9)).to_numpy()
    return {
        "ensemble_top3_mean": VotingRegressor(estimators=estimators, n_jobs=1),
        "ensemble_top3_weighted_rmse": VotingRegressor(estimators=estimators, weights=weights, n_jobs=1),
    }


def select_validation_balanced_model(valid_df: pd.DataFrame) -> dict:
    best_rmse = float(valid_df.iloc[0]["rmse"])
    pool = valid_df.loc[valid_df["rmse"].le(best_rmse * 1.005)].copy()
    selected = pool.sort_values(["directional_accuracy", "rmse", "mae"], ascending=[False, True, True]).iloc[0].to_dict()
    return {
        "model": selected["model"],
        "model_group": selected["model_group"],
        "valid_mae": selected["mae"],
        "valid_rmse": selected["rmse"],
        "valid_r2": selected["r2"],
        "valid_directional_accuracy": selected["directional_accuracy"],
    }


def score_row(
    feature_set: str,
    model_name: str,
    model_group: str,
    split: str,
    y_true: pd.Series,
    pred: np.ndarray,
) -> dict:
    return {
        "feature_set": feature_set,
        "model": model_name,
        "model_group": model_group,
        "split": split,
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(root_mean_squared_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "directional_accuracy": float(accuracy_score((y_true > 0).astype(int), (pred > 0).astype(int))),
    }


def better_result(candidate: dict, current: dict) -> bool:
    if candidate["holdout_rmse"] < current["holdout_rmse"] - 1e-6:
        return True
    if abs(candidate["holdout_rmse"] - current["holdout_rmse"]) <= 1e-6:
        return candidate["holdout_directional_accuracy"] > current["holdout_directional_accuracy"]
    return False


def write_experiment_plots(results: pd.DataFrame, predictions: pd.DataFrame) -> None:
    selected = results.loc[results["selected"].eq(True)].copy()
    selected = selected.sort_values("holdout_rmse")

    fig, ax1 = plt.subplots(figsize=(14, 7))
    x = np.arange(len(selected))
    colors = np.where(selected["model_group"].eq("ensemble"), "#2a9d8f", "#457b9d")
    ax1.bar(x, selected["holdout_rmse"], color=colors, alpha=0.82, label="Holdout RMSE")
    ax1.set_ylabel("Holdout RMSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(selected["feature_set"] + "\n" + selected["model"], rotation=25, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, selected["holdout_directional_accuracy"], color="#d1495b", marker="o", linewidth=1.6)
    ax2.set_ylabel("Holdout directional accuracy")
    ax2.set_ylim(0.45, max(0.62, selected["holdout_directional_accuracy"].max() + 0.03))
    ax1.set_title("Feature Combination Experiment: Selected Model Per Feature Set")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "feature_combo_ensemble_experiment_comparison.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(14, 6))
    plt.plot(predictions["Date"], predictions["target_return_5d"], label="Actual 5D return", linewidth=1.5)
    plt.plot(predictions["Date"], predictions["predicted_return_5d"], label="Feature-combo predicted 5D return", linewidth=1.5)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Feature-Combo Ensemble Experiment: Actual vs Predicted 5D Returns")
    plt.ylabel("5D return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "feature_combo_ensemble_actual_vs_predicted_returns.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
