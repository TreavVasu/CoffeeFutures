from __future__ import annotations

import argparse
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
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.append(str(SCRIPT_DIR))

from train_top3_cot_weather_model import (  # noqa: E402
    MODEL_DIR,
    OUTPUT_DIR,
    PLOT_DIR,
    add_weather_features,
    load_model_data,
    select_top3_cot_features,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Arabica 5D return models with and without GDELT weekly event features."
    )
    parser.add_argument(
        "--weekly-events",
        default="project/artifacts/outputs/gdelt_coffee_event_weekly_scored_2000.csv",
    )
    parser.add_argument("--year", type=int, default=2000)
    parser.add_argument("--train-size", type=float, default=0.60)
    parser.add_argument("--valid-size", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    data, base_features = build_model_frame(args.weekly_events, args.year)
    train, valid, test = chronological_period_split(data, args.train_size, args.valid_size)
    feature_sets = prune_feature_sets(build_feature_sets(base_features), train, valid)

    comparison_rows = []
    prediction_parts = []
    selected_models = {}
    feature_importance_parts = []

    for set_name, payload in feature_sets.items():
        features = payload["features"]
        result = evaluate_feature_set(set_name, features, payload["notes"], train, valid, test)
        comparison_rows.extend(result["comparison_rows"])
        prediction_parts.append(result["predictions"])
        selected_models[set_name] = result["model_payload"]
        feature_importance_parts.append(result["feature_importance"])

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["selected", "rmse", "directional_accuracy"],
        ascending=[False, True, False],
    )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    feature_importance = pd.concat(feature_importance_parts, ignore_index=True)
    diff = build_prediction_diff(predictions)
    metric_diff = build_metric_diff(comparison)

    comparison_path = OUTPUT_DIR / f"gdelt_event_model_comparison_{args.year}.csv"
    predictions_path = OUTPUT_DIR / f"gdelt_event_model_predictions_{args.year}.csv"
    diff_path = OUTPUT_DIR / f"gdelt_event_model_prediction_diff_{args.year}.csv"
    metric_diff_path = OUTPUT_DIR / f"gdelt_event_model_metric_diff_{args.year}.csv"
    feature_importance_path = OUTPUT_DIR / f"gdelt_event_model_feature_importance_{args.year}.csv"
    metrics_path = OUTPUT_DIR / f"gdelt_event_model_metrics_{args.year}.json"
    model_path = MODEL_DIR / f"gdelt_event_feature_comparison_models_{args.year}.joblib"

    comparison.to_csv(comparison_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    diff.to_csv(diff_path, index=False)
    metric_diff.to_csv(metric_diff_path, index=False)
    feature_importance.to_csv(feature_importance_path, index=False)

    metrics = {
        "year": args.year,
        "objective": "Compare Arabica 5D return model performance with weekly GDELT event features over the event-covered period.",
        "period_scope": {
            "start": data["Date"].min().date().isoformat(),
            "end": data["Date"].max().date().isoformat(),
            "rows": int(len(data)),
            "weekly_periods": int(data["period_id"].nunique()),
        },
        "split": {
            "train": date_span(train),
            "valid": date_span(valid),
            "test": date_span(test),
        },
        "feature_sets": {
            name: {
                "feature_count": len(payload["features"]),
                "notes": payload["notes"],
            }
            for name, payload in feature_sets.items()
        },
        "selected_results": comparison.loc[comparison["selected"]].to_dict(orient="records"),
        "metric_diff_vs_baseline": metric_diff.to_dict(orient="records"),
        "leakage_note": (
            "event_safe uses raw/event-count features only. event_scored_explanatory includes Ollama/final period scores "
            "that were produced with Coffee C price/COT context and should be treated as explanatory/audit features, not deployment-safe predictors."
        ),
    }
    metrics_path.write_text(json.dumps(json_ready(metrics), indent=2))
    joblib.dump(selected_models, model_path)
    write_plots(comparison, predictions, diff, feature_importance, args.year)

    print("Selected model comparison:")
    print(comparison.loc[comparison["selected"]].to_string(index=False))
    print("\nMetric diff vs baseline:")
    print(metric_diff.to_string(index=False))
    print("Saved:", comparison_path)
    print("Saved:", predictions_path)
    print("Saved:", diff_path)
    print("Saved:", metric_diff_path)
    print("Saved:", feature_importance_path)
    print("Saved:", metrics_path)
    print("Saved:", model_path)


def build_model_frame(weekly_events_path: str, year: int) -> tuple[pd.DataFrame, list[str]]:
    data = load_model_data()
    top3_cot_features = select_top3_cot_features(data)
    add_weather_features(data)

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
    weather_core = [
        "weather_brazil_minas_gerais_stress_index",
        "weather_colombia_huila_stress_index",
        "weather_vietnam_dak_lak_stress_index",
    ]
    weather_core = [col for col in weather_core if col in data.columns]
    base_features = list(dict.fromkeys(yahoo_features + top3_cot_features + weather_core))

    events = load_weekly_event_features(resolve_path(weekly_events_path))
    data["period_start"] = data["Date"].dt.to_period("W-SUN").dt.start_time.dt.normalize()
    data["period_id"] = data["period_start"].dt.strftime("%Y-%m-%d")
    merged = data.merge(events, on="period_id", how="inner", suffixes=("", "_event"))
    merged = merged.loc[
        merged["Date"].dt.year.eq(year) & merged["target_return_5d"].notna() & merged["Close"].notna()
    ].copy()
    merged = add_event_derived_features(merged)
    merged = merged.replace([np.inf, -np.inf], np.nan).sort_values("Date").reset_index(drop=True)
    if len(merged) < 30:
        raise RuntimeError(f"Only {len(merged)} rows available after merging weekly events; not enough for comparison.")
    return merged, base_features


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_weekly_event_features(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path)
    numeric_cols = [
        "event_count",
        "unique_event_ids",
        "event_day_count",
        "explicit_coffee_event_count",
        "producer_country_event_count",
        "bullish_event_count",
        "bearish_event_count",
        "neutral_event_count",
        "mean_event_intensity_score_0_1",
        "event_volume_score_0_1",
        "direction_concentration_score_0_1",
        "ollama_period_impact_score_0_1",
        "ollama_confidence_0_1",
        "deterministic_period_score_0_1",
        "final_period_impact_score_0_1",
    ]
    keep = ["period_id", "ollama_direction", "ollama_status", *numeric_cols]
    events = events[[col for col in keep if col in events.columns]].copy()
    for col in numeric_cols:
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce")
    return events


def add_event_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    event_count = out["event_count"].replace(0, np.nan)
    out["gdelt_log_event_count"] = np.log1p(out["event_count"])
    out["gdelt_explicit_coffee_share"] = out["explicit_coffee_event_count"] / event_count
    out["gdelt_producer_country_share"] = out["producer_country_event_count"] / event_count
    out["gdelt_bullish_share"] = out["bullish_event_count"] / event_count
    out["gdelt_bearish_share"] = out["bearish_event_count"] / event_count
    out["gdelt_neutral_share"] = out["neutral_event_count"] / event_count
    out["gdelt_net_bullish_share"] = (out["bullish_event_count"] - out["bearish_event_count"]) / event_count
    out["gdelt_intensity_x_volume"] = out["mean_event_intensity_score_0_1"] * out["gdelt_log_event_count"]
    out["gdelt_ollama_bullish"] = out["ollama_direction"].fillna("").eq("bullish").astype(float)
    out["gdelt_ollama_bearish"] = out["ollama_direction"].fillna("").eq("bearish").astype(float)
    out["gdelt_ollama_neutral"] = out["ollama_direction"].fillna("").eq("neutral").astype(float)
    return out


def build_feature_sets(base_features: list[str]) -> dict[str, dict]:
    safe_event_features = [
        "event_count",
        "unique_event_ids",
        "event_day_count",
        "explicit_coffee_event_count",
        "producer_country_event_count",
        "bullish_event_count",
        "bearish_event_count",
        "neutral_event_count",
        "mean_event_intensity_score_0_1",
        "event_volume_score_0_1",
        "direction_concentration_score_0_1",
        "gdelt_log_event_count",
        "gdelt_explicit_coffee_share",
        "gdelt_producer_country_share",
        "gdelt_bullish_share",
        "gdelt_bearish_share",
        "gdelt_neutral_share",
        "gdelt_net_bullish_share",
        "gdelt_intensity_x_volume",
    ]
    scored_event_features = safe_event_features + [
        "ollama_period_impact_score_0_1",
        "ollama_confidence_0_1",
        "deterministic_period_score_0_1",
        "final_period_impact_score_0_1",
        "gdelt_ollama_bullish",
        "gdelt_ollama_bearish",
        "gdelt_ollama_neutral",
    ]
    return {
        "baseline_no_events": {
            "features": base_features,
            "notes": "Yahoo + top COT + compact weather features only.",
        },
        "event_safe": {
            "features": list(dict.fromkeys(base_features + safe_event_features)),
            "notes": "Adds raw weekly event-count/intensity/direction features only; avoids Ollama/final scores.",
        },
        "event_scored_explanatory": {
            "features": list(dict.fromkeys(base_features + scored_event_features)),
            "notes": "Adds weekly Ollama/final impact scores. Explanatory/audit only because scoring used price/COT context.",
        },
    }


def prune_feature_sets(feature_sets: dict[str, dict], train: pd.DataFrame, valid: pd.DataFrame) -> dict[str, dict]:
    training_frame = pd.concat([train, valid], ignore_index=True)
    pruned = {}
    for name, payload in feature_sets.items():
        kept = []
        dropped = []
        for feature in payload["features"]:
            if feature not in training_frame.columns:
                dropped.append(feature)
            elif training_frame[feature].notna().any():
                kept.append(feature)
            else:
                dropped.append(feature)
        pruned[name] = {
            **payload,
            "features": kept,
            "dropped_empty_features": dropped,
        }
    return pruned


def chronological_period_split(frame: pd.DataFrame, train_size: float, valid_size: float):
    frame = frame.sort_values("Date").reset_index(drop=True)
    n_rows = len(frame)
    train_end = max(10, int(n_rows * train_size))
    valid_end = max(train_end + 5, int(n_rows * (train_size + valid_size)))
    valid_end = min(valid_end, n_rows - 5)
    return frame.iloc[:train_end].copy(), frame.iloc[train_end:valid_end].copy(), frame.iloc[valid_end:].copy()


def evaluate_feature_set(
    set_name: str,
    features: list[str],
    notes: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    candidates = build_candidates(features)
    valid_rows = []
    fitted_valid = {}
    for model_name, model in candidates.items():
        fitted = clone(model).fit(train[features], train["target_return_5d"].astype(float))
        fitted_valid[model_name] = fitted
        pred = fitted.predict(valid[features])
        valid_rows.append(metric_row(set_name, model_name, "valid", valid, pred, selected=False, notes=notes))
    valid_df = pd.DataFrame(valid_rows).sort_values(["rmse", "mae", "directional_accuracy"], ascending=[True, True, False])
    selected_model_name = str(valid_df.iloc[0]["model"])
    selected_model = candidates[selected_model_name]

    train_valid = pd.concat([train, valid], ignore_index=True)
    final_model = clone(selected_model).fit(train_valid[features], train_valid["target_return_5d"].astype(float))
    test_pred = final_model.predict(test[features])
    test_row = metric_row(set_name, selected_model_name, "test", test, test_pred, selected=True, notes=notes)

    predictions = test[
        [
            "Date",
            "period_id",
            "Close",
            "target_close_5d",
            "target_return_5d",
            "target_direction_5d",
            "final_period_impact_score_0_1",
            "ollama_period_impact_score_0_1",
            "ollama_direction",
            "event_count",
        ]
    ].copy()
    predictions["feature_set"] = set_name
    predictions["model"] = selected_model_name
    predictions["predicted_return_5d"] = test_pred
    predictions["predicted_close_5d"] = predictions["Close"] * (1 + predictions["predicted_return_5d"])
    predictions["predicted_direction_5d"] = predictions["predicted_return_5d"] > 0
    predictions["absolute_return_error"] = (predictions["target_return_5d"] - predictions["predicted_return_5d"]).abs()
    predictions["direction_correct"] = predictions["predicted_direction_5d"].eq(predictions["target_direction_5d"].astype(bool))

    importance = extract_feature_importance(final_model, features)
    importance["feature_set"] = set_name
    importance["model"] = selected_model_name

    comparison_rows = valid_df.to_dict(orient="records") + [test_row]
    return {
        "comparison_rows": comparison_rows,
        "predictions": predictions,
        "model_payload": {
            "feature_set": set_name,
            "model": final_model,
            "features": features,
            "selected_model": selected_model_name,
            "notes": notes,
            "test_metrics": test_row,
        },
        "feature_importance": importance,
    }


def build_candidates(features: list[str]) -> dict[str, Pipeline]:
    def make_pipeline(model) -> Pipeline:
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

    return {
        "ridge_alpha_3": make_pipeline(Ridge(alpha=3.0)),
        "ridge_alpha_10": make_pipeline(Ridge(alpha=10.0)),
        "bayesian_ridge": make_pipeline(BayesianRidge()),
        "gradient_boosting_small": make_pipeline(
            GradientBoostingRegressor(
                n_estimators=60,
                learning_rate=0.025,
                max_depth=2,
                min_samples_leaf=8,
                random_state=42,
            )
        ),
        "extra_trees_shallow": make_pipeline(
            ExtraTreesRegressor(n_estimators=120, max_depth=4, min_samples_leaf=6, random_state=42, n_jobs=1)
        ),
        "random_forest_shallow": make_pipeline(
            RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=6, random_state=42, n_jobs=1)
        ),
    }


def metric_row(
    feature_set: str,
    model_name: str,
    split: str,
    frame: pd.DataFrame,
    pred: np.ndarray,
    selected: bool,
    notes: str,
) -> dict:
    y = frame["target_return_5d"].astype(float)
    return {
        "feature_set": feature_set,
        "model": model_name,
        "split": split,
        "selected": bool(selected),
        "rows": int(len(frame)),
        "date_start": frame["Date"].min().date().isoformat(),
        "date_end": frame["Date"].max().date().isoformat(),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(root_mean_squared_error(y, pred)),
        "r2": float(r2_score(y, pred)) if len(frame) >= 2 else np.nan,
        "directional_accuracy": float(accuracy_score((y > 0).astype(int), (pred > 0).astype(int))),
        "mean_predicted_return": float(np.mean(pred)),
        "mean_actual_return": float(y.mean()),
        "notes": notes,
    }


def extract_feature_importance(model: Pipeline, features: list[str]) -> pd.DataFrame:
    fitted = model.named_steps["model"]
    if hasattr(fitted, "feature_importances_"):
        values = np.asarray(fitted.feature_importances_, dtype=float)
    elif hasattr(fitted, "coef_"):
        values = np.abs(np.asarray(fitted.coef_, dtype=float)).ravel()
    else:
        values = np.zeros(len(features), dtype=float)
    if values.sum() > 0:
        values = values / values.sum()
    return pd.DataFrame({"feature": features, "importance": values}).sort_values("importance", ascending=False)


def build_prediction_diff(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = predictions.loc[predictions["feature_set"].eq("baseline_no_events")].copy()
    baseline = baseline.rename(
        columns={
            "predicted_return_5d": "baseline_predicted_return_5d",
            "absolute_return_error": "baseline_absolute_return_error",
            "direction_correct": "baseline_direction_correct",
        }
    )
    keep = ["Date", "baseline_predicted_return_5d", "baseline_absolute_return_error", "baseline_direction_correct"]
    pieces = []
    for feature_set in ["event_safe", "event_scored_explanatory"]:
        current = predictions.loc[predictions["feature_set"].eq(feature_set)].copy()
        merged = current.merge(baseline[keep], on="Date", how="left")
        merged["return_prediction_delta_vs_baseline"] = (
            merged["predicted_return_5d"] - merged["baseline_predicted_return_5d"]
        )
        merged["absolute_error_delta_vs_baseline"] = (
            merged["absolute_return_error"] - merged["baseline_absolute_return_error"]
        )
        merged["direction_correct_delta_vs_baseline"] = (
            merged["direction_correct"].astype(int) - merged["baseline_direction_correct"].astype(int)
        )
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def build_metric_diff(comparison: pd.DataFrame) -> pd.DataFrame:
    test = comparison.loc[comparison["selected"] & comparison["split"].eq("test")].copy()
    baseline = test.loc[test["feature_set"].eq("baseline_no_events")].iloc[0]
    rows = []
    for _, row in test.iterrows():
        rows.append(
            {
                "feature_set": row["feature_set"],
                "model": row["model"],
                "rows": int(row["rows"]),
                "rmse": row["rmse"],
                "rmse_delta_vs_baseline": row["rmse"] - baseline["rmse"],
                "mae": row["mae"],
                "mae_delta_vs_baseline": row["mae"] - baseline["mae"],
                "directional_accuracy": row["directional_accuracy"],
                "directional_accuracy_delta_vs_baseline": row["directional_accuracy"]
                - baseline["directional_accuracy"],
                "r2": row["r2"],
                "r2_delta_vs_baseline": row["r2"] - baseline["r2"],
                "notes": row["notes"],
            }
        )
    return pd.DataFrame(rows)


def write_plots(
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    diff: pd.DataFrame,
    feature_importance: pd.DataFrame,
    year: int,
) -> None:
    selected = comparison.loc[comparison["selected"] & comparison["split"].eq("test")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=selected, x="feature_set", y="rmse", ax=axes[0], color="#457b9d")
    axes[0].set_title(f"GDELT Event Feature Test RMSE, {year}")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25)
    sns.barplot(data=selected, x="feature_set", y="directional_accuracy", ax=axes[1], color="#2a9d8f")
    axes[1].set_title(f"GDELT Event Feature Directional Accuracy, {year}")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"gdelt_event_model_metric_comparison_{year}.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(14, 6))
    actual = predictions.drop_duplicates("Date").sort_values("Date")
    plt.plot(actual["Date"], actual["target_return_5d"], color="black", linewidth=2.0, label="Actual 5D return")
    for feature_set, group in predictions.groupby("feature_set"):
        group = group.sort_values("Date")
        plt.plot(group["Date"], group["predicted_return_5d"], linewidth=1.4, label=feature_set)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.title(f"Actual Vs Predicted 5D Return With GDELT Event Features, {year}")
    plt.ylabel("5D return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"gdelt_event_model_actual_vs_predicted_{year}.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 5))
    sns.lineplot(data=diff, x="Date", y="absolute_error_delta_vs_baseline", hue="feature_set", marker="o")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(f"Prediction Error Delta Vs Baseline After Adding GDELT Features, {year}")
    plt.ylabel("Absolute return error delta")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"gdelt_event_model_error_delta_{year}.png", dpi=180)
    plt.close()

    event_importance = feature_importance.loc[feature_importance["feature"].str.contains("gdelt|event|ollama", case=False, regex=True)].copy()
    if not event_importance.empty:
        top = event_importance.sort_values("importance", ascending=False).head(20)
        plt.figure(figsize=(12, 7))
        sns.barplot(data=top, y="feature", x="importance", hue="feature_set", dodge=False)
        plt.title(f"Top GDELT Event Feature Importance, {year}")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"gdelt_event_model_feature_importance_{year}.png", dpi=180)
        plt.close()


def date_span(frame: pd.DataFrame) -> dict:
    return {
        "start": frame["Date"].min().date().isoformat(),
        "end": frame["Date"].max().date().isoformat(),
        "rows": int(len(frame)),
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


if __name__ == "__main__":
    main()
