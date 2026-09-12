from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from experiment_release_safe_regime_classification import (  # noqa: E402
    add_crop_weather_features,
    add_release_safe_cot_features,
    better_classifier,
    fit_predict_regime_models,
    make_classifier_pipeline,
    metric_row,
    predict_probability,
)
from train_top3_cot_weather_model import (  # noqa: E402
    MODEL_DIR,
    OUTPUT_DIR,
    PLOT_DIR,
    add_weather_features,
    load_model_data,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation for release-safe Arabica 5D direction model.")
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--min-train-rows", type=int, default=1500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    data, features, feature_metadata, cot_audit = build_walk_forward_frame()
    years = sorted(data["Date"].dt.year.unique())
    end_year = args.end_year or max(years)
    test_years = [year for year in years if args.start_year <= year <= end_year]

    fold_rows = []
    prediction_parts = []
    for test_year in test_years:
        fold = run_year_fold(data, features, test_year, args.min_train_rows)
        if fold is None:
            continue
        fold_rows.append(fold["summary"])
        prediction_parts.append(fold["predictions"])

    if not fold_rows:
        raise RuntimeError("No walk-forward folds were produced. Lower --start-year or --min-train-rows.")

    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values("Date").reset_index(drop=True)
    overall = overall_metrics(predictions)

    metrics = {
        "objective": "Walk-forward validation for 5D Arabica direction using release-safe COT, crop-weather features, global classifiers, ensembles, and regime-specific classifiers.",
        "fold_design": "For each test year, train on data before the prior calendar year, validate on the prior calendar year, then refit on train+validation and test on the target calendar year.",
        "start_year": int(folds["test_year"].min()),
        "end_year": int(folds["test_year"].max()),
        "fold_count": int(len(folds)),
        "overall": overall,
        "feature_counts": feature_metadata,
        "cot_release_policy": "COT report values are first usable four business days after the Tuesday report date, approximating the next trading day after Friday release.",
    }

    folds_path = OUTPUT_DIR / "walk_forward_release_safe_classification_folds.csv"
    predictions_path = OUTPUT_DIR / "walk_forward_release_safe_classification_predictions.csv"
    metrics_path = OUTPUT_DIR / "walk_forward_release_safe_classification_metrics.json"
    cot_audit_path = OUTPUT_DIR / "walk_forward_release_safe_cot_audit.csv"

    folds.to_csv(folds_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(json_ready(metrics), indent=2))
    cot_audit.to_csv(cot_audit_path, index=False)

    write_plots(folds, predictions)

    print("Walk-forward overall:")
    print(pd.Series(overall).to_string())
    print("Fold summary:")
    print(folds[["test_year", "selected_strategy", "selected_model", "test_accuracy", "test_roc_auc", "test_rows"]].to_string(index=False))
    print("Saved:", metrics_path)
    print("Saved:", folds_path)
    print("Saved:", predictions_path)
    print("Saved:", cot_audit_path)


def build_walk_forward_frame() -> tuple[pd.DataFrame, list[str], dict, pd.DataFrame]:
    data = load_model_data()
    weather_features, _, _ = add_weather_features(data)
    safe_cot_features, cot_audit = add_release_safe_cot_features(data)
    crop_weather_features = add_crop_weather_features(data)

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
    features = list(dict.fromkeys(yahoo_features + safe_cot_features + weather_features + crop_weather_features))
    frame = data.loc[data["target_return_5d"].notna()].replace([np.inf, -np.inf], np.nan).copy()
    frame = frame.sort_values("Date").reset_index(drop=True)
    metadata = {
        "yahoo": len(yahoo_features),
        "release_safe_cot": len(safe_cot_features),
        "generic_weather": len(weather_features),
        "crop_weather": len(crop_weather_features),
        "total": len(features),
    }
    return frame, features, metadata, cot_audit


def build_walk_forward_candidates(features: list[str]) -> dict:
    """Smaller candidate set designed to be rerun across many yearly folds."""
    return {
        "logistic_l2": make_classifier_pipeline(
            features,
            LogisticRegression(max_iter=1200, class_weight="balanced", C=0.5, solver="lbfgs"),
        ),
        "extra_trees_fast": make_classifier_pipeline(
            features,
            ExtraTreesClassifier(
                n_estimators=120,
                max_depth=6,
                min_samples_leaf=24,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
        ),
        "random_forest_fast": make_classifier_pipeline(
            features,
            RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=24,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
        ),
        "gradient_boosting_fast": make_classifier_pipeline(
            features,
            GradientBoostingClassifier(n_estimators=80, learning_rate=0.03, max_depth=2, random_state=42),
        ),
    }


def build_walk_forward_ensembles(candidates: dict, validation_df: pd.DataFrame) -> dict:
    top2 = validation_df.head(2)
    estimators = [(row["model"], clone(candidates[row["model"]])) for _, row in top2.iterrows()]
    weights = top2["accuracy"].clip(lower=0.001).to_numpy()
    return {
        "ensemble_top2_soft_vote": VotingClassifier(estimators=estimators, voting="soft", n_jobs=1),
        "ensemble_top2_weighted_soft_vote": VotingClassifier(estimators=estimators, voting="soft", weights=weights, n_jobs=1),
    }


def run_year_fold(data: pd.DataFrame, features: list[str], test_year: int, min_train_rows: int) -> dict | None:
    valid_year = test_year - 1
    train = data.loc[data["Date"].dt.year < valid_year].copy()
    valid = data.loc[data["Date"].dt.year.eq(valid_year)].copy()
    test = data.loc[data["Date"].dt.year.eq(test_year)].copy()
    if len(train) < min_train_rows or len(valid) < 80 or len(test) < 20:
        return None

    X_train = train[features]
    y_train = train["target_direction_5d"].astype(int)
    X_valid = valid[features]
    y_valid = valid["target_direction_5d"].astype(int)

    print(f"Walk-forward fold {test_year}: train={len(train)} valid={len(valid)} test={len(test)}", flush=True)
    candidates = build_walk_forward_candidates(features)
    validation_rows = []
    for model_name, model in candidates.items():
        fitted = clone(model).fit(X_train, y_train)
        pred = fitted.predict(X_valid)
        prob = predict_probability(fitted, X_valid)
        validation_rows.append(metric_row("global", model_name, "valid", y_valid, pred, prob))

    validation_df = pd.DataFrame(validation_rows).sort_values(
        ["accuracy", "roc_auc", "f1"], ascending=[False, False, False]
    )
    ensembles = build_walk_forward_ensembles(candidates, validation_df)
    ensemble_rows = []
    for model_name, model in ensembles.items():
        fitted = clone(model).fit(X_train, y_train)
        pred = fitted.predict(X_valid)
        prob = predict_probability(fitted, X_valid)
        ensemble_rows.append(metric_row("global", model_name, "valid", y_valid, pred, prob))

    validation_df = (
        pd.concat([validation_df, pd.DataFrame(ensemble_rows)], ignore_index=True)
        .sort_values(["accuracy", "roc_auc", "f1"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    all_candidates = {**candidates, **ensembles}
    best_global_valid = validation_df.iloc[0].to_dict()
    best_global_model = all_candidates[best_global_valid["model"]]

    train_valid = pd.concat([train, valid], ignore_index=True)
    y_test = test["target_direction_5d"].astype(int)

    fitted_global = clone(best_global_model).fit(train_valid[features], train_valid["target_direction_5d"].astype(int))
    global_pred = fitted_global.predict(test[features])
    global_prob = predict_probability(fitted_global, test[features])
    global_test = metric_row("global", best_global_valid["model"], "test", y_test, global_pred, global_prob)

    regime_payload = fit_predict_regime_models(
        base_model=best_global_model,
        train=train,
        valid=valid,
        test=test,
        train_valid=train_valid,
        features=features,
    )
    regime_valid = regime_payload["valid_metrics"]
    regime_test = regime_payload["holdout_metrics"].copy()
    regime_test["split"] = "test"

    if better_classifier(regime_valid, best_global_valid):
        selected_strategy = "regime_specific"
        selected_model = regime_test["model"]
        selected_pred = regime_payload["test_pred"]
        selected_prob = regime_payload["test_prob"]
        selected_test = regime_test
    else:
        selected_strategy = "global"
        selected_model = global_test["model"]
        selected_pred = global_pred
        selected_prob = global_prob
        selected_test = global_test

    predictions = test[["Date", "Close", "target_return_5d", "target_direction_5d"]].copy()
    predictions["test_year"] = test_year
    predictions["selected_strategy"] = selected_strategy
    predictions["selected_model"] = selected_model
    predictions["predicted_direction_5d"] = selected_pred.astype(bool)
    predictions["predicted_up_probability_5d"] = selected_prob
    predictions["correct_direction_5d"] = predictions["predicted_direction_5d"].eq(predictions["target_direction_5d"].astype(bool))
    predictions["regime"] = regime_payload["test_regime"].to_numpy()

    summary = {
        "test_year": test_year,
        "train_start": train["Date"].min().date().isoformat(),
        "train_end": train["Date"].max().date().isoformat(),
        "valid_start": valid["Date"].min().date().isoformat(),
        "valid_end": valid["Date"].max().date().isoformat(),
        "test_start": test["Date"].min().date().isoformat(),
        "test_end": test["Date"].max().date().isoformat(),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "selected_strategy": selected_strategy,
        "selected_model": selected_model,
        "valid_global_model": best_global_valid["model"],
        "valid_global_accuracy": best_global_valid["accuracy"],
        "valid_regime_accuracy": regime_valid["accuracy"],
        "test_global_accuracy": global_test["accuracy"],
        "test_regime_accuracy": regime_test["accuracy"],
        "test_accuracy": selected_test["accuracy"],
        "test_precision": selected_test["precision"],
        "test_recall": selected_test["recall"],
        "test_f1": selected_test["f1"],
        "test_roc_auc": selected_test["roc_auc"],
    }
    return {"summary": summary, "predictions": predictions}


def overall_metrics(predictions: pd.DataFrame) -> dict:
    y_true = predictions["target_direction_5d"].astype(int)
    y_pred = predictions["predicted_direction_5d"].astype(int)
    prob = predictions["predicted_up_probability_5d"].astype(float)
    return {
        "rows": int(len(predictions)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "first_test_date": predictions["Date"].min().date().isoformat(),
        "last_test_date": predictions["Date"].max().date().isoformat(),
    }


def write_plots(folds: pd.DataFrame, predictions: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(folds["test_year"], folds["test_accuracy"], marker="o", linewidth=2, label="Selected strategy")
    plt.plot(folds["test_year"], folds["test_global_accuracy"], marker="s", linewidth=1.4, label="Global selected")
    plt.plot(folds["test_year"], folds["test_regime_accuracy"], marker="^", linewidth=1.4, label="Regime-specific")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.title("Walk-Forward 5D Direction Accuracy By Test Year")
    plt.xlabel("Test year")
    plt.ylabel("Directional accuracy")
    plt.ylim(0.35, max(0.70, folds[["test_accuracy", "test_global_accuracy", "test_regime_accuracy"]].max().max() + 0.04))
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "walk_forward_release_safe_accuracy_by_year.png", dpi=180)
    plt.close()

    cm = confusion_matrix(
        predictions["target_direction_5d"].astype(int),
        predictions["predicted_direction_5d"].astype(int),
        labels=[0, 1],
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=False,
        xticklabels=["Pred Down/Flat", "Pred Up"],
        yticklabels=["Actual Down/Flat", "Actual Up"],
    )
    plt.title("Walk-Forward Direction Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "walk_forward_release_safe_confusion_matrix.png", dpi=180)
    plt.close()

    fpr, tpr, _ = roc_curve(
        predictions["target_direction_5d"].astype(int),
        predictions["predicted_up_probability_5d"].astype(float),
    )
    auc_value = roc_auc_score(
        predictions["target_direction_5d"].astype(int),
        predictions["predicted_up_probability_5d"].astype(float),
    )
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC AUC {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.title("Walk-Forward ROC Curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "walk_forward_release_safe_roc.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=folds, x="selected_strategy", color="#457b9d")
    plt.title("Walk-Forward Selected Strategy Count")
    plt.xlabel("Selected strategy")
    plt.ylabel("Fold count")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "walk_forward_release_safe_strategy_counts.png", dpi=180)
    plt.close()


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
