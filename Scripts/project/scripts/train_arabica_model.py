from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - xgboost is optional at runtime
    XGBClassifier = None
    XGBRegressor = None

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "Scripts" / "project" / "src"))

from arabica_modeling import (  # noqa: E402
    TARGET_CLOSE,
    TARGET_DIRECTION,
    TARGET_RETURN,
    build_modeling_frame,
    chronological_split,
    feature_frame_for_model,
    fetch_open_meteo_weather,
    safe_json_dump,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Arabica Coffee C futures return model.")
    parser.add_argument("--central-data", default="data/centralData/yahoo_cot_full_outer_by_date.csv")
    parser.add_argument("--weather-cache", default="data/weather/open_meteo_coffee_regions_daily.csv")
    parser.add_argument("--skip-weather", action="store_true", help="Do not fetch or use weather features.")
    parser.add_argument("--force-weather", action="store_true", help="Refresh Open-Meteo cache.")
    parser.add_argument("--models-dir", default="Scripts/project/artifacts/models")
    parser.add_argument("--plots-dir", default="Scripts/project/artifacts/plots")
    parser.add_argument("--outputs-dir", default="Scripts/project/artifacts/outputs")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--include-xgboost", action="store_true", help="Include slower XGBoost candidates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ROOT
    central_path = root / args.central_data
    weather_path = None if args.skip_weather else root / args.weather_cache

    if weather_path is not None:
        dates = pd.read_csv(central_path, usecols=["Date"])
        start_date = pd.to_datetime(dates["Date"]).min().date().isoformat()
        end_date = pd.to_datetime(dates["Date"]).max().date().isoformat()
        print(f"Fetching/loading Open-Meteo weather cache: {start_date} to {end_date}", flush=True)
        fetch_open_meteo_weather(start_date, end_date, weather_path, force=args.force_weather)

    feature_set = build_modeling_frame(central_path, weather_path=weather_path)
    frame = feature_set.frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.loc[frame[TARGET_RETURN].notna()].copy()
    train, valid, test = chronological_split(frame)

    print(f"Rows: train={len(train)}, valid={len(valid)}, test={len(test)}", flush=True)
    print(f"Date ranges: train={train['Date'].min().date()}..{train['Date'].max().date()}, "
          f"valid={valid['Date'].min().date()}..{valid['Date'].max().date()}, "
          f"test={test['Date'].min().date()}..{test['Date'].max().date()}", flush=True)
    print(f"Numeric features: {len(feature_set.numeric_features)}; dropped features: {len(feature_set.dropped_features)}", flush=True)

    X_train = feature_frame_for_model(train, feature_set.numeric_features)
    X_valid = feature_frame_for_model(valid, feature_set.numeric_features)
    X_test = feature_frame_for_model(test, feature_set.numeric_features)
    y_train = train[TARGET_RETURN].astype(float)
    y_valid = valid[TARGET_RETURN].astype(float)
    y_test = test[TARGET_RETURN].astype(float)
    y_train_cls = train[TARGET_DIRECTION].astype(int)
    y_valid_cls = valid[TARGET_DIRECTION].astype(int)
    y_test_cls = test[TARGET_DIRECTION].astype(int)

    regressors = build_regressor_candidates(args.random_state, include_xgboost=args.include_xgboost)
    classifiers = build_classifier_candidates(args.random_state, include_xgboost=args.include_xgboost)

    reg_results, best_reg_name, best_reg = fit_regressors(regressors, X_train, y_train, X_valid, y_valid)
    clf_results, best_clf_name, best_clf = fit_classifiers(
        classifiers, X_train, y_train_cls, X_valid, y_valid_cls
    )

    train_valid = pd.concat([train, valid], ignore_index=True)
    X_train_valid = feature_frame_for_model(train_valid, feature_set.numeric_features)
    y_train_valid = train_valid[TARGET_RETURN].astype(float)
    y_train_valid_cls = train_valid[TARGET_DIRECTION].astype(int)

    best_reg.fit(X_train_valid, y_train_valid)
    best_clf.fit(X_train_valid, y_train_valid_cls)

    test_pred_return = best_reg.predict(X_test)
    test_pred_prob = positive_class_probability(best_clf, X_test)
    test_pred_direction = (test_pred_prob >= 0.5).astype(int)

    reg_test_metrics = regression_metrics(y_test, test_pred_return)
    reg_test_metrics["directional_accuracy_from_return"] = float(
        accuracy_score(y_test_cls, (test_pred_return > 0).astype(int))
    )
    clf_test_metrics = classification_metrics(y_test_cls, test_pred_direction, test_pred_prob)

    plots_dir = root / args.plots_dir
    outputs_dir = root / args.outputs_dir
    models_dir = root / args.models_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    prediction_frame = make_prediction_frame(test, y_test, test_pred_return, test_pred_prob)
    prediction_csv = outputs_dir / "arabica_holdout_predictions.csv"
    prediction_frame.to_csv(prediction_csv, index=False)

    write_plots(
        plots_dir=plots_dir,
        outputs_dir=outputs_dir,
        reg_results=reg_results,
        clf_results=clf_results,
        y_test=y_test,
        y_test_cls=y_test_cls,
        test_pred_return=test_pred_return,
        test_pred_prob=test_pred_prob,
        prediction_frame=prediction_frame,
        best_reg=best_reg,
        train_valid=train_valid,
        test=test,
        numeric_features=feature_set.numeric_features,
    )

    metrics = {
        "target": "5-trading-day future return",
        "central_data": str(central_path),
        "weather_cache": str(weather_path) if weather_path is not None else None,
        "model_selection": {
            "best_regressor": best_reg_name,
            "best_classifier": best_clf_name,
            "validation_regression": reg_results,
            "validation_classification": clf_results,
        },
        "holdout_test": {
            "regression": reg_test_metrics,
            "classification": clf_test_metrics,
            "confusion_matrix": confusion_matrix(y_test_cls, test_pred_direction).tolist(),
            "date_start": test["Date"].min(),
            "date_end": test["Date"].max(),
            "rows": len(test),
        },
        "features": {
            "numeric_count": len(feature_set.numeric_features),
            "dropped_count": len(feature_set.dropped_features),
            "dropped_features": feature_set.dropped_features,
            "uses_weather": weather_path is not None and weather_path.exists(),
            "uses_cot_release_lag": True,
            "cot_release_lag_days": 3,
            "removed_useless_metadata": True,
        },
        "artifacts": {
            "model": str(models_dir / "arabica_returns_model.joblib"),
            "predictions": str(prediction_csv),
            "plots_dir": str(plots_dir),
        },
    }
    metrics_path = outputs_dir / "arabica_model_metrics.json"
    safe_json_dump(metrics_path, metrics)

    bundle = {
        "regressor": best_reg,
        "classifier": best_clf,
        "numeric_features": feature_set.numeric_features,
        "dropped_features": feature_set.dropped_features,
        "target_return": TARGET_RETURN,
        "target_direction": TARGET_DIRECTION,
        "target_close": TARGET_CLOSE,
        "metrics": metrics,
        "training_rows": len(train_valid),
        "training_end_date": train_valid["Date"].max().isoformat(),
    }
    model_path = models_dir / "arabica_returns_model.joblib"
    joblib.dump(bundle, model_path)

    print("Best regressor:", best_reg_name, reg_test_metrics)
    print("Best classifier:", best_clf_name, clf_test_metrics)
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Saved holdout predictions:", prediction_csv)
    print("Saved plots:", plots_dir)


def build_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
        ],
        remainder="drop",
    )


def make_pipeline(estimator, numeric_features: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features)),
            ("model", estimator),
        ]
    )


def build_regressor_candidates(random_state: int, include_xgboost: bool = False):
    base = [
        ("ridge_alpha_1", Ridge(alpha=1.0)),
        (
            "extra_trees_depth_8_leaf_15",
            ExtraTreesRegressor(
                n_estimators=240,
                max_depth=8,
                min_samples_leaf=15,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "hist_gbr_l2_005",
            HistGradientBoostingRegressor(
                learning_rate=0.035,
                max_iter=180,
                max_leaf_nodes=15,
                l2_regularization=0.05,
                early_stopping=True,
                random_state=random_state,
            ),
        ),
        (
            "rf_depth_6_leaf_20",
            RandomForestRegressor(
                n_estimators=180,
                max_depth=6,
                min_samples_leaf=20,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "rf_depth_10_leaf_10",
            RandomForestRegressor(
                n_estimators=220,
                max_depth=10,
                min_samples_leaf=10,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ]
    if include_xgboost and XGBRegressor is not None:
        base.extend(
            [
                (
                    "xgb_depth_2_lr_003",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=220,
                        max_depth=2,
                        learning_rate=0.03,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=3.0,
                        random_state=random_state,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
                (
                    "xgb_depth_3_lr_004",
                    XGBRegressor(
                        objective="reg:squarederror",
                        n_estimators=220,
                        max_depth=3,
                        learning_rate=0.04,
                        subsample=0.80,
                        colsample_bytree=0.80,
                        reg_lambda=5.0,
                        random_state=random_state,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        )
    return base


def build_classifier_candidates(random_state: int, include_xgboost: bool = False):
    base = [
        (
            "logistic_l2",
            LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5, solver="liblinear"),
        ),
        (
            "extra_trees_classifier_depth_8",
            ExtraTreesClassifier(
                n_estimators=240,
                max_depth=8,
                min_samples_leaf=15,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "hist_gbc_l2_005",
            HistGradientBoostingClassifier(
                learning_rate=0.035,
                max_iter=180,
                max_leaf_nodes=15,
                l2_regularization=0.05,
                early_stopping=True,
                random_state=random_state,
            ),
        ),
        (
            "rf_classifier_depth_6",
            RandomForestClassifier(
                n_estimators=180,
                max_depth=6,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ]
    if include_xgboost and XGBClassifier is not None:
        base.extend(
            [
                (
                    "xgb_classifier_depth_2",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=220,
                        max_depth=2,
                        learning_rate=0.03,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=4.0,
                        random_state=random_state,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
                (
                    "xgb_classifier_depth_3",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=220,
                        max_depth=3,
                        learning_rate=0.035,
                        subsample=0.80,
                        colsample_bytree=0.80,
                        reg_lambda=6.0,
                        random_state=random_state,
                        n_jobs=-1,
                        tree_method="hist",
                    ),
                ),
            ]
        )
    return base


def fit_regressors(candidates, X_train, y_train, X_valid, y_valid):
    results = {}
    fitted = {}
    numeric_features = list(X_train.columns)
    for name, estimator in candidates:
        pipeline = make_pipeline(estimator, numeric_features)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_valid)
        results[name] = regression_metrics(y_valid, pred)
        results[name]["directional_accuracy_from_return"] = float(
            accuracy_score((y_valid > 0).astype(int), (pred > 0).astype(int))
        )
        fitted[name] = pipeline
        print("regressor", name, results[name], flush=True)
    best_name = min(results, key=lambda name: (results[name]["rmse"], -results[name]["directional_accuracy_from_return"]))
    return results, best_name, fitted[best_name]


def fit_classifiers(candidates, X_train, y_train, X_valid, y_valid):
    results = {}
    fitted = {}
    numeric_features = list(X_train.columns)
    for name, estimator in candidates:
        pipeline = make_pipeline(estimator, numeric_features)
        pipeline.fit(X_train, y_train)
        prob = positive_class_probability(pipeline, X_valid)
        pred = (prob >= 0.5).astype(int)
        results[name] = classification_metrics(y_valid, pred, prob)
        fitted[name] = pipeline
        print("classifier", name, results[name], flush=True)
    best_name = max(results, key=lambda name: (results[name]["roc_auc"], results[name]["accuracy"]))
    return results, best_name, fitted[best_name]


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "correlation": float(np.corrcoef(np.asarray(y_true), np.asarray(y_pred))[0, 1]),
    }


def classification_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def positive_class_probability(model: Pipeline, X) -> np.ndarray:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def make_prediction_frame(test, y_test, pred_return, pred_prob) -> pd.DataFrame:
    out = test[["Date", "Close", TARGET_CLOSE, TARGET_RETURN, TARGET_DIRECTION]].copy()
    out["predicted_return_5d"] = pred_return
    out["predicted_direction_probability"] = pred_prob
    out["predicted_direction"] = pred_prob >= 0.5
    out["predicted_close_5d"] = out["Close"] * (1.0 + out["predicted_return_5d"])
    out["absolute_return_error"] = (np.asarray(y_test) - pred_return).round(10)
    out["absolute_price_error"] = out[TARGET_CLOSE] - out["predicted_close_5d"]
    return out


def write_plots(
    plots_dir: Path,
    outputs_dir: Path,
    reg_results: dict,
    clf_results: dict,
    y_test,
    y_test_cls,
    test_pred_return,
    test_pred_prob,
    prediction_frame: pd.DataFrame,
    best_reg: Pipeline,
    train_valid: pd.DataFrame,
    test: pd.DataFrame,
    numeric_features: list[str],
) -> None:
    sns.set_theme(style="whitegrid")

    reg_df = pd.DataFrame(reg_results).T.sort_values("rmse")
    plt.figure(figsize=(11, 6))
    reg_df[["mae", "rmse"]].plot(kind="bar", ax=plt.gca())
    plt.title("Validation Regression Error By Model")
    plt.ylabel("Return error")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "regression_metrics.png", dpi=180)
    plt.close()

    clf_df = pd.DataFrame(clf_results).T.sort_values("roc_auc", ascending=False)
    plt.figure(figsize=(11, 6))
    clf_df[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind="bar", ax=plt.gca())
    plt.title("Validation Classification Metrics By Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "classification_accuracy_metrics.png", dpi=180)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test_cls, test_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, label=f"Holdout ROC AUC = {roc_auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No-skill")
    plt.title("Holdout ROC Curve: 5D Return Direction")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plots_dir / "roc_curve.png", dpi=180)
    plt.close()

    plot_frame = prediction_frame.sort_values("Date").copy()
    plt.figure(figsize=(14, 6))
    plt.plot(plot_frame["Date"], plot_frame[TARGET_RETURN], label="Actual 5D return", linewidth=1.5)
    plt.plot(plot_frame["Date"], plot_frame["predicted_return_5d"], label="Predicted 5D return", linewidth=1.5)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Actual Vs Predicted Arabica Futures Returns")
    plt.ylabel("5D return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "actual_vs_predicted_returns.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 6))
    plt.plot(plot_frame["Date"], plot_frame[TARGET_CLOSE], label="Actual future close", linewidth=1.5)
    plt.plot(plot_frame["Date"], plot_frame["predicted_close_5d"], label="Predicted future close", linewidth=1.5)
    plt.title("Real Price Vs Predicted Price: 5 Trading Days Ahead")
    plt.ylabel("Coffee C futures close")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "real_vs_predicted_price.png", dpi=180)
    plt.close()

    feature_importance = get_feature_importance(best_reg)
    if feature_importance is not None and not feature_importance.empty:
        plt.figure(figsize=(10, 8))
        top = feature_importance.head(25).sort_values("importance")
        plt.barh(top["feature"], top["importance"])
        plt.title("Top Regression Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_importance.png", dpi=180)
        plt.close()

    write_drift_and_cot_diagnostics(
        plots_dir=plots_dir,
        outputs_dir=outputs_dir,
        train_valid=train_valid,
        test=test,
        prediction_frame=prediction_frame,
        numeric_features=numeric_features,
    )


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_") and not hasattr(model, "coef_"):
        return None
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    else:
        importance = np.abs(np.asarray(model.coef_)).ravel()
    if len(names) != len(importance):
        return None
    return (
        pd.DataFrame({"feature": names, "importance": importance})
        .sort_values("importance", ascending=False)
        .query("importance > 0")
    )


def write_drift_and_cot_diagnostics(
    plots_dir: Path,
    outputs_dir: Path,
    train_valid: pd.DataFrame,
    test: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    numeric_features: list[str],
) -> None:
    plot_frame = prediction_frame.sort_values("Date").copy()
    plot_frame["abs_return_error"] = (plot_frame[TARGET_RETURN] - plot_frame["predicted_return_5d"]).abs()
    plot_frame["rolling_63d_rmse"] = (
        (plot_frame[TARGET_RETURN] - plot_frame["predicted_return_5d"]) ** 2
    ).rolling(63, min_periods=20).mean().pow(0.5)
    plot_frame["rolling_63d_direction_accuracy"] = (
        (plot_frame[TARGET_DIRECTION].astype(bool) == (plot_frame["predicted_return_5d"] > 0))
        .astype(float)
        .rolling(63, min_periods=20)
        .mean()
    )

    plt.figure(figsize=(14, 7))
    plt.plot(plot_frame["Date"], plot_frame["rolling_63d_rmse"], label="Rolling 63-trading-day RMSE", linewidth=2)
    plt.plot(
        plot_frame["Date"],
        plot_frame["abs_return_error"].rolling(21, min_periods=5).mean(),
        label="Rolling 21-trading-day absolute error",
        linewidth=1.5,
        alpha=0.85,
    )
    worst = plot_frame.nlargest(8, "abs_return_error")
    plt.scatter(worst["Date"], worst["abs_return_error"], color="crimson", s=35, label="Largest error dates")
    plt.title("Model Error Drift Over Holdout Period")
    plt.ylabel("Return error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "prediction_error_drift.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 4.8))
    plt.plot(plot_frame["Date"], plot_frame["rolling_63d_direction_accuracy"], linewidth=2)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.title("Rolling Direction Accuracy Drift")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(plots_dir / "rolling_direction_accuracy.png", dpi=180)
    plt.close()

    diagnostics = build_drift_diagnostics(train_valid, test, prediction_frame, numeric_features)
    diagnostics.to_csv(outputs_dir / "feature_drift_cot_correlation_report.csv", index=False)

    top_drift = diagnostics.sort_values("abs_standardized_mean_diff", ascending=False).head(30)
    if not top_drift.empty:
        plt.figure(figsize=(11, 8))
        top_plot = top_drift.sort_values("abs_standardized_mean_diff")
        plt.barh(top_plot["feature"], top_plot["standardized_mean_diff"])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title("Largest Train-To-Holdout Feature Drift")
        plt.xlabel("Standardized mean difference")
        plt.tight_layout()
        plt.savefig(plots_dir / "feature_drift_top.png", dpi=180)
        plt.close()

    cot_diag = diagnostics.loc[diagnostics["is_cot_feature"]].copy()
    if not cot_diag.empty:
        top_target = cot_diag.reindex(cot_diag["corr_with_target_train"].abs().sort_values(ascending=False).index).head(25)
        plt.figure(figsize=(10, 8))
        target_plot = top_target.sort_values("corr_with_target_train")
        plt.barh(target_plot["feature"], target_plot["corr_with_target_train"])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title("COT Feature Correlation With Future 5D Returns")
        plt.xlabel("Pearson correlation on train+validation")
        plt.tight_layout()
        plt.savefig(plots_dir / "cot_target_correlation.png", dpi=180)
        plt.close()

        top_error = cot_diag.reindex(cot_diag["corr_with_abs_error_test"].abs().sort_values(ascending=False).index).head(25)
        plt.figure(figsize=(10, 8))
        error_plot = top_error.sort_values("corr_with_abs_error_test")
        plt.barh(error_plot["feature"], error_plot["corr_with_abs_error_test"])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title("COT Feature Correlation With Holdout Absolute Error")
        plt.xlabel("Pearson correlation on holdout")
        plt.tight_layout()
        plt.savefig(plots_dir / "cot_error_correlation.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 7))
        plt.scatter(
            cot_diag["standardized_mean_diff"],
            cot_diag["corr_with_abs_error_test"],
            alpha=0.65,
            s=28,
        )
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.axvline(0, color="gray", linestyle="--", linewidth=1)
        plt.title("COT Drift Vs Error Correlation")
        plt.xlabel("Train-to-holdout standardized drift")
        plt.ylabel("Correlation with absolute forecast error")
        plt.tight_layout()
        plt.savefig(plots_dir / "cot_drift_vs_error_correlation.png", dpi=180)
        plt.close()

    worst_dates = explain_worst_error_dates(train_valid, test, prediction_frame, diagnostics)
    worst_dates.to_csv(outputs_dir / "worst_error_dates_with_drift_context.csv", index=False)


def build_drift_diagnostics(
    train_valid: pd.DataFrame,
    test: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    numeric_features: list[str],
) -> pd.DataFrame:
    pred = prediction_frame[["Date", TARGET_RETURN, "predicted_return_5d"]].copy()
    pred["signed_error"] = pred[TARGET_RETURN] - pred["predicted_return_5d"]
    pred["abs_error"] = pred["signed_error"].abs()
    test_with_error = test.merge(pred[["Date", "signed_error", "abs_error"]], on="Date", how="left")

    rows = []
    for feature in numeric_features:
        train_values = pd.to_numeric(train_valid[feature], errors="coerce")
        test_values = pd.to_numeric(test[feature], errors="coerce")
        train_std = train_values.std(skipna=True)
        pooled_std = train_std if pd.notna(train_std) and train_std > 1e-12 else np.nan
        train_mean = train_values.mean(skipna=True)
        test_mean = test_values.mean(skipna=True)
        standardized_diff = (test_mean - train_mean) / pooled_std if pd.notna(pooled_std) else np.nan
        rows.append(
            {
                "feature": feature,
                "is_cot_feature": is_cot_feature(feature),
                "is_weather_feature": feature.startswith("weather_"),
                "train_mean": train_mean,
                "test_mean": test_mean,
                "train_std": train_std,
                "standardized_mean_diff": standardized_diff,
                "abs_standardized_mean_diff": abs(standardized_diff) if pd.notna(standardized_diff) else np.nan,
                "corr_with_target_train": safe_corr(train_values, train_valid[TARGET_RETURN]),
                "corr_with_target_test": safe_corr(test_values, test[TARGET_RETURN]),
                "corr_with_abs_error_test": safe_corr(test_with_error[feature], test_with_error["abs_error"]),
                "corr_with_signed_error_test": safe_corr(test_with_error[feature], test_with_error["signed_error"]),
                "missing_rate_train": float(train_values.isna().mean()),
                "missing_rate_test": float(test_values.isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_standardized_mean_diff", ascending=False)


def explain_worst_error_dates(
    train_valid: pd.DataFrame,
    test: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    pred = prediction_frame.sort_values("Date").copy()
    pred["abs_return_error"] = (pred[TARGET_RETURN] - pred["predicted_return_5d"]).abs()
    worst = pred.nlargest(25, "abs_return_error").copy()
    context_features = diagnostics.head(40)["feature"].tolist()
    train_stats = train_valid[context_features].agg(["mean", "std"]).T
    test_lookup = test.set_index("Date")

    explanations = []
    for _, row in worst.iterrows():
        date = row["Date"]
        if date not in test_lookup.index:
            explanations.append("")
            continue
        values = test_lookup.loc[date, context_features]
        z_scores = ((values - train_stats["mean"]) / train_stats["std"].replace(0, np.nan)).abs()
        z_scores = z_scores.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False).head(4)
        pieces = [f"{feature} z={score:.1f}" for feature, score in z_scores.items()]
        explanations.append("; ".join(pieces))
    worst["dominant_shift_context"] = explanations
    return worst[
        [
            "Date",
            "Close",
            TARGET_CLOSE,
            TARGET_RETURN,
            "predicted_return_5d",
            "predicted_close_5d",
            "abs_return_error",
            "absolute_price_error",
            "dominant_shift_context",
        ]
    ]


def is_cot_feature(feature: str) -> bool:
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
    return any(token in feature for token in tokens)


def safe_corr(left, right) -> float:
    pair = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")],
        axis=1,
    ).dropna()
    if len(pair) < 20:
        return np.nan
    if pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


if __name__ == "__main__":
    main()
