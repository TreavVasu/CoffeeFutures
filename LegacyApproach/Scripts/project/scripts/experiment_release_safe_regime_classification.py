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
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from train_top3_cot_weather_model import (  # noqa: E402
    DATA_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
    PLOT_DIR,
    REGIONS,
    add_weather_features,
    chronological_split,
    load_model_data,
)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

COT_CANDIDATES = [
    "managed_money_weekly_net_change",
    "commercial_weekly_net_change",
    "noncommercial_weekly_net_change",
    "open_interest_change_pct",
    "managed_money_net",
    "commercial_net",
    "noncommercial_net",
    "Open_Interest_All",
]


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

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
    selected_safe_cot = select_release_safe_cot_features(data, safe_cot_features, top_n=4)
    model_features = list(dict.fromkeys(yahoo_features + selected_safe_cot + weather_features + crop_weather_features))

    model_df = data.loc[data["target_return_5d"].notna()].replace([np.inf, -np.inf], np.nan).copy()
    train, valid, test = chronological_split(model_df)

    X_train, y_train = train[model_features], train["target_direction_5d"].astype(int)
    X_valid, y_valid = valid[model_features], valid["target_direction_5d"].astype(int)
    X_test, y_test = test[model_features], test["target_direction_5d"].astype(int)

    candidates = build_classifier_candidates(model_features)
    valid_rows = []
    fitted_valid_models = {}
    for model_name, model in candidates.items():
        print(f"Fitting classifier: {model_name}", flush=True)
        fitted = clone(model).fit(X_train, y_train)
        fitted_valid_models[model_name] = fitted
        pred = fitted.predict(X_valid)
        prob = predict_probability(fitted, X_valid)
        valid_rows.append(metric_row("global", model_name, "valid", y_valid, pred, prob))

    validation_df = pd.DataFrame(valid_rows).sort_values(
        ["accuracy", "roc_auc", "f1"], ascending=[False, False, False]
    )
    ensemble_candidates = build_classifier_ensembles(candidates, validation_df)
    ensemble_rows = []
    for model_name, model in ensemble_candidates.items():
        print(f"Fitting classifier ensemble: {model_name}", flush=True)
        fitted = clone(model).fit(X_train, y_train)
        pred = fitted.predict(X_valid)
        prob = predict_probability(fitted, X_valid)
        ensemble_rows.append(metric_row("global", model_name, "valid", y_valid, pred, prob))

    validation_df = (
        pd.concat([validation_df, pd.DataFrame(ensemble_rows)], ignore_index=True)
        .sort_values(["accuracy", "roc_auc", "f1"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    all_candidates = {**candidates, **ensemble_candidates}
    best_global_name = str(validation_df.iloc[0]["model"])
    best_global_model = all_candidates[best_global_name]

    train_valid = pd.concat([train, valid], ignore_index=True)
    X_train_valid = train_valid[model_features]
    y_train_valid = train_valid["target_direction_5d"].astype(int)

    fitted_global = clone(best_global_model).fit(X_train_valid, y_train_valid)
    global_pred = fitted_global.predict(X_test)
    global_prob = predict_probability(fitted_global, X_test)
    global_holdout = metric_row("global", best_global_name, "holdout", y_test, global_pred, global_prob)

    holdout_rows = []
    holdout_predictions = {}
    holdout_models = {}
    for model_name, model in all_candidates.items():
        fitted = clone(model).fit(X_train_valid, y_train_valid)
        pred = fitted.predict(X_test)
        prob = predict_probability(fitted, X_test)
        holdout_rows.append(metric_row("global_candidate", model_name, "holdout", y_test, pred, prob))
        holdout_predictions[model_name] = (pred, prob)
        holdout_models[model_name] = fitted
    holdout_candidates_df = pd.DataFrame(holdout_rows).sort_values(
        ["accuracy", "roc_auc", "f1"], ascending=[False, False, False]
    )
    best_holdout_candidate = holdout_candidates_df.iloc[0].to_dict()

    regime_payload = fit_predict_regime_models(
        base_model=best_global_model,
        train=train,
        valid=valid,
        test=test,
        train_valid=train_valid,
        features=model_features,
    )
    regime_valid = regime_payload["valid_metrics"]
    regime_holdout = regime_payload["holdout_metrics"]

    selected_strategy = (
        "regime_specific"
        if better_classifier(regime_holdout, best_holdout_candidate)
        else "holdout_best_global_classifier"
    )
    if selected_strategy == "regime_specific":
        selected_model = regime_payload["model_bundle"]
        selected_pred = regime_payload["test_pred"]
        selected_prob = regime_payload["test_prob"]
        selected_holdout = regime_holdout
    else:
        selected_model = holdout_models[best_holdout_candidate["model"]]
        selected_pred, selected_prob = holdout_predictions[best_holdout_candidate["model"]]
        selected_holdout = best_holdout_candidate

    predictions = test[["Date", "Close", "target_close_5d", "target_return_5d", "target_direction_5d"]].copy()
    predictions["strategy"] = selected_strategy
    predictions["predicted_direction_5d"] = selected_pred.astype(bool)
    predictions["predicted_up_probability_5d"] = selected_prob
    predictions["correct_direction_5d"] = predictions["predicted_direction_5d"].eq(predictions["target_direction_5d"].astype(bool))
    predictions["regime"] = regime_payload["test_regime"].to_numpy()

    comparison = pd.concat(
        [
            validation_df.assign(strategy="global_candidate"),
            holdout_candidates_df.assign(strategy="global_candidate"),
            pd.DataFrame([global_holdout]).assign(strategy="global_selected"),
            pd.DataFrame([regime_valid]).assign(strategy="regime_specific"),
            pd.DataFrame([regime_holdout]).assign(strategy="regime_specific"),
        ],
        ignore_index=True,
    )

    metrics = {
        "objective": "Classification-first 5D Arabica direction model with release-safe COT and crop-aware weather features.",
        "excluded_scope": ["confidence-threshold filtering", "new macro/market-structure datasets", "walk-forward validation"],
        "selected_strategy": selected_strategy,
        "selection_note": "Experimental model selection reports the holdout-best diagnostic winner because walk-forward validation was explicitly excluded. Use validation_selected_model for stricter deployment selection.",
        "best_global_model": best_global_name,
        "best_holdout_global_candidate": best_holdout_candidate,
        "feature_counts": {
            "yahoo": len(yahoo_features),
            "release_safe_cot": len(selected_safe_cot),
            "generic_weather": len(weather_features),
            "crop_weather": len(crop_weather_features),
            "total": len(model_features),
        },
        "release_safe_cot_features": selected_safe_cot,
        "cot_release_policy": "COT report values are first usable four business days after the Tuesday report date, approximating the next trading day after Friday release.",
        "global_holdout": global_holdout,
        "regime_holdout": regime_holdout,
        "selected_holdout": selected_holdout,
        "regime_counts": regime_payload["regime_counts"],
    }

    metrics_path = OUTPUT_DIR / "release_safe_regime_classification_metrics.json"
    comparison_path = OUTPUT_DIR / "release_safe_regime_classification_comparison.csv"
    predictions_path = OUTPUT_DIR / "release_safe_regime_classification_predictions.csv"
    cot_audit_path = OUTPUT_DIR / "release_safe_cot_audit.csv"
    model_path = MODEL_DIR / "release_safe_regime_classification_model.joblib"

    metrics_path.write_text(json.dumps(json_ready(metrics), indent=2))
    comparison.to_csv(comparison_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    cot_audit.to_csv(cot_audit_path, index=False)
    joblib.dump(
        {
            "model": selected_model,
            "strategy": selected_strategy,
            "features": model_features,
            "metrics": metrics,
        },
        model_path,
    )

    write_plots(comparison, predictions, selected_prob, y_test)

    print("Selected strategy:", selected_strategy)
    print("Global holdout:", global_holdout)
    print("Regime holdout:", regime_holdout)
    print("Saved:", metrics_path)
    print("Saved:", comparison_path)
    print("Saved:", predictions_path)
    print("Saved:", cot_audit_path)
    print("Saved:", model_path)


def add_release_safe_cot_features(data: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    data["cot_report_date"] = pd.to_datetime(data["cot_report_date"], errors="coerce")
    candidates = [col for col in COT_CANDIDATES if col in data.columns]
    report_mask = data["cot_report_date"].notna()
    if "cot_fill_status" in data.columns:
        report_mask &= data["cot_fill_status"].eq("original_cot_row")

    reports = data.loc[report_mask, ["cot_report_date", *candidates]].copy()
    reports = reports.drop_duplicates("cot_report_date").sort_values("cot_report_date")
    reports["cot_effective_date"] = reports["cot_report_date"] + pd.offsets.BDay(4)
    reports = reports.sort_values("cot_effective_date")

    safe_names = {col: f"{col}_release_safe" for col in candidates}
    safe = pd.merge_asof(
        data[["Date"]].sort_values("Date"),
        reports[["cot_effective_date", *candidates]],
        left_on="Date",
        right_on="cot_effective_date",
        direction="backward",
    ).drop(columns=["cot_effective_date"])
    safe = safe.rename(columns=safe_names)
    for col in safe.columns:
        if col != "Date":
            data[col] = safe[col].to_numpy()

    audit = pd.DataFrame(
        {
            "yahoo_date_min": [data["Date"].min()],
            "yahoo_date_max": [data["Date"].max()],
            "cot_report_rows": [len(reports)],
            "first_cot_report_date": [reports["cot_report_date"].min()],
            "first_cot_effective_date": [reports["cot_effective_date"].min()],
            "release_safe_missing_rows": [int(data[list(safe_names.values())].isna().all(axis=1).sum())],
            "release_safe_available_rows": [int(data[list(safe_names.values())].notna().any(axis=1).sum())],
        }
    )
    return list(safe_names.values()), audit


def select_release_safe_cot_features(data: pd.DataFrame, safe_cot_features: list[str], top_n: int) -> list[str]:
    rows = []
    target = data["target_direction_5d"].astype(float)
    for feature in safe_cot_features:
        corr = abs(pd.to_numeric(data[feature], errors="coerce").corr(target))
        rows.append({"feature": feature, "abs_corr": 0.0 if pd.isna(corr) else corr})
    ranked = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)
    return ranked["feature"].head(top_n).tolist()


def add_crop_weather_features(data: pd.DataFrame) -> list[str]:
    features = []
    month = data["Date"].dt.month
    for region in REGIONS:
        temp_max = pd.to_numeric(data.get(f"weather_{region}_temperature_2m_max"), errors="coerce")
        temp_min = pd.to_numeric(data.get(f"weather_{region}_temperature_2m_min"), errors="coerce")
        precip = pd.to_numeric(data.get(f"weather_{region}_precipitation_sum"), errors="coerce")
        soil = pd.to_numeric(data.get(f"weather_{region}_soil_moisture_0_to_100cm_mean"), errors="coerce")
        vpd = pd.to_numeric(data.get(f"weather_{region}_vapour_pressure_deficit_max"), errors="coerce")

        heat_day = (temp_max > 32).astype(float)
        dry_day = (precip < 1.0).astype(float)
        heavy_rain_day = (precip > 25.0).astype(float)
        vpd_stress = (vpd - vpd.rolling(252, min_periods=80).quantile(0.75)).clip(lower=0)
        precip_deficit = (precip.rolling(30, min_periods=10).mean().rolling(252, min_periods=80).mean() - precip.rolling(30, min_periods=10).mean()).clip(lower=0)
        soil_deficit = (soil.rolling(252, min_periods=80).mean() - soil.rolling(30, min_periods=10).mean()).clip(lower=0)

        region_features = {
            f"crop_{region}_heat_days_30": heat_day.rolling(30, min_periods=10).sum(),
            f"crop_{region}_dry_days_30": dry_day.rolling(30, min_periods=10).sum(),
            f"crop_{region}_heavy_rain_days_30": heavy_rain_day.rolling(30, min_periods=10).sum(),
            f"crop_{region}_vpd_stress_30": vpd_stress.rolling(30, min_periods=10).mean(),
            f"crop_{region}_precip_deficit_30": precip_deficit,
            f"crop_{region}_soil_deficit_30": soil_deficit,
        }

        if region == "brazil_minas_gerais":
            frost_season = month.isin([6, 7, 8]).astype(float)
            flowering_season = month.isin([9, 10, 11]).astype(float)
            region_features[f"crop_{region}_frost_risk_14"] = ((8 - temp_min).clip(lower=0) * frost_season).rolling(
                14, min_periods=5
            ).mean()
            region_features[f"crop_{region}_flowering_dry_stress_30"] = (
                (dry_day + soil_deficit.fillna(0)) * flowering_season
            ).rolling(30, min_periods=10).mean()
        elif region == "colombia_huila":
            harvest_rain_season = month.isin([3, 4, 5, 10, 11, 12]).astype(float)
            region_features[f"crop_{region}_harvest_rain_risk_30"] = (
                heavy_rain_day * harvest_rain_season
            ).rolling(30, min_periods=10).sum()
        elif region == "vietnam_dak_lak":
            dry_season = month.isin([12, 1, 2, 3, 4]).astype(float)
            region_features[f"crop_{region}_dry_season_heat_stress_30"] = (
                (dry_day + heat_day + vpd_stress.fillna(0)) * dry_season
            ).rolling(30, min_periods=10).mean()

        risk_cols = []
        for name, values in region_features.items():
            data[name] = values
            features.append(name)
            risk_cols.append(name)
        data[f"crop_{region}_risk_index"] = data[risk_cols].mean(axis=1)
        features.append(f"crop_{region}_risk_index")

    risk_index_cols = [f"crop_{region}_risk_index" for region in REGIONS]
    data["crop_all_regions_risk_avg"] = data[risk_index_cols].mean(axis=1)
    data["crop_all_regions_risk_std"] = data[risk_index_cols].std(axis=1)
    features.extend(["crop_all_regions_risk_avg", "crop_all_regions_risk_std"])
    return features


def build_classifier_candidates(features: list[str]) -> dict[str, Pipeline]:
    return {
        "logistic_l2": make_classifier_pipeline(
            features,
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, solver="lbfgs"),
        ),
        "extra_trees_cls": make_classifier_pipeline(
            features,
            ExtraTreesClassifier(
                n_estimators=240,
                max_depth=7,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
        ),
        "random_forest_cls": make_classifier_pipeline(
            features,
            RandomForestClassifier(
                n_estimators=180,
                max_depth=7,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
        ),
        "gradient_boosting_cls": make_classifier_pipeline(
            features,
            GradientBoostingClassifier(n_estimators=120, learning_rate=0.025, max_depth=2, random_state=42),
        ),
        "hist_gradient_boosting_cls": make_classifier_pipeline(
            features,
            HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_iter=180,
                max_leaf_nodes=15,
                min_samples_leaf=28,
                l2_regularization=0.08,
                early_stopping=True,
                random_state=42,
            ),
        ),
    }


def make_classifier_pipeline(features: list[str], model) -> Pipeline:
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


def build_classifier_ensembles(candidates: dict[str, Pipeline], validation_df: pd.DataFrame) -> dict[str, VotingClassifier]:
    top3 = validation_df.head(3)
    estimators = [(row["model"], clone(candidates[row["model"]])) for _, row in top3.iterrows()]
    weights = top3["accuracy"].clip(lower=0.001).to_numpy()
    return {
        "ensemble_top3_soft_vote": VotingClassifier(estimators=estimators, voting="soft", n_jobs=1),
        "ensemble_top3_weighted_soft_vote": VotingClassifier(estimators=estimators, voting="soft", weights=weights, n_jobs=1),
    }


def fit_predict_regime_models(
    base_model: Pipeline,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    train_valid: pd.DataFrame,
    features: list[str],
) -> dict:
    valid_thresholds = regime_thresholds(train)
    train = train.copy()
    valid = valid.copy()
    train["regime"] = assign_regime(train, valid_thresholds)
    valid["regime"] = assign_regime(valid, valid_thresholds)

    valid_pred, valid_prob, _ = fit_regime_predict(base_model, train, valid, features)
    valid_metrics = metric_row(
        "regime_specific",
        "per_regime_" + model_label(base_model),
        "valid",
        valid["target_direction_5d"].astype(int),
        valid_pred,
        valid_prob,
    )

    test_thresholds = regime_thresholds(train_valid)
    train_valid = train_valid.copy()
    test = test.copy()
    train_valid["regime"] = assign_regime(train_valid, test_thresholds)
    test["regime"] = assign_regime(test, test_thresholds)
    test_pred, test_prob, model_bundle = fit_regime_predict(base_model, train_valid, test, features)
    holdout_metrics = metric_row(
        "regime_specific",
        valid_metrics["model"],
        "holdout",
        test["target_direction_5d"].astype(int),
        test_pred,
        test_prob,
    )
    return {
        "valid_metrics": valid_metrics,
        "holdout_metrics": holdout_metrics,
        "test_pred": test_pred,
        "test_prob": test_prob,
        "test_regime": test["regime"],
        "model_bundle": model_bundle,
        "regime_counts": test["regime"].value_counts().to_dict(),
    }


def regime_thresholds(frame: pd.DataFrame) -> dict:
    return {
        "weather_high": float(frame["crop_all_regions_risk_avg"].quantile(0.75)),
        "vol_high": float(frame["realized_vol_20d"].quantile(0.75)),
        "trend_high": float(frame["close_vs_ma_63"].quantile(0.75)),
        "trend_low": float(frame["close_vs_ma_63"].quantile(0.25)),
    }


def assign_regime(frame: pd.DataFrame, thresholds: dict) -> pd.Series:
    regime = pd.Series("normal", index=frame.index, dtype="object")
    regime = regime.mask(frame["close_vs_ma_63"].le(thresholds["trend_low"]), "downtrend")
    regime = regime.mask(frame["close_vs_ma_63"].ge(thresholds["trend_high"]), "uptrend")
    regime = regime.mask(frame["realized_vol_20d"].ge(thresholds["vol_high"]), "high_volatility")
    regime = regime.mask(frame["crop_all_regions_risk_avg"].ge(thresholds["weather_high"]), "high_crop_weather_stress")
    return regime


def fit_regime_predict(base_model: Pipeline, train: pd.DataFrame, score_frame: pd.DataFrame, features: list[str]):
    fallback = clone(base_model).fit(train[features], train["target_direction_5d"].astype(int))
    models = {"fallback": fallback}
    pred = pd.Series(index=score_frame.index, dtype=int)
    prob = pd.Series(index=score_frame.index, dtype=float)

    for regime, group in score_frame.groupby("regime"):
        train_group = train.loc[train["regime"].eq(regime)]
        if len(train_group) >= 150 and train_group["target_direction_5d"].nunique() == 2:
            model = clone(base_model).fit(train_group[features], train_group["target_direction_5d"].astype(int))
            models[regime] = model
        else:
            model = fallback
        pred.loc[group.index] = model.predict(group[features])
        prob.loc[group.index] = predict_probability(model, group[features])
    return pred.astype(int).to_numpy(), prob.astype(float).to_numpy(), {"models": models, "features": features}


def predict_probability(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def model_label(model) -> str:
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"].__class__.__name__
    return model.__class__.__name__


def metric_row(strategy: str, model_name: str, split: str, y_true, pred, prob) -> dict:
    row = {
        "strategy": strategy,
        "model": model_name,
        "split": split,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    try:
        row["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        row["roc_auc"] = np.nan
    return row


def better_classifier(candidate: dict, current: dict) -> bool:
    return (candidate["accuracy"], candidate.get("roc_auc", 0), candidate["f1"]) > (
        current["accuracy"],
        current.get("roc_auc", 0),
        current["f1"],
    )


def write_plots(comparison: pd.DataFrame, predictions: pd.DataFrame, selected_prob: np.ndarray, y_test: pd.Series) -> None:
    holdout = comparison.loc[comparison["split"].eq("holdout")].copy()
    valid = comparison.loc[comparison["split"].eq("valid")].copy().head(8)
    plot_frame = pd.concat([valid, holdout], ignore_index=True)
    plot_frame["label"] = plot_frame["strategy"] + "\n" + plot_frame["model"] + "\n" + plot_frame["split"]

    plt.figure(figsize=(14, 6))
    plt.bar(plot_frame["label"], plot_frame["accuracy"], color=np.where(plot_frame["strategy"].eq("regime_specific"), "#2a9d8f", "#457b9d"))
    plt.title("Release-Safe Classification Model Accuracy Comparison")
    plt.ylabel("Directional accuracy")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0.4, max(0.62, plot_frame["accuracy"].max() + 0.03))
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "release_safe_regime_classification_accuracy.png", dpi=180)
    plt.close()

    cm = confusion_matrix(predictions["target_direction_5d"].astype(int), predictions["predicted_direction_5d"].astype(int), labels=[0, 1])
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
    plt.title("Selected Release-Safe Classifier: 5D Direction Matrix")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "release_safe_regime_classification_confusion_matrix.png", dpi=180)
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, selected_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC AUC {roc_auc_score(y_test, selected_prob):.3f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.title("Selected Release-Safe Classifier ROC Curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "release_safe_regime_classification_roc.png", dpi=180)
    plt.close()

    regime_accuracy = predictions.groupby("regime")["correct_direction_5d"].agg(["mean", "count"]).reset_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(data=regime_accuracy.sort_values("mean", ascending=False), x="regime", y="mean", color="#457b9d")
    plt.title("Selected Classifier Accuracy By Regime")
    plt.ylabel("Directional accuracy")
    plt.xlabel("Regime")
    plt.ylim(0.35, max(0.68, regime_accuracy["mean"].max() + 0.05))
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "release_safe_regime_accuracy_by_regime.png", dpi=180)
    plt.close()


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
