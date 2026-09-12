from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "Scripts/project/artifacts/.matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from integrate_news_scores_final_model import (
    DAILY_COLUMNS, WEEKLY_COLUMNS, FEATURES, LIMITATIONS, TARGET_RETURN,
    bootstrap_lift, daily_news_features, metric_row, weekly_news_features,
)
from train_research_paper_ensemble import (
    add_paper_positioning_features, build_feature_groups, is_cot_feature,
    load_prepared_data, make_pipeline, purged_chronological_split, split_summary,
)


OUTPUTS = ROOT / "Scripts/project/artifacts/outputs"
MODELS = ROOT / "Scripts/project/artifacts/models"
PLOTS = ROOT / "Scripts/project/artifacts/plots"
PREFIX = "arabica_all_inputs_news"


def build_frame():
    frame, audit = load_prepared_data(ROOT / "data/centralData/arabica_ml_model_ready.csv", 5)
    expected = frame["Close"].shift(-5) / frame["Close"] - 1
    np.testing.assert_allclose(frame[TARGET_RETURN], expected, atol=1e-10, equal_nan=True)
    frame = add_paper_positioning_features(frame)
    calendar = pd.DatetimeIndex(frame["Date"])
    daily = pd.read_csv(OUTPUTS / "gdelt_coffee_events_2000_2026_daily_summary.csv", usecols=DAILY_COLUMNS)
    weekly = pd.read_csv(OUTPUTS / "gdelt_coffee_events_2000_2026_weekly_summary.csv", usecols=WEEKLY_COLUMNS)
    frame = frame.merge(daily_news_features(daily, calendar), on="Date", validate="one_to_one")
    frame = pd.merge_asof(frame, weekly_news_features(weekly, calendar), left_on="Date", right_on="weekly_available_date")
    frame["_row_id"] = np.arange(len(frame))
    return frame, audit


def model_metrics(y_frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    return metric_row(y_frame.assign(prediction=prediction), "prediction")


def used_feature_counts(pipeline, columns: list[str], source_features: dict) -> dict:
    estimator = pipeline.named_steps["model"]
    if hasattr(estimator, "coef_"):
        used = set(np.flatnonzero(estimator.coef_))
    else:
        used = set()
        for iteration in estimator._predictors:
            for tree in iteration:
                nodes = tree.nodes
                used.update(nodes["feature_idx"][nodes["is_leaf"] == 0].tolist())
    used_names = {columns[int(index)] for index in used}
    return {source: len(set(names) & used_names) for source, names in source_features.items()}


def write_plot(comparison: pd.DataFrame):
    data = comparison.loc[comparison.phase.eq("test")].set_index("variant")
    names = ["without_news", "structured_news", "all_inputs"]
    labels = ["Price + COT + weather", "+ News indicators", "+ News text"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, key, title in zip(axes, ["rmse", "directional_accuracy"], ["5-day return RMSE (%)", "Direction accuracy (%)"]):
        bars = axis.bar(labels, data.loc[names, key] * 100, color=["#547890", "#cb8b3e", "#358b68"])
        axis.bar_label(bars, fmt="%.4f" if key == "rmse" else "%.2f", padding=4)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=13)
        axis.set_ylim(0, 100 if key == "directional_accuracy" else data.loc[names, key].max() * 125)
    fig.suptitle("Arabica Coffee C returns: direct all-input comparison")
    fig.tight_layout()
    fig.savefig(PLOTS / f"{PREFIX}_evaluation.png", dpi=160)
    plt.close(fig)


def main():
    for path in [OUTPUTS, MODELS, PLOTS]:
        path.mkdir(parents=True, exist_ok=True)
    frame, data_audit = build_frame()
    labeled = frame.loc[frame[TARGET_RETURN].notna()].copy()
    split = purged_chronological_split(labeled, 0.70, 0.15, 5)
    groups, feature_audit = build_feature_groups(frame, split["train"], 0.20)
    all_features = groups["integrated"]
    features = {
        "without_news": [col for col in all_features if not col.startswith("news_")],
        "structured_news": [col for col in all_features if not col.startswith("news_weekly_text_")],
        "all_inputs": all_features,
    }
    source_features = {
        "yahoo_price": [col for col in all_features if not col.startswith(("weather_", "news_")) and not is_cot_feature(col)],
        "cot_positioning": [col for col in all_features if is_cot_feature(col)],
        "weather": [col for col in all_features if col.startswith("weather_")],
        "news": [col for col in all_features if col.startswith("news_")],
        "news_text_subset": [col for col in all_features if col.startswith("news_weekly_text_")],
    }
    if not all(source_features.values()):
        raise ValueError("Every requested source, including news text, must have nonconstant training features.")
    recipes = {
        "ridge": make_pipeline(Ridge(alpha=25.0), scale=True),
        "hist_gradient_boosting": make_pipeline(HistGradientBoostingRegressor(
            learning_rate=0.04, max_iter=100, max_leaf_nodes=15, min_samples_leaf=25,
            l2_regularization=1.0, early_stopping=False, random_state=42,
        ), scale=False),
    }
    validation_recipes = {}
    for name, recipe in recipes.items():
        fitted = clone(recipe).fit(split["train"][features["without_news"]], split["train"][TARGET_RETURN])
        validation_recipes[name] = model_metrics(split["valid"], fitted.predict(split["valid"][features["without_news"]]))
    learner = min(validation_recipes, key=lambda name: validation_recipes[name]["rmse"])
    print(f"Learner selected on no-news validation RMSE: {learner}; fitting paired feature variants", flush=True)
    rows, eval_models = [], {}
    prediction_frame = split["test"][["Date", "Close", TARGET_RETURN]].copy()
    validation = {}
    for variant, columns in features.items():
        for phase, train, test in [
            ("validation", split["train"], split["valid"]),
            ("test", split["dev_train"], split["test"]),
        ]:
            fitted = clone(recipes[learner]).fit(train[columns], train[TARGET_RETURN])
            prediction = fitted.predict(test[columns])
            metrics = model_metrics(test, prediction)
            rows.append({"phase": phase, "variant": variant, "learner": learner, **metrics})
            if phase == "validation":
                validation[variant] = metrics
            else:
                eval_models[variant] = fitted
                prediction_frame[f"{variant}_predicted_return_5d"] = prediction
    selected_variant = min(validation, key=lambda name: validation[name]["rmse"])
    comparison = pd.DataFrame(rows)
    test_metrics = comparison.loc[comparison.phase.eq("test")].set_index("variant")
    base, news = test_metrics.loc["without_news"], test_metrics.loc["all_inputs"]
    uncertainty_frame = prediction_frame.rename(columns={
        "without_news_predicted_return_5d": "calibrated_base_predicted_return_5d",
        "all_inputs_predicted_return_5d": "final_news_predicted_return_5d",
    })
    historical = pd.read_csv(OUTPUTS / "research_paper_model_holdout_predictions.csv", parse_dates=["Date"])
    old = prediction_frame[["Date", TARGET_RETURN]].merge(
        historical[["Date", "research_ensemble_predicted_return_5d"]], on="Date", validate="one_to_one",
    )
    assert len(old) == len(prediction_frame)
    yearly = {}
    for year, part in prediction_frame.groupby(prediction_frame["Date"].dt.year):
        yearly[str(year)] = {variant: metric_row(part, f"{variant}_predicted_return_5d") for variant in features}
    production = {}
    latest = frame.tail(1).copy()
    for variant in ["without_news", "all_inputs"]:
        production[variant] = clone(recipes[learner]).fit(labeled[features[variant]], labeled[TARGET_RETURN])
    latest_output = latest[["Date", "Close", "daily_latest_source_date", "weekly_latest_source_date"]].copy()
    for variant, fitted in production.items():
        latest_output[f"{variant}_predicted_return_5d"] = fitted.predict(latest[features[variant]])
        latest_output[f"{variant}_predicted_close_5d"] = latest_output.Close * (1 + latest_output[f"{variant}_predicted_return_5d"])
    metadata = {
        "objective": "Predict Arabica Coffee C five-trading-day futures returns from Yahoo price history, COT positioning, weather and text-derived news features.",
        "target_formula": "Close[t+5] / Close[t] - 1", "forecast_timing": "after close on Date",
        "inputs": {"prepared_market_cot_weather": "data/centralData/arabica_ml_model_ready.csv",
                   "daily_news": "Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_daily_summary.csv",
                   "weekly_news": "Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_weekly_summary.csv"},
        "data_audit": data_audit, "feature_audit": feature_audit,
        "source_feature_counts": {key: len(value) for key, value in source_features.items()},
        "source_features": source_features, "split": split_summary(split),
        "learner": learner, "validation_learner_metrics": validation_recipes,
        "learner_selection": "Choose recipe using without-news validation RMSE; use identical recipe, rows and splits in all ablations.",
        "selected_variant_by_validation": selected_variant,
        "comparison": rows, "yearly_test_metrics": yearly,
        "news_delta_vs_without_news": {
            "rmse_delta": float(news.rmse - base.rmse),
            "rmse_relative_change_pct": float(100 * (news.rmse / base.rmse - 1)),
            "directional_accuracy_delta_pp": float(100 * (news.directional_accuracy - base.directional_accuracy)),
            "mae_delta": float(news.mae - base.mae),
        },
        "news_bootstrap_vs_without_news": {**bootstrap_lift(uncertainty_frame, 42), "reference": "without_news"},
        "features_used_by_fitted_model": {
            "evaluation": used_feature_counts(eval_models["all_inputs"], all_features, source_features),
            "production": used_feature_counts(production["all_inputs"], all_features, source_features),
        },
        "existing_research_ensemble_same_dates": metric_row(old, "research_ensemble_predicted_return_5d"),
        "news_method": "Ten candidate structured/text indicators from local GDELT summaries. Text uses fixed disruption, support and coffee-relevance term counts in event digests. Training-constant features are dropped. All news delayed six trading sessions because upstream event selection used future five-session returns. No impact scores or future-price fields are features.",
        "conclusion": ("Adding news improved both test RMSE and direction accuracy." if news.rmse < base.rmse and news.directional_accuracy > base.directional_accuracy else "Adding news did not improve both test RMSE and direction accuracy; see separate metric changes."),
        "limitations": LIMITATIONS + ["The direct all-input model is a separate experiment from the notebook's price-only selected ensemble. COT represents aggregate participant positioning, not individual counterparty records.",
            "The local text is structured event-summary text; these rule-based indicators are not a trained article-sentiment model. Delayed, retrospectively selected news cannot establish immediate news alpha."],
        "evaluation_model_fit_end": str(split["dev_train"].Date.max().date()),
        "production_model_fit_end": str(labeled.Date.max().date()),
        "latest_prediction_date": str(latest.Date.iloc[0].date()),
    }
    comparison.to_csv(OUTPUTS / f"{PREFIX}_comparison.csv", index=False)
    prediction_frame.to_csv(OUTPUTS / f"{PREFIX}_holdout_predictions.csv", index=False)
    latest_output.to_csv(OUTPUTS / f"{PREFIX}_latest_predictions.csv", index=False)
    (OUTPUTS / f"{PREFIX}_metrics.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    joblib.dump({"model": production["all_inputs"], "features": features["all_inputs"],
                 "without_news_model": production["without_news"], "without_news_features": features["without_news"],
                 "evaluation_models": eval_models, "feature_variants": features, "metadata": metadata},
                MODELS / f"{PREFIX}_model.joblib")
    write_plot(comparison)
    print(comparison.to_string(index=False), flush=True)
    print(json.dumps({key: metadata[key] for key in ["source_feature_counts", "selected_variant_by_validation", "news_delta_vs_without_news", "conclusion"]}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
