from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
TARGET_RETURN = "target_return_5d"
BASE_PREDICTION = "predicted_return_5d"
HORIZON = 5
# Upstream retained events were ranked using their subsequent five-session return.
NEWS_DELAY = HORIZON + 1
PREDICTIONS = {
    "base_model": BASE_PREDICTION,
    "calibrated_base": "calibrated_base_predicted_return_5d",
    "news_only": "news_only_predicted_return_5d",
    "final_news_blend": "final_news_predicted_return_5d",
}
DAILY_COLUMNS = ["trading_date", "event_count", "bullish_events", "bearish_events"]
WEEKLY_COLUMNS = [
    "period_id", "period_end", "event_count", "bullish_event_count",
    "bearish_event_count", "explicit_coffee_event_count", "mean_event_intensity_score_0_1",
    "top_event_digest",
]
FEATURES = [
    "news_daily_event_count_log_5d", "news_daily_bull_minus_bear_share_5d",
    "news_weekly_event_count_log", "news_weekly_bull_minus_bear_share",
    "news_weekly_direction_concentration", "news_weekly_explicit_coffee_share",
    "news_weekly_mean_event_intensity",
    "news_weekly_text_disruption", "news_weekly_text_support", "news_weekly_text_coffee",
]
LIMITATIONS = [
    "The earlier 61.35% directional accuracy and positive news-lift conclusion are withdrawn: "
    "the impact scores and event selection used future returns.",
    "The stored daily-cap selection uses future five-session returns. Even counts and intensity "
    "are delayed until those returns have matured; this evaluates delayed selected-news summaries, "
    "not immediate news sentiment. Raw unfiltered history is not available locally.",
    "No original publication-time snapshots are available. The base prediction pipeline is reused, "
    "not independently certified as point-in-time correct. No separately identified Astra artifact was found.",
    "Five-session returns overlap. Block-bootstrap intervals and chronological checks describe this "
    "local sample; the previously inspected holdout is not a new untouched test set.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate delayed local news summaries against existing Arabica predictions.")
    for name, default in {
        "base-predictions": "Scripts/project/artifacts/outputs/arabica_holdout_predictions.csv",
        "weekly-news": "Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_weekly_summary.csv",
        "daily-news": "Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_daily_summary.csv",
        "price-calendar": "data/centralData/yahoo_cot_full_outer_by_date.csv",
        "outputs-dir": "Scripts/project/artifacts/outputs",
        "plots-dir": "Scripts/project/artifacts/plots",
        "models-dir": "Scripts/project/artifacts/models",
    }.items():
        parser.add_argument(f"--{name}", default=default)
    parser.add_argument("--calibration-size", type=float, default=0.50)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("All inputs and outputs must remain inside this repository.")
    return path


def load_calendar(path: Path) -> pd.DatetimeIndex:
    price = pd.read_csv(path, usecols=["Date", "Close"], parse_dates=["Date"])
    return pd.DatetimeIndex(price.loc[price["Close"].notna(), "Date"].dropna().drop_duplicates().sort_values())


def daily_news_features(daily: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    daily = daily[DAILY_COLUMNS].copy().rename(columns={"trading_date": "Date"})
    daily["Date"] = pd.to_datetime(daily["Date"])
    if daily["Date"].duplicated().any():
        raise ValueError("Daily news dates must be unique.")
    daily = daily.set_index("Date").reindex(calendar)
    counts = daily[["event_count", "bullish_events", "bearish_events"]].apply(pd.to_numeric, errors="coerce")
    values = pd.DataFrame({
        "news_daily_event_count_log_5d": np.log1p(counts["event_count"]),
        "news_daily_bull_minus_bear_share_5d": (
            (counts["bullish_events"] - counts["bearish_events"]) / counts["event_count"].replace(0, np.nan)
        ),
    }, index=calendar)
    values = values.shift(NEWS_DELAY).rolling(5, min_periods=1).mean()
    values["daily_latest_source_date"] = pd.Series(calendar, index=calendar).shift(NEWS_DELAY)
    return values.rename_axis("Date").reset_index()


def weekly_news_features(weekly: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    weekly = weekly[WEEKLY_COLUMNS].copy()
    ends = pd.to_datetime(weekly["period_end"])
    last_session = calendar.searchsorted(ends, side="right") - 1
    available_session = last_session + NEWS_DELAY
    valid = (last_session >= 0) & (available_session < len(calendar))
    weekly = weekly.loc[valid].copy()
    weekly["weekly_available_date"] = calendar[available_session[valid]].to_numpy()
    weekly["weekly_latest_source_date"] = calendar[last_session[valid]].to_numpy()
    for col in WEEKLY_COLUMNS[2:-1]:
        weekly[col] = pd.to_numeric(weekly[col], errors="coerce")
    count = weekly["event_count"].replace(0, np.nan)
    share = (weekly["bullish_event_count"] - weekly["bearish_event_count"]) / count
    out = pd.DataFrame({
        "period_id": weekly["period_id"],
        "weekly_available_date": weekly["weekly_available_date"],
        "weekly_latest_source_date": weekly["weekly_latest_source_date"],
        "news_weekly_event_count_log": np.log1p(weekly["event_count"]),
        "news_weekly_bull_minus_bear_share": share,
        "news_weekly_direction_concentration": share.abs(),
        "news_weekly_explicit_coffee_share": weekly["explicit_coffee_event_count"] / count,
        "news_weekly_mean_event_intensity": weekly["mean_event_intensity_score_0_1"],
    })
    for col, values in text_news_features(weekly["top_event_digest"]).items():
        out[col] = values
    return out.sort_values("weekly_available_date")


def text_news_features(text: pd.Series) -> pd.DataFrame:
    """Simple news-text indicators; not a trained financial sentiment model."""
    terms = {
        "disruption": ["frost", "drought", "flood", "wildfire", "strike", "conflict", "war",
                       "sanctions", "blockade", "disease", "shortage", "export ban"],
        "support": ["bumper", "recovery", "surplus", "ceasefire", "agreement", "reopen"],
        "coffee": ["coffee", "arabica", "cafe", "caffeine", "roaster"],
    }
    text = text.fillna("").str.lower()
    return pd.DataFrame({
        f"news_weekly_text_{name}": np.log1p(text.str.count(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b"))
        for name, words in terms.items()
    }, index=text.index)


def build_final_frame(base_path: Path, weekly_path: Path, daily_path: Path, calendar_path: Path):
    base = pd.read_csv(base_path, parse_dates=["Date"])
    required = {"Date", BASE_PREDICTION}
    if not required.issubset(base):
        raise ValueError(f"Base predictions missing columns: {sorted(required.difference(base))}")
    if base["Date"].duplicated().any() or base["Date"].isna().any():
        raise ValueError("Base prediction dates must be unique and non-null.")
    base = base.sort_values("Date").reset_index(drop=True)
    calendar = load_calendar(calendar_path)
    if not base["Date"].isin(calendar).all():
        raise ValueError("All prediction dates must be in the local price calendar.")
    daily = pd.read_csv(daily_path, usecols=DAILY_COLUMNS)
    weekly = pd.read_csv(weekly_path, usecols=WEEKLY_COLUMNS)
    out = base.merge(daily_news_features(daily, calendar), on="Date", validate="one_to_one")
    out = pd.merge_asof(out, weekly_news_features(weekly, calendar), left_on="Date", right_on="weekly_available_date")
    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan)
    maturity = calendar.get_indexer(out["Date"]) + HORIZON
    out["target_available_date"] = pd.NaT
    valid = maturity < len(calendar)
    out.loc[valid, "target_available_date"] = calendar[maturity[valid]].to_numpy()
    audit = {
        "base_rows": len(base), "daily_rows": len(daily), "weekly_rows": len(weekly),
        "weekly_match_rate": float(out["period_id"].notna().mean()),
        "news_delay_trading_sessions": NEWS_DELAY,
        "feature_missing_rates": out[FEATURES].isna().mean().to_dict(),
        "excluded_inputs": ["all price-response, rule-impact, final-impact, deterministic-period and LLM scores"],
        "daily_source_columns": DAILY_COLUMNS, "weekly_source_columns": WEEKLY_COLUMNS,
        "lookahead_policy": "Use news counts/intensity after source date + 5-session return maturity + 1 session.",
        "base_identity": "Existing Arabica ExtraTrees return predictions; Astra identity unverified.",
    }
    return out, FEATURES.copy(), audit


def eligible_training_rows(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[frame["target_available_date"].lt(cutoff)].copy()


def chronological_split(frame: pd.DataFrame, calibration_size: float):
    if not 0.10 <= calibration_size <= 0.80:
        raise ValueError("--calibration-size must be between 0.10 and 0.80.")
    frame = frame.sort_values("Date").reset_index(drop=True)
    test = frame.iloc[int(len(frame) * calibration_size):].copy()
    if len(test) < 30:
        raise ValueError("At least 30 test rows are required.")
    calibration = eligible_training_rows(frame.loc[frame["Date"] < test["Date"].min()], test["Date"].min())
    if len(calibration) < 30:
        raise ValueError("At least 30 calibration rows are required after purging.")
    return calibration, test


def fit_models(train: pd.DataFrame, news_features: list[str]):
    specs = {
        "calibrated_base": [BASE_PREDICTION], "news_only": news_features,
        "final_news_blend": [BASE_PREDICTION, *news_features],
    }
    models = {}
    for name, features in specs.items():
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()), ("ridge", Ridge(alpha=3.0)),
        ])
        model.fit(train[features], train[TARGET_RETURN])
        models[name] = model
    return models, specs


def attach_predictions(frame: pd.DataFrame, models: dict, specs: dict) -> pd.DataFrame:
    out = frame.copy()
    for name, model in models.items():
        out[PREDICTIONS[name]] = model.predict(out[specs[name]])
    return out


def metric_row(frame: pd.DataFrame, prediction_col: str) -> dict:
    y, pred = frame[TARGET_RETURN].astype(float), frame[prediction_col].astype(float)
    return {
        "rows": len(frame), "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(root_mean_squared_error(y, pred)), "r2": float(r2_score(y, pred)),
        "directional_accuracy": float(accuracy_score(y > 0, pred > 0)),
    }


def all_metrics(frame: pd.DataFrame) -> dict:
    return {name: metric_row(frame, column) for name, column in PREDICTIONS.items()}


def delta_metrics(final: dict, base: dict) -> dict:
    return {f"{key}_delta": final[key] - base[key] for key in ("mae", "rmse", "r2", "directional_accuracy")}


def span(frame: pd.DataFrame) -> dict:
    return {"start": str(frame["Date"].min().date()), "end": str(frame["Date"].max().date()), "rows": len(frame)}


def bootstrap_lift(test: pd.DataFrame, seed: int) -> dict:
    y = test[TARGET_RETURN].to_numpy()
    final, base = test[PREDICTIONS["final_news_blend"]].to_numpy(), test[PREDICTIONS["calibrated_base"]].to_numpy()
    rng = np.random.default_rng(seed)
    block_size, repetitions = 20, 1000
    starts = rng.integers(0, len(test) - block_size + 1, size=(repetitions, int(np.ceil(len(test) / block_size))))
    indices = (starts[:, :, None] + np.arange(block_size)).reshape(repetitions, -1)[:, :len(test)]
    rmse_gain = np.sqrt(((y - base) ** 2)[indices].mean(axis=1)) - np.sqrt(((y - final) ** 2)[indices].mean(axis=1))
    direction_gain = (((y > 0) == (final > 0)).astype(float) - ((y > 0) == (base > 0)))[indices].mean(axis=1)
    return {
        "reference": "calibrated_base", "block_sessions": block_size, "repetitions": repetitions,
        "positive_means_news_improvement": True,
        "rmse_reduction_95pct_interval": np.quantile(rmse_gain, [0.025, 0.975]).tolist(),
        "directional_accuracy_gain_95pct_interval": np.quantile(direction_gain, [0.025, 0.975]).tolist(),
    }


def walk_forward(frame: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    predictions, folds = [], []
    for number, indices in enumerate(np.array_split(np.arange(len(test)), 3), start=1):
        block = test.iloc[indices].copy()
        train = eligible_training_rows(frame, block["Date"].min())
        models, specs = fit_models(train, features)
        scored = attach_predictions(block, models, specs).assign(fold=number)
        predictions.append(scored)
        folds.append({"fold": number, "train": span(train), "test": span(block), "metrics": all_metrics(scored)})
    return pd.concat(predictions, ignore_index=True), folds


def conclusion(metrics: dict) -> str:
    delta = metrics["test_delta_vs_calibrated_base"]
    if delta["rmse_delta"] < 0 and delta["directional_accuracy_delta"] > 0:
        result = "Delayed news improves RMSE and direction accuracy against the calibrated base on this holdout."
    elif delta["rmse_delta"] < 0:
        result = "Delayed news improves RMSE but not direction accuracy against the calibrated base on this holdout."
    elif delta["directional_accuracy_delta"] > 0:
        result = "Delayed news improves direction accuracy but worsens RMSE against the calibrated base on this holdout."
    else:
        result = "Delayed news does not improve RMSE or direction accuracy against the calibrated base on this holdout."
    return result + " This is exploratory evidence from previously selected local news summaries."


def write_plot(metrics: dict, path: Path) -> None:
    labels = ["Base", "Calibrated base", "News only", "Base + news"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, key, title in zip(axes, ["rmse", "directional_accuracy"], ["5-session return RMSE", "Direction accuracy"]):
        values = [metrics["test_metrics"][name][key] for name in PREDICTIONS]
        bars = axis.bar(labels, values, color=["#537895", "#757575", "#cd8b35", "#3a986b"])
        axis.bar_label(bars, fmt="%.4f" if key == "rmse" else "%.3f", padding=3)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        axis.set_ylim(0, max(values) * 1.22 if key == "rmse" else 1)
    fig.suptitle("Corrected evaluation: delayed news, purged split")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(metrics: dict, path: Path) -> None:
    lines = ["# Corrected Local News Evaluation", "", metrics["conclusion"], "",
             "This report supersedes the earlier evaluation and saved model.", "",
             "| Model | RMSE | MAE | Direction accuracy |", "| --- | ---: | ---: | ---: |"]
    for name, row in metrics["test_metrics"].items():
        lines.append(f"| {name} | {row['rmse']:.5f} | {row['mae']:.5f} | {row['directional_accuracy']:.2%} |")
    lines += ["", f"Test: {metrics['split']['test']}. Calibration: {metrics['split']['calibration']}.", "",
              f"Recommendation selected on calibration validation RMSE: `{metrics['selected_model']}`.", "",
              "## Method", "", metrics["method"], "",
              "Daily features average five eligible trading sessions. Weekly features become eligible six "
              "price-calendar sessions after the week's last trading session. Missing news is imputed from "
              "training only. Counts describe retained events, not total news volume.", "",
              "`gdelt_coffee_event_impact.py:score_events` includes future returns in impact scores; "
              "`gdelt_stream_all_years_coffee_events.py:update_day_buffers` selects events using those scores. "
              "`gdelt_weekly_ollama_batch_score.py:add_deterministic_period_score` also uses future returns "
              "and full-sample normalization. These scores are excluded from the corrected features.", "",
              "## Validation", "",
              "All fitted comparisons use the same preprocessing, Ridge alpha=3, training rows and test dates. "
              "Selection uses an inner purged calibration split; no test-driven tuning is performed. "
              "The bundle retains that selection plus the news candidate and base control, fitted on calibration only.", "",
              "| Walk-forward model | RMSE | Direction accuracy |", "| --- | ---: | ---: |"]
    for name, row in metrics["walk_forward"]["aggregate_metrics"].items():
        lines.append(f"| {name} | {row['rmse']:.5f} | {row['directional_accuracy']:.2%} |")
    uncertainty = metrics["uncertainty_vs_calibrated_base"]
    lines += ["", "Three expanding-window folds are pooled above. Paired 20-session moving-block bootstrap "
              "versus calibrated base (1,000 samples; positive means improvement):", "",
              f"- RMSE reduction, 95% interval: {uncertainty['rmse_reduction_95pct_interval']}",
              f"- Direction accuracy gain, 95% interval: {uncertainty['directional_accuracy_gain_95pct_interval']}",
              "", "## Limits", ""]
    lines.extend(f"- {item}" for item in LIMITATIONS)
    lines += ["", "## Reproduce", "", "Run from the repository root:", "", "```bash",
              ".venv/bin/python Scripts/project/scripts/integrate_news_scores_final_model.py", "```", "",
              "No network access or external data is used. The `.joblib` bundle's `model` is the validation-selected "
              "pipeline and `features` is its ordered input list. When `model` is `None`, use the existing base "
              "prediction unchanged. `news_blend_model` always retains the evaluated news candidate. "
              "The saved pipelines reproduce evaluation predictions; they are not refitted on test labels.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    paths = {key: resolve_path(getattr(args, key)) for key in (
        "base_predictions", "weekly_news", "daily_news", "price_calendar", "outputs_dir", "plots_dir", "models_dir",
    )}
    frame, features, audit = build_final_frame(paths["base_predictions"], paths["weekly_news"], paths["daily_news"], paths["price_calendar"])
    frame = frame.loc[np.isfinite(frame[TARGET_RETURN]) & np.isfinite(frame[BASE_PREDICTION]) & frame["target_available_date"].notna()].copy()
    calibration, test = chronological_split(frame, args.calibration_size)
    # Initial base-holdout labels may overlap the base model's training boundary.
    initial_dates = frame["Date"].iloc[:HORIZON]
    calibration = calibration.loc[~calibration["Date"].isin(initial_dates)].copy()
    research_frame = frame.loc[~frame["Date"].isin(initial_dates)].copy()
    inner_train, inner_valid = chronological_split(calibration, 0.70)
    inner_models, specs = fit_models(inner_train, features)
    validation_metrics = all_metrics(attach_predictions(inner_valid, inner_models, specs))
    candidates = ["base_model", "calibrated_base", "final_news_blend"]
    selected = min(candidates, key=lambda name: validation_metrics[name]["rmse"])
    models, specs = fit_models(calibration, features)
    calibration = attach_predictions(calibration, models, specs)
    test = attach_predictions(test, models, specs)
    wf_predictions, folds = walk_forward(research_frame, test, features)
    metrics = {
        "schema_version": 2, "objective": "Check incremental forecasting value of delayed local news summaries.",
        "method": "Price-derived news scores are excluded. All news summaries are delayed six trading sessions "
        "because upstream row selection used five-session future returns. All calibration/validation/test "
        "boundaries purge labels whose maturity is on or after the next period's first date. "
        "Base-only calibration and news blends share a fixed Ridge alpha=3. No cloud scoring or downloads.",
        "inputs": {key: str(value.relative_to(ROOT)) for key, value in paths.items() if not key.endswith("_dir")},
        "audit": audit, "limitations": LIMITATIONS,
        "split": {"calibration": span(calibration), "test": span(test), "initial_base_boundary_rows_excluded": HORIZON,
                  "calibration_last_label_date": str(calibration["target_available_date"].max().date()),
                  "purged_boundary_rows": int(((frame["Date"] < test["Date"].min()) & (frame["target_available_date"] >= test["Date"].min())).sum())},
        "news_features": features, "selected_model": selected,
        "selection": {"criterion": "minimum inner validation RMSE", "train": span(inner_train), "validation": span(inner_valid), "metrics": validation_metrics},
        "calibration_metrics_in_sample": all_metrics(calibration), "test_metrics": all_metrics(test),
        "walk_forward": {"folds": folds, "aggregate_metrics": all_metrics(wf_predictions)},
        "uncertainty_vs_calibrated_base": bootstrap_lift(test, args.random_state),
    }
    for reference in ("base_model", "calibrated_base"):
        metrics[f"test_delta_vs_{'base' if reference == 'base_model' else reference}"] = delta_metrics(metrics["test_metrics"]["final_news_blend"], metrics["test_metrics"][reference])
    metrics["conclusion"] = conclusion(metrics)
    for key in ("outputs_dir", "plots_dir", "models_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    prefix = "final_news_integrated_model"
    predictions = pd.concat([calibration.assign(split="calibration"), test.assign(split="test")], ignore_index=True)
    predictions.to_csv(paths["outputs_dir"] / f"{prefix}_predictions.csv", index=False)
    wf_predictions.to_csv(paths["outputs_dir"] / f"{prefix}_walk_forward_predictions.csv", index=False)
    (paths["outputs_dir"] / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    joblib.dump({
        "schema_version": 2, "model": models.get(selected), "selected_model_name": selected,
        "features": specs.get(selected, [BASE_PREDICTION]), "news_features": features,
        "news_blend_model": models["final_news_blend"], "news_blend_features": specs["final_news_blend"],
        "calibrated_base_model": models["calibrated_base"], "news_only_model": models["news_only"], "metrics": metrics,
    }, paths["models_dir"] / f"{prefix}.joblib")
    write_plot(metrics, paths["plots_dir"] / f"{prefix}_eval.png")
    write_report(metrics, paths["outputs_dir"] / f"{prefix}_report.md")
    print(json.dumps({key: metrics[key] for key in ("split", "test_metrics", "selected_model", "conclusion")}, indent=2))


if __name__ == "__main__":
    main()
