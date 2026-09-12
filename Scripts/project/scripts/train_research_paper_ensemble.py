from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/arabica-futures-matplotlib")

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import binomtest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional at runtime
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "Scripts" / "project" / "src"))

from research_models import (  # noqa: E402
    ExtremeLearningMachineRegressor,
    ThresholdAutoregressiveRegressor,
    fit_arima111,
    fit_garch11,
    fit_holt_winters,
    fit_simple_exp_smoothing,
    rolling_polynomial_return_forecast,
    walk_forward_garch_forecast,
    walk_forward_return_forecast,
)


TARGET_RETURN = "target_return_5d"
TARGET_CLOSE = "target_close_5d"
TARGET_DIRECTION = "target_direction_5d"
TARGET_VOLATILITY = "target_realized_volatility_5d"
CORE_COLUMNS = ["Date", "Close", "High", "Low", "Open", "Volume"]

PAPER_REFERENCES = {
    "elm_coffee_forecasting": {
        "path": "Scripts/researchpapers/pdfs/Amethodologyforcoffeepriceforecastingbasedonextremelearningmachines.pdf",
        "approaches": ["ELM", "MLP", "AR", "ARIMA", "exponential smoothing"],
    },
    "arima_coffee_forecasting": {
        "path": "Scripts/researchpapers/pdfs/An_Algorithm_for_Predicting_Coffee_Prices_Using_AR.pdf",
        "approaches": ["ARIMA(1,1,1)"],
    },
    "coffee_c_ml": {
        "path": "Scripts/researchpapers/pdfs/SDPIT2024-199-209.pdf",
        "approaches": ["linear regression", "random forest", "OHLCV technical features"],
    },
    "volatility_models": {
        "path": "Scripts/researchpapers/pdfs/sustainability-18-05491.pdf",
        "approaches": ["linear trend", "quadratic trend", "Holt-Winters", "ARIMA", "GARCH(1,1)"],
    },
    "hedonic_weather": {
        "path": "Scripts/researchpapers/pdfs/bachelor_Nguyen_Tien_2017.pdf",
        "approaches": ["hedonic regression", "weather", "lagged futures variables"],
    },
    "contract_behavior": {
        "path": "Scripts/researchpapers/pdfs/423528_cr170220rm.pdf",
        "approaches": ["contract behavior", "price shocks", "counterparty relationships"],
    },
    "strategic_default": {
        "path": "Scripts/researchpapers/pdfs/StrategicDefault.pdf",
        "approaches": ["market-price shocks", "commercial pressure", "default-risk proxy"],
    },
    "financialization_thresholds": {
        "path": "Scripts/researchpapers/pdfs/S1703494926000101.html",
        "approaches": ["net positioning index", "positional extremes ratio", "threshold autoregression"],
    },
}


@dataclass
class ModelSpec:
    name: str
    approach: str
    paper_keys: tuple[str, ...]
    kind: str
    feature_group: str
    features: list[str]
    estimator: object | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare paper-inspired Arabica futures models and a final ensemble."
    )
    parser.add_argument(
        "--data",
        default="data/centralData/arabica_ml_model_ready.csv",
        help="Prepared Yahoo, COT, and weather training dataset.",
    )
    parser.add_argument(
        "--events-data",
        default="data/events/gdelt_coffee_events_2000_2026_filtered_scored.csv",
        help="Local GDELT file. Only raw, deployment-safe columns are read.",
    )
    parser.add_argument("--research-dir", default="Scripts/researchpapers/pdfs")
    parser.add_argument("--models-dir", default="Scripts/project/artifacts/models")
    parser.add_argument("--outputs-dir", default="Scripts/project/artifacts/outputs")
    parser.add_argument("--plots-dir", default="Scripts/project/artifacts/plots")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--max-feature-missing", type=float, default=0.20)
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-events", action="store_true")
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fewer trees and two CV folds for a quick smoke run.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    if args.horizon != 5:
        raise ValueError("This prepared dataset contains 5-trading-day targets; use --horizon 5.")
    if not 0 < args.train_size < 1 or not 0 < args.valid_size < 1:
        raise ValueError("Train and validation fractions must be between zero and one.")
    if args.train_size + args.valid_size >= 1:
        raise ValueError("Train and validation fractions must leave a test holdout.")
    if not 0 <= args.max_feature_missing < 1:
        raise ValueError("--max-feature-missing must be in [0, 1).")

    started = time.perf_counter()
    random_state = int(args.random_state)
    sns.set_theme(style="whitegrid", context="notebook")
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    data_path = resolve_path(args.data)
    event_path = resolve_path(args.events_data)
    outputs_dir = resolve_path(args.outputs_dir)
    plots_dir = resolve_path(args.plots_dir)
    models_dir = resolve_path(args.models_dir)
    for directory in (outputs_dir, plots_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    research_manifest = validate_research_files(resolve_path(args.research_dir))
    full_frame, load_audit = load_prepared_data(data_path, args.horizon)
    full_frame = add_paper_positioning_features(full_frame)

    event_features = None
    event_audit = {"enabled": False, "reason": "--skip-events"}
    if not args.skip_events and event_path.exists():
        print("Building leak-safe daily GDELT features from local events...", flush=True)
        event_features, event_audit = build_safe_event_features(
            event_path,
            start_date=full_frame["Date"].min(),
            end_date=full_frame["Date"].max(),
        )
        full_frame = full_frame.merge(event_features, on="Date", how="left", validate="one_to_one")
    elif not args.skip_events:
        event_audit = {"enabled": False, "reason": f"File not found: {event_path}"}

    full_frame = add_local_regime_features(full_frame)

    model_frame = full_frame.loc[full_frame[TARGET_RETURN].notna()].copy().reset_index(drop=True)
    model_frame["_row_id"] = np.arange(len(model_frame), dtype=int)
    split = purged_chronological_split(
        model_frame,
        train_size=args.train_size,
        valid_size=args.valid_size,
        horizon=args.horizon,
    )
    train = split["train"]
    valid = split["valid"]
    validation_context = split["validation_context"]
    dev_train = split["dev_train"]
    test = split["test"]
    test_context = split["test_context"]

    feature_groups, feature_audit = build_feature_groups(
        model_frame,
        train,
        max_missing=args.max_feature_missing,
    )
    specs = build_model_specs(
        feature_groups,
        random_state=random_state,
        include_xgboost=not args.skip_xgboost,
        fast=args.fast,
    )
    leakage_audit = validate_no_feature_leakage(specs, train)

    print(
        f"Rows: train={len(train)}, validation={len(valid)}, test={len(test)}; "
        f"models={len(specs)}; features={len(feature_groups['integrated'])}",
        flush=True,
    )
    print("Evaluating paper-inspired models on validation...", flush=True)
    validation_results = evaluate_specs(
        specs,
        train=train,
        context=validation_context,
        evaluation=valid,
        transaction_cost_bps=args.transaction_cost_bps,
        phase="validation",
    )
    ensemble_members, ensemble_weights = fit_ensemble_weights(
        valid[TARGET_RETURN].to_numpy(dtype=float),
        validation_results["predictions"],
        validation_results["metrics"],
        max_members=6,
    )
    validation_ensemble_pred = blend_predictions(
        validation_results["predictions"], ensemble_members, ensemble_weights
    )
    validation_ensemble_row = metric_row(
        model="research_ensemble",
        approach="Validation-optimized non-negative blend",
        paper_keys=tuple(PAPER_REFERENCES),
        feature_group="mixed",
        phase="validation",
        frame=valid,
        predictions=validation_ensemble_pred,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    print("Refitting base models through validation and evaluating the untouched test set...", flush=True)
    test_results = evaluate_specs(
        specs,
        train=dev_train,
        context=test_context,
        evaluation=test,
        transaction_cost_bps=args.transaction_cost_bps,
        phase="test",
    )
    test_ensemble_pred = blend_predictions(test_results["predictions"], ensemble_members, ensemble_weights)
    test_ensemble_row = metric_row(
        model="research_ensemble",
        approach="Validation-optimized non-negative blend",
        paper_keys=tuple(PAPER_REFERENCES),
        feature_group="mixed",
        phase="test",
        frame=test,
        predictions=test_ensemble_pred,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    comparison = pd.concat(
        [
            validation_results["metrics"],
            pd.DataFrame([validation_ensemble_row]),
            test_results["metrics"],
            pd.DataFrame([test_ensemble_row]),
        ],
        ignore_index=True,
    )

    cv_splits = min(args.cv_splits, 2) if args.fast else args.cv_splits
    print(f"Running {cv_splits}-fold expanding-window checks...", flush=True)
    cv_metrics = expanding_window_cv(
        model_frame=model_frame.iloc[: split["test_start"] - args.horizon].copy(),
        specs=specs,
        horizon=args.horizon,
        n_splits=cv_splits,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    cv_summary = summarize_cv(cv_metrics)

    print("Fitting and checking GARCH(1,1) volatility forecasts...", flush=True)
    volatility = evaluate_volatility_models(
        train=train,
        validation_context=validation_context,
        valid=valid,
        dev_train=dev_train,
        test_context=test_context,
        test=test,
        horizon=args.horizon,
    )

    predictions = make_prediction_frame(test, test_results["predictions"], test_ensemble_pred)
    confusion = make_confusion_table(test, test_results["predictions"], test_ensemble_pred)
    regime_definitions = fit_regime_definitions(train)
    validation_regimes = assign_regimes(valid, regime_definitions)
    test_regimes = assign_regimes(test, regime_definitions)
    regime_columns = [
        *regime_definitions,
        "trend_volatility_regime",
        "calendar_year",
    ]
    regime_metrics = pd.concat(
        [
            evaluate_regime_metrics(
                valid,
                validation_regimes,
                {**validation_results["predictions"], "research_ensemble": validation_ensemble_pred},
                phase="validation",
                regime_columns=regime_columns,
            ),
            evaluate_regime_metrics(
                test,
                test_regimes,
                {**test_results["predictions"], "research_ensemble": test_ensemble_pred},
                phase="test",
                regime_columns=regime_columns,
            ),
        ],
        ignore_index=True,
    )
    regime_assignments = pd.concat(
        [
            predictions.reset_index(drop=True),
            test_regimes.drop(columns="Date").reset_index(drop=True),
        ],
        axis=1,
    )
    regime_summary = summarize_regime_analysis(
        regime_metrics,
        regime_definitions,
        calibration_frame=train,
    )
    uncertainty = block_bootstrap_diagnostics(
        y_true=test[TARGET_RETURN].to_numpy(dtype=float),
        predictions={**test_results["predictions"], "research_ensemble": test_ensemble_pred},
        baseline_name="zero_return",
        random_state=random_state,
    )

    selected_base = str(
        validation_results["metrics"]
        .loc[lambda frame: frame["model"].ne("zero_return")]
        .sort_values(["rmse", "mae", "directional_accuracy"], ascending=[True, True, False])
        .iloc[0]["model"]
    )
    test_table = comparison.loc[comparison["phase"].eq("test")].copy()
    best_holdout_rmse = test_table.sort_values("rmse").iloc[0]
    best_holdout_direction = test_table.sort_values(
        ["directional_accuracy", "roc_auc"], ascending=False
    ).iloc[0]
    production_bundle = fit_production_bundle(
        specs=specs,
        selected_members=ensemble_members,
        labeled_frame=model_frame,
        full_frame=full_frame,
        horizon=args.horizon,
    )

    comparison_path = outputs_dir / "research_paper_model_comparison.csv"
    cv_path = outputs_dir / "research_paper_model_cross_validation.csv"
    cv_summary_path = outputs_dir / "research_paper_model_cv_summary.csv"
    predictions_path = outputs_dir / "research_paper_model_holdout_predictions.csv"
    confusion_path = outputs_dir / "research_paper_model_confusion_matrices.csv"
    volatility_path = outputs_dir / "research_paper_volatility_predictions.csv"
    safe_event_path = outputs_dir / "research_paper_safe_daily_event_features.csv"
    regime_metrics_path = outputs_dir / "research_paper_regime_metrics.csv"
    regime_assignments_path = outputs_dir / "research_paper_holdout_regimes.csv"
    regime_summary_path = outputs_dir / "research_paper_regime_summary.json"
    metrics_path = outputs_dir / "research_paper_ensemble_metrics.json"
    model_path = models_dir / "research_paper_ensemble.joblib"

    comparison.to_csv(comparison_path, index=False)
    cv_metrics.to_csv(cv_path, index=False)
    cv_summary.to_csv(cv_summary_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    confusion.to_csv(confusion_path, index=False)
    volatility["predictions"].to_csv(volatility_path, index=False)
    regime_metrics.to_csv(regime_metrics_path, index=False)
    regime_assignments.to_csv(regime_assignments_path, index=False)
    regime_summary_path.write_text(json.dumps(json_ready(regime_summary), indent=2))
    if event_features is not None:
        event_features.to_csv(safe_event_path, index=False)

    artifacts = {
        "model": portable_path(model_path),
        "metrics": portable_path(metrics_path),
        "comparison": portable_path(comparison_path),
        "cross_validation": portable_path(cv_path),
        "cross_validation_summary": portable_path(cv_summary_path),
        "predictions": portable_path(predictions_path),
        "confusion_matrices": portable_path(confusion_path),
        "volatility_predictions": portable_path(volatility_path),
        "safe_daily_event_features": (
            portable_path(safe_event_path) if event_features is not None else None
        ),
        "regime_metrics": portable_path(regime_metrics_path),
        "holdout_regimes": portable_path(regime_assignments_path),
        "regime_summary": portable_path(regime_summary_path),
        "plots_dir": portable_path(plots_dir),
    }
    metrics_payload = {
        "objective": "Forecast 5-trading-day Arabica Coffee C returns and direction using local paper-inspired approaches.",
        "target": TARGET_RETURN,
        "data": portable_path(data_path),
        "research_sources": research_manifest,
        "paper_method_notes": {
            "npi_proxy": "noncommercial_net divided by Open_Interest_All; the local abstract does not expose the exact paper formula.",
            "per_proxy": "65-trading-day share of observations with absolute 3-year NPI z-score above 1.5; explicitly a proxy.",
            "unavailable_inputs": ["spot coffee price", "USD/BRL exchange rate"],
            "news_policy": "Only raw GDELT event fields delayed by one calendar day are used. Future returns and derived impact scores are excluded.",
        },
        "data_audit": load_audit,
        "event_audit": event_audit,
        "feature_audit": feature_audit,
        "leakage_audit": leakage_audit,
        "split": split_summary(split),
        "models": [spec_to_dict(spec) for spec in specs],
        "model_availability": {
            "xgboost_loaded": XGBRegressor is not None,
            "xgboost_import_error": XGBOOST_IMPORT_ERROR,
            "xgboost_requested": not args.skip_xgboost,
            "xgboost_enabled": XGBRegressor is not None and not args.skip_xgboost,
            "hist_gradient_boosting_included": True,
        },
        "ensemble": {
            "members": ensemble_members,
            "weights": {name: float(weight) for name, weight in zip(ensemble_members, ensemble_weights)},
            "selection_data": "validation only",
            "selected_base_model": selected_base,
        },
        "validation_metrics": comparison.loc[comparison["phase"].eq("validation")].to_dict(orient="records"),
        "test_metrics": comparison.loc[comparison["phase"].eq("test")].to_dict(orient="records"),
        "cross_validation_summary": cv_summary.to_dict(orient="records"),
        "volatility_metrics": volatility["metrics"],
        "regime_analysis": regime_summary,
        "holdout_uncertainty": uncertainty,
        "holdout_summary": {
            "best_rmse_model_post_hoc": str(best_holdout_rmse["model"]),
            "best_rmse": float(best_holdout_rmse["rmse"]),
            "best_direction_model_post_hoc": str(best_holdout_direction["model"]),
            "best_directional_accuracy": float(best_holdout_direction["directional_accuracy"]),
            "best_direction_roc_auc": float(best_holdout_direction["roc_auc"]),
            "ensemble_beats_zero_return_rmse": bool(
                test_ensemble_row["rmse"]
                < float(test_table.loc[test_table["model"].eq("zero_return"), "rmse"].iloc[0])
            ),
            "selection_warning": (
                "Holdout winners are post-hoc diagnostics and were not used to tune models or ensemble weights. "
                "Use expanding-window and block-bootstrap results to judge stability."
            ),
        },
        "artifacts": artifacts,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    metrics_path.write_text(json.dumps(json_ready(metrics_payload), indent=2))

    joblib.dump(
        {
            "objective": metrics_payload["objective"],
            "target": TARGET_RETURN,
            "horizon": args.horizon,
            "ensemble_members": ensemble_members,
            "ensemble_weights": dict(zip(ensemble_members, ensemble_weights)),
            "feature_groups": feature_groups,
            "model_specs": [spec_to_dict(spec) for spec in specs],
            "evaluation_models": test_results["fitted_models"],
            "production": production_bundle,
            "research_sources": research_manifest,
            "metrics_path": str(metrics_path),
            "training_end_date": model_frame["Date"].max().date().isoformat(),
        },
        model_path,
        compress=3,
    )

    write_all_plots(
        comparison=comparison,
        cv_summary=cv_summary,
        test=test,
        predictions=test_results["predictions"],
        ensemble_prediction=test_ensemble_pred,
        selected_base=selected_base,
        volatility_predictions=volatility["predictions"],
        plots_dir=plots_dir,
        horizon=args.horizon,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    write_regime_plots(regime_metrics, comparison, plots_dir)

    print("\nHoldout comparison (sorted by RMSE):")
    display_columns = [
        "model",
        "rmse",
        "mae",
        "r2",
        "directional_accuracy",
        "roc_auc",
        "strategy_sharpe",
        "strategy_max_drawdown",
    ]
    print(
        comparison.loc[comparison["phase"].eq("test"), display_columns]
        .sort_values("rmse")
        .to_string(index=False)
    )
    print("\nEnsemble weights:", dict(zip(ensemble_members, np.round(ensemble_weights, 4))))
    print("Saved model:", model_path)
    print("Saved metrics:", metrics_path)
    print("Saved comparison:", comparison_path)
    print("Saved plots:", plots_dir)


def validate_research_files(research_dir: Path) -> dict:
    manifest = {}
    for key, details in PAPER_REFERENCES.items():
        path = research_dir / Path(details["path"]).name
        manifest[key] = {
            **details,
            "exists": path.exists(),
            "size_bytes": int(path.stat().st_size) if path.exists() else None,
        }
    missing = [details["path"] for details in manifest.values() if not details["exists"]]
    if missing:
        raise FileNotFoundError(f"Local research files are missing: {missing}")
    return manifest


def load_prepared_data(path: Path, horizon: int) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False, parse_dates=["Date"])
    required = set(CORE_COLUMNS + [TARGET_RETURN, TARGET_CLOSE, TARGET_DIRECTION])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prepared data is missing required columns: {missing}")

    original_rows = len(frame)
    frame = frame.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)
    for column in CORE_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid_core_rows = int(frame[CORE_COLUMNS].isna().any(axis=1).sum())
    if invalid_core_rows:
        raise ValueError(f"Prepared data has {invalid_core_rows} rows with missing Date/OHLCV values.")
    if not frame["Date"].is_monotonic_increasing or frame["Date"].duplicated().any():
        raise ValueError("Dates must be sorted and unique.")

    numeric_columns = [column for column in frame.columns if column != "Date"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame[TARGET_DIRECTION] = np.where(
        frame[TARGET_RETURN].notna(), (frame[TARGET_RETURN] > 0).astype(float), np.nan
    )
    frame["log_return_1d"] = np.log(frame["Close"]).diff()
    future_squared_returns = pd.concat(
        [frame["log_return_1d"].shift(-step).pow(2) for step in range(1, horizon + 1)],
        axis=1,
    )
    frame[TARGET_VOLATILITY] = np.sqrt(future_squared_returns.sum(axis=1, min_count=horizon))
    audit = {
        "input_rows": int(original_rows),
        "unique_sorted_rows": int(len(frame)),
        "date_start": frame["Date"].min().date().isoformat(),
        "date_end": frame["Date"].max().date().isoformat(),
        "duplicate_dates_removed": int(original_rows - len(frame)),
        "core_missing_rows": invalid_core_rows,
        "target_rows": int(frame[TARGET_RETURN].notna().sum()),
        "positive_target_rate": float((frame[TARGET_RETURN].dropna() > 0).mean()),
        "numeric_infinite_values_after_cleanup": int(
            np.isinf(frame.select_dtypes(include=np.number).to_numpy(dtype=float)).sum()
        ),
    }
    return frame, audit


def rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    rolling = series.rolling(window, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std().replace(0, np.nan)
    return (series - mean) / std


def add_paper_positioning_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    open_interest = pd.to_numeric(out.get("Open_Interest_All"), errors="coerce").replace(0, np.nan)
    for name in ["commercial_net", "noncommercial_net", "nonreportable_net"]:
        if name in out.columns:
            out[f"{name}_pct_oi"] = pd.to_numeric(out[name], errors="coerce") / open_interest

    if "noncommercial_net" in out.columns:
        net_speculative = pd.to_numeric(out["noncommercial_net"], errors="coerce")
        npi_source = "noncommercial_net"
    elif "commercial_net" in out.columns:
        net_speculative = -pd.to_numeric(out["commercial_net"], errors="coerce")
        npi_source = "negative_commercial_net"
    else:
        net_speculative = pd.Series(np.nan, index=out.index)
        npi_source = "unavailable"

    out["npi_proxy"] = net_speculative / open_interest
    out["npi_z_3y"] = rolling_zscore(out["npi_proxy"], window=756, min_periods=252)
    out["npi_abs_z_3y"] = out["npi_z_3y"].abs()
    out["position_extreme_proxy"] = np.where(
        out["npi_abs_z_3y"].notna(), (out["npi_abs_z_3y"] > 1.5).astype(float), np.nan
    )
    out["per_proxy_13w"] = out["position_extreme_proxy"].rolling(65, min_periods=20).mean()
    out["npi_change_5d"] = out["npi_proxy"].diff(5)

    if "commercial_net_pct_oi" in out.columns and "noncommercial_net_pct_oi" in out.columns:
        out["commercial_vs_speculator_spread_pct_oi"] = (
            out["noncommercial_net_pct_oi"] - out["commercial_net_pct_oi"]
        )
    if "Open_Interest_All" in out.columns:
        out["open_interest_z_1y"] = rolling_zscore(
            pd.to_numeric(out["Open_Interest_All"], errors="coerce"),
            window=252,
            min_periods=80,
        )
    weekly_change_column = next(
        (
            column
            for column in ["noncommercial_weekly_net_change", "commercial_weekly_net_change"]
            if column in out.columns
        ),
        None,
    )
    if weekly_change_column:
        weekly_change = pd.to_numeric(out[weekly_change_column], errors="coerce")
    else:
        weekly_change = out["npi_proxy"].diff(5)
    out["cot_position_reversal_proxy"] = np.where(
        weekly_change.notna() & weekly_change.shift(5).notna(),
        (np.sign(weekly_change) != np.sign(weekly_change.shift(5))).astype(float),
        np.nan,
    )

    volatility = pd.to_numeric(out.get("volatility_60"), errors="coerce").replace(0, np.nan)
    out["positive_price_shock_proxy"] = (
        pd.to_numeric(out.get("return_5d"), errors="coerce") / (volatility * np.sqrt(5.0))
    ).clip(lower=0)
    if "commercial_net_pct_oi" in out.columns:
        commercial_short_pressure = (-out["commercial_net_pct_oi"]).clip(lower=0)
        out["strategic_default_stress_proxy"] = (
            out["positive_price_shock_proxy"] * commercial_short_pressure
        )
    out.attrs["npi_source"] = npi_source
    return out


def add_local_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    def trailing_thresholds(series: pd.Series) -> tuple[pd.Series, pd.Series]:
        history = series.rolling(756, min_periods=252)
        return history.quantile(1.0 / 3.0).shift(1), history.quantile(2.0 / 3.0).shift(1)

    def state_indicators(source: str, prefix: str) -> None:
        if source not in out.columns:
            return
        values = pd.to_numeric(out[source], errors="coerce")
        low, high = trailing_thresholds(values)
        available = values.notna() & low.notna() & high.notna()
        out[f"regime_{prefix}_low"] = np.where(available, values.le(low).astype(float), np.nan)
        out[f"regime_{prefix}_high"] = np.where(available, values.ge(high).astype(float), np.nan)
        historical_mean = values.rolling(756, min_periods=252).mean().shift(1)
        historical_std = values.rolling(756, min_periods=252).std().shift(1).replace(0, np.nan)
        out[f"regime_{prefix}_z_3y"] = (values - historical_mean) / historical_std

    state_indicators("volatility_20", "volatility")
    state_indicators("return_20d", "trend")
    state_indicators("volume_vs_ma_20", "liquidity")

    if {"return_20d", "volatility_20"}.issubset(out.columns):
        out["regime_trend_volatility_interaction"] = (
            pd.to_numeric(out["return_20d"], errors="coerce")
            * pd.to_numeric(out["volatility_20"], errors="coerce")
        )
    if {"npi_abs_z_3y", "volatility_20"}.issubset(out.columns):
        out["regime_cot_crowding_volatility"] = (
            pd.to_numeric(out["npi_abs_z_3y"], errors="coerce")
            * pd.to_numeric(out["volatility_20"], errors="coerce")
        )
    if {"news_supply_shock_proxy", "volatility_20"}.issubset(out.columns):
        out["regime_news_shock_volatility"] = (
            pd.to_numeric(out["news_supply_shock_proxy"], errors="coerce")
            * pd.to_numeric(out["volatility_20"], errors="coerce")
        )

    weather_components = {
        "weather_brazil_minas_gerais_temperature_2m_max_roll20": 1.0,
        "weather_brazil_minas_gerais_vapour_pressure_deficit_max_roll20": 1.0,
        "weather_brazil_minas_gerais_soil_moisture_0_to_100cm_mean_roll20": -1.0,
        "weather_brazil_minas_gerais_precipitation_sum_roll20": -1.0,
    }
    standardized = []
    for column, sign in weather_components.items():
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        mean = values.rolling(756, min_periods=252).mean().shift(1)
        std = values.rolling(756, min_periods=252).std().shift(1).replace(0, np.nan)
        standardized.append((values - mean) / std * sign)
    if standardized:
        out["regime_brazil_weather_stress_z"] = pd.concat(standardized, axis=1).mean(
            axis=1,
            skipna=False,
        )
        stress = out["regime_brazil_weather_stress_z"]
        stress_threshold = stress.rolling(756, min_periods=252).quantile(2.0 / 3.0).shift(1)
        available = stress.notna() & stress_threshold.notna()
        out["regime_brazil_weather_stress_high"] = np.where(
            available,
            stress.ge(stress_threshold).astype(float),
            np.nan,
        )
    return out


def build_safe_event_features(
    path: Path,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    use_columns = [
        "GLOBALEVENTID",
        "event_date",
        "IsRootEvent",
        "QuadClass",
        "GoldsteinScale",
        "NumMentions",
        "NumSources",
        "NumArticles",
        "AvgTone",
        "primary_country",
    ]
    source_stat_before = path.stat()
    events = pd.read_csv(path, usecols=use_columns, low_memory=False)
    source_stat_after = path.stat()
    source_changed_during_read = (
        source_stat_before.st_size != source_stat_after.st_size
        or source_stat_before.st_mtime_ns != source_stat_after.st_mtime_ns
    )
    if source_changed_during_read:
        raise RuntimeError(
            "The GDELT source changed while it was being read. Wait for ingestion to finish and rerun."
        )
    original_rows = len(events)
    events = events.drop_duplicates("GLOBALEVENTID", keep="first").copy()
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce").dt.normalize()
    events = events.loc[events["event_date"].notna()].copy()
    events["available_date"] = events["event_date"] + pd.Timedelta(days=1)

    for column in [
        "IsRootEvent",
        "QuadClass",
        "GoldsteinScale",
        "NumMentions",
        "NumSources",
        "NumArticles",
        "AvgTone",
    ]:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    events["_event"] = 1.0
    events["_negative_goldstein"] = events["GoldsteinScale"].lt(0).astype(float)
    events["_material_conflict"] = events["QuadClass"].eq(4).astype(float)
    events["_verbal_conflict"] = events["QuadClass"].eq(3).astype(float)
    events["_brazil"] = events["primary_country"].eq("BR").astype(float)
    events["_colombia"] = events["primary_country"].eq("CO").astype(float)
    events["_vietnam"] = events["primary_country"].isin(["VM", "VN"]).astype(float)
    events["_tone_weight"] = events["NumArticles"].fillna(1.0).clip(lower=1.0)
    events["_tone_weighted"] = events["AvgTone"].fillna(0.0) * events["_tone_weight"]
    events["_tone_observed_weight"] = events["_tone_weight"].where(events["AvgTone"].notna(), 0.0)
    events["_goldstein_weighted"] = events["GoldsteinScale"].fillna(0.0) * events["_tone_weight"]
    events["_goldstein_observed_weight"] = events["_tone_weight"].where(
        events["GoldsteinScale"].notna(), 0.0
    )

    daily = events.groupby("available_date", sort=True).agg(
        news_event_count=("_event", "sum"),
        news_root_event_count=("IsRootEvent", "sum"),
        news_mentions_sum=("NumMentions", "sum"),
        news_sources_sum=("NumSources", "sum"),
        news_articles_sum=("NumArticles", "sum"),
        news_negative_goldstein_count=("_negative_goldstein", "sum"),
        news_material_conflict_count=("_material_conflict", "sum"),
        news_verbal_conflict_count=("_verbal_conflict", "sum"),
        news_brazil_count=("_brazil", "sum"),
        news_colombia_count=("_colombia", "sum"),
        news_vietnam_count=("_vietnam", "sum"),
        news_tone_weighted_sum=("_tone_weighted", "sum"),
        news_tone_observed_weight=("_tone_observed_weight", "sum"),
        news_goldstein_weighted_sum=("_goldstein_weighted", "sum"),
        news_goldstein_observed_weight=("_goldstein_observed_weight", "sum"),
    )
    coverage_start = daily.index.min()
    coverage_end = daily.index.max()
    calendar = pd.date_range(pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize(), freq="D")
    daily = daily.reindex(calendar)
    within_coverage = (daily.index >= coverage_start) & (daily.index <= coverage_end)
    additive_columns = list(daily.columns)
    daily.loc[within_coverage, additive_columns] = daily.loc[within_coverage, additive_columns].fillna(0.0)

    for window in [7, 30]:
        suffix = f"_{window}d"
        count = daily["news_event_count"].rolling(window, min_periods=1).sum()
        daily[f"news_event_count{suffix}"] = count
        daily[f"news_mentions{suffix}"] = daily["news_mentions_sum"].rolling(window, min_periods=1).sum()
        daily[f"news_articles{suffix}"] = daily["news_articles_sum"].rolling(window, min_periods=1).sum()
        tone_weight = daily["news_tone_observed_weight"].rolling(window, min_periods=1).sum()
        goldstein_weight = daily["news_goldstein_observed_weight"].rolling(window, min_periods=1).sum()
        daily[f"news_avg_tone{suffix}"] = (
            daily["news_tone_weighted_sum"].rolling(window, min_periods=1).sum()
            / tone_weight.replace(0, np.nan)
        )
        daily[f"news_avg_goldstein{suffix}"] = (
            daily["news_goldstein_weighted_sum"].rolling(window, min_periods=1).sum()
            / goldstein_weight.replace(0, np.nan)
        )
        for source, output in [
            ("news_root_event_count", "news_root_share"),
            ("news_negative_goldstein_count", "news_negative_goldstein_share"),
            ("news_material_conflict_count", "news_material_conflict_share"),
            ("news_verbal_conflict_count", "news_verbal_conflict_share"),
            ("news_brazil_count", "news_brazil_share"),
            ("news_colombia_count", "news_colombia_share"),
            ("news_vietnam_count", "news_vietnam_share"),
        ]:
            daily[f"{output}{suffix}"] = (
                daily[source].rolling(window, min_periods=1).sum() / count.replace(0, np.nan)
            )

    log_event_count = np.log1p(daily["news_event_count_7d"])
    baseline = log_event_count.rolling(365, min_periods=90)
    daily["news_event_volume_z_365d"] = (
        log_event_count - baseline.mean().shift(1)
    ) / baseline.std().shift(1).replace(0, np.nan)
    daily["news_tone_change_7d"] = daily["news_avg_tone_7d"] - daily["news_avg_tone_7d"].shift(7)
    daily["news_supply_shock_proxy"] = (
        daily["news_material_conflict_share_7d"].fillna(0.0) * log_event_count.fillna(0.0)
        + (-daily["news_avg_tone_7d"].clip(upper=0.0)).fillna(0.0) / 10.0
    )
    derived_columns = [
        column
        for column in daily.columns
        if column.endswith(("_7d", "_30d", "_365d"))
        or column in {"news_tone_change_7d", "news_supply_shock_proxy"}
    ]
    daily.loc[~within_coverage, derived_columns] = np.nan
    daily["news_data_available"] = within_coverage.astype(float)
    output_columns = ["news_data_available", *derived_columns]
    output = daily[output_columns].reset_index().rename(columns={"index": "Date"})
    audit = {
        "enabled": True,
        "source": portable_path(path),
        "source_rows": int(original_rows),
        "source_size_bytes": int(source_stat_after.st_size),
        "source_modified_ns": int(source_stat_after.st_mtime_ns),
        "source_changed_during_read": source_changed_during_read,
        "unique_event_rows": int(len(events)),
        "duplicate_event_ids_removed": int(original_rows - len(events)),
        "event_date_start": events["event_date"].min().date().isoformat(),
        "event_date_end": events["event_date"].max().date().isoformat(),
        "availability_lag": "one calendar day",
        "feature_count": int(len(output_columns)),
        "safe_source_columns": use_columns,
        "explicitly_excluded_leaky_columns": [
            "future_return_1d",
            "future_return_5d",
            "future_return_10d",
            "future_return_20d",
            "price_response_score_0_1",
            "direction_alignment_score_0_1",
            "rule_impact_score_0_1",
            "final_impact_score_0_1",
        ],
    }
    return output, audit


def purged_chronological_split(
    frame: pd.DataFrame,
    train_size: float,
    valid_size: float,
    horizon: int,
) -> dict:
    rows = len(frame)
    train_boundary = int(rows * train_size)
    test_start = int(rows * (train_size + valid_size))
    if train_boundary <= horizon or test_start - train_boundary <= horizon:
        raise ValueError("Dataset is too small for the requested purged split.")
    train = frame.iloc[: train_boundary - horizon].copy()
    validation_context = frame.iloc[train_boundary - horizon : train_boundary].copy()
    valid = frame.iloc[train_boundary : test_start - horizon].copy()
    dev_train = frame.iloc[: test_start - horizon].copy()
    test_context = frame.iloc[test_start - horizon : test_start].copy()
    test = frame.iloc[test_start:].copy()
    if int(train["_row_id"].max()) + horizon >= int(valid["_row_id"].min()):
        raise AssertionError("Training targets cross the validation boundary.")
    if int(dev_train["_row_id"].max()) + horizon >= int(test["_row_id"].min()):
        raise AssertionError("Development targets cross the test boundary.")
    return {
        "train": train,
        "validation_context": validation_context,
        "valid": valid,
        "dev_train": dev_train,
        "test_context": test_context,
        "test": test,
        "train_boundary": train_boundary,
        "test_start": test_start,
        "purge_rows": horizon,
    }


def split_summary(split: dict) -> dict:
    return {
        "method": "chronological train/validation/test with a 5-row target purge before validation and test",
        "train": date_span(split["train"]),
        "validation_context_purge": date_span(split["validation_context"]),
        "validation": date_span(split["valid"]),
        "development_train_for_test": date_span(split["dev_train"]),
        "test_context_purge": date_span(split["test_context"]),
        "test": date_span(split["test"]),
    }


def date_span(frame: pd.DataFrame) -> dict:
    return {
        "start": frame["Date"].min().date().isoformat(),
        "end": frame["Date"].max().date().isoformat(),
        "rows": int(len(frame)),
    }


def is_cot_feature(column: str) -> bool:
    lowered = column.lower()
    tokens = [
        "open_interest",
        "position",
        "pct_of_oi",
        "noncomm",
        "comm_",
        "commercial",
        "trader",
        "conc_",
        "cot_",
        "npi_",
        "per_proxy",
        "speculator",
        "strategic_default",
    ]
    return any(token in lowered for token in tokens)


def unique_existing(columns: list[str], frame: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(column for column in columns if column in frame.columns))


def build_feature_groups(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    max_missing: float,
) -> tuple[dict[str, list[str]], dict]:
    target_columns = {column for column in frame.columns if column.startswith("target_")}
    excluded = target_columns | {"Date", "_row_id"}
    numeric_candidates = [
        column
        for column in frame.select_dtypes(include=np.number).columns
        if column not in excluded
    ]
    history_features = [
        column
        for column in numeric_candidates
        if column.startswith(
            (
                "return_",
                "log_return_",
                "range_pct",
                "close_to_open_pct",
                "volume_change_pct",
                "close_vs_ma_",
                "volatility_",
                "volume_vs_ma_",
            )
        )
    ]
    calendar_features = unique_existing(
        ["month", "quarter", "day_of_week", "month_sin", "month_cos"], frame
    )
    technical = unique_existing(
        ["Close", "High", "Low", "Open", "Volume", *history_features, *calendar_features],
        frame,
    )
    weather = [column for column in numeric_candidates if column.startswith("weather_")]
    news = [column for column in numeric_candidates if column.startswith("news_")]
    regime = [column for column in numeric_candidates if column.startswith("regime_")]
    cot = [
        column
        for column in numeric_candidates
        if is_cot_feature(column) and not column.startswith("regime_")
    ]
    cot_stress = unique_existing(
        [
            "npi_proxy",
            "npi_z_3y",
            "npi_abs_z_3y",
            "position_extreme_proxy",
            "per_proxy_13w",
            "npi_change_5d",
            "commercial_net_pct_oi",
            "noncommercial_net_pct_oi",
            "commercial_vs_speculator_spread_pct_oi",
            "open_interest_z_1y",
            "cot_position_reversal_proxy",
            "positive_price_shock_proxy",
            "strategic_default_stress_proxy",
            "news_supply_shock_proxy",
            "news_event_volume_z_365d",
            "news_negative_goldstein_share_7d",
        ],
        frame,
    )
    groups = {
        "price_history": unique_existing([*history_features, *calendar_features], frame),
        "technical_ohlcv": technical,
        "hedonic_weather_cot": unique_existing([*history_features, *calendar_features, *cot, *weather], frame),
        "cot_contract_stress": unique_existing([*history_features, *cot_stress], frame),
        "threshold_ar": unique_existing(
            [
                "npi_abs_z_3y",
                "return_1d",
                "return_2d",
                "return_5d",
                "return_10d",
                "return_20d",
                "volatility_20",
                "per_proxy_13w",
            ],
            frame,
        ),
        "integrated": unique_existing([*technical, *cot, *weather, *news], frame),
    }
    if regime:
        groups["regime_aware_integrated"] = unique_existing(
            [*technical, *cot, *weather, *news, *regime], frame
        )

    dropped = {}
    for group_name, columns in groups.items():
        kept = []
        reasons = {}
        for column in columns:
            missing_rate = float(train[column].isna().mean())
            unique_count = int(train[column].nunique(dropna=True))
            if missing_rate > max_missing:
                reasons[column] = f"missing_rate={missing_rate:.6f}"
            elif unique_count <= 1:
                reasons[column] = f"unique_count={unique_count}"
            else:
                kept.append(column)
        if not kept:
            raise ValueError(f"No usable features remain for group {group_name}.")
        groups[group_name] = kept
        dropped[group_name] = reasons
    if groups["threshold_ar"][0] != "npi_abs_z_3y":
        raise ValueError("Threshold AR requires npi_abs_z_3y as its first feature.")

    audit = {
        "max_training_missing_rate": float(max_missing),
        "group_counts": {name: len(columns) for name, columns in groups.items()},
        "dropped_by_group": dropped,
        "weather_feature_count": len(weather),
        "cot_feature_count": len(cot),
        "news_feature_count": len(news),
        "regime_feature_count": len(regime),
    }
    return groups, audit


def make_pipeline(estimator, scale: bool) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def build_model_specs(
    feature_groups: dict[str, list[str]],
    random_state: int,
    include_xgboost: bool,
    fast: bool,
) -> list[ModelSpec]:
    tree_count = 80 if fast else 220
    mlp_iterations = 100 if fast else 260
    specs = [
        ModelSpec(
            "zero_return",
            "No-change return benchmark",
            ("arima_coffee_forecasting",),
            "zero",
            "none",
            [],
        ),
        ModelSpec(
            "ar_ridge",
            "Autoregressive return benchmark",
            ("elm_coffee_forecasting", "arima_coffee_forecasting"),
            "sklearn",
            "price_history",
            feature_groups["price_history"],
            make_pipeline(Ridge(alpha=2.0), scale=True),
        ),
        ModelSpec(
            "arima_111",
            "ARIMA(1,1,1) on log price with walk-forward state updates",
            ("elm_coffee_forecasting", "arima_coffee_forecasting", "volatility_models"),
            "arima_111",
            "close_only",
            [],
        ),
        ModelSpec(
            "simple_exp_smoothing",
            "Simple exponential smoothing on log price",
            ("elm_coffee_forecasting",),
            "simple_exp_smoothing",
            "close_only",
            [],
        ),
        ModelSpec(
            "holt_winters_5d",
            "Additive Holt-Winters with five-trading-day seasonality",
            ("elm_coffee_forecasting", "volatility_models"),
            "holt_winters",
            "close_only",
            [],
        ),
        ModelSpec(
            "linear_trend_252d",
            "Rolling linear trend on log price",
            ("volatility_models",),
            "linear_trend",
            "close_only",
            [],
        ),
        ModelSpec(
            "quadratic_trend_252d",
            "Rolling quadratic trend on log price",
            ("volatility_models",),
            "quadratic_trend",
            "close_only",
            [],
        ),
        ModelSpec(
            "linear_technical",
            "Linear regression using OHLCV and technical features",
            ("coffee_c_ml",),
            "sklearn",
            "technical_ohlcv",
            feature_groups["technical_ohlcv"],
            make_pipeline(LinearRegression(), scale=True),
        ),
        ModelSpec(
            "random_forest_technical",
            "Random Forest using OHLCV and technical features",
            ("coffee_c_ml",),
            "sklearn",
            "technical_ohlcv",
            feature_groups["technical_ohlcv"],
            make_pipeline(
                RandomForestRegressor(
                    n_estimators=tree_count,
                    max_depth=10,
                    min_samples_leaf=10,
                    max_features=0.75,
                    random_state=random_state,
                    n_jobs=1,
                ),
                scale=False,
            ),
        ),
        ModelSpec(
            "mlp_price_history",
            "Multilayer perceptron using lagged price-history features",
            ("elm_coffee_forecasting",),
            "sklearn",
            "price_history",
            feature_groups["price_history"],
            make_pipeline(
                MLPRegressor(
                    hidden_layer_sizes=(48, 24),
                    alpha=0.01,
                    learning_rate_init=0.001,
                    max_iter=mlp_iterations,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=18,
                    random_state=random_state,
                ),
                scale=True,
            ),
        ),
        ModelSpec(
            "elm_price_history",
            "Extreme Learning Machine using lagged price-history features",
            ("elm_coffee_forecasting",),
            "sklearn",
            "price_history",
            feature_groups["price_history"],
            make_pipeline(
                ExtremeLearningMachineRegressor(
                    n_hidden=160 if not fast else 80,
                    alpha=8.0,
                    random_state=random_state,
                ),
                scale=True,
            ),
        ),
        ModelSpec(
            "hedonic_weather_cot_ridge",
            "Regularized hedonic/econometric model with weather and released COT",
            ("hedonic_weather",),
            "sklearn",
            "hedonic_weather_cot",
            feature_groups["hedonic_weather_cot"],
            make_pipeline(Ridge(alpha=25.0), scale=True),
        ),
        ModelSpec(
            "threshold_ar_npi",
            "Threshold autoregression using the NPI/PER proxy regime",
            ("financialization_thresholds",),
            "sklearn",
            "threshold_ar",
            feature_groups["threshold_ar"],
            make_pipeline(
                ThresholdAutoregressiveRegressor(alpha=4.0, min_regime_size=120),
                scale=True,
            ),
        ),
        ModelSpec(
            "cot_contract_stress_forest",
            "COT crowding, reversal, contract-shock, and default-stress proxy model",
            ("contract_behavior", "strategic_default", "financialization_thresholds"),
            "sklearn",
            "cot_contract_stress",
            feature_groups["cot_contract_stress"],
            make_pipeline(
                RandomForestRegressor(
                    n_estimators=tree_count,
                    max_depth=8,
                    min_samples_leaf=14,
                    max_features=0.8,
                    random_state=random_state + 7,
                    n_jobs=1,
                ),
                scale=False,
            ),
        ),
        ModelSpec(
            "hist_gradient_boosting_integrated",
            "Integrated histogram gradient boosting using price, COT, weather, and safe news features",
            tuple(PAPER_REFERENCES),
            "sklearn",
            "integrated",
            feature_groups["integrated"],
            make_pipeline(
                HistGradientBoostingRegressor(
                    learning_rate=0.04,
                    max_iter=100 if fast else 240,
                    max_leaf_nodes=15,
                    min_samples_leaf=25,
                    l2_regularization=1.0,
                    random_state=random_state,
                ),
                scale=False,
            ),
        ),
    ]
    if include_xgboost and XGBRegressor is not None:
        specs.append(
            ModelSpec(
                "xgboost_integrated",
                "Integrated nonlinear model using price, COT, weather, and safe news features",
                tuple(PAPER_REFERENCES),
                "sklearn",
                "integrated",
                feature_groups["integrated"],
                make_pipeline(
                    XGBRegressor(
                        n_estimators=100 if fast else 280,
                        max_depth=3,
                        learning_rate=0.03,
                        min_child_weight=12,
                        subsample=0.80,
                        colsample_bytree=0.70,
                        reg_alpha=0.05,
                        reg_lambda=2.0,
                        objective="reg:squarederror",
                        random_state=random_state,
                        n_jobs=1,
                        tree_method="hist",
                    ),
                    scale=False,
                ),
            )
        )
        if "regime_aware_integrated" in feature_groups:
            specs.append(
                ModelSpec(
                    "xgboost_regime_aware",
                    "XGBoost with past-only volatility, trend, liquidity, COT, weather, and news regime interactions",
                    tuple(PAPER_REFERENCES),
                    "sklearn",
                    "regime_aware_integrated",
                    feature_groups["regime_aware_integrated"],
                    make_pipeline(
                        XGBRegressor(
                            n_estimators=100 if fast else 280,
                            max_depth=3,
                            learning_rate=0.03,
                            min_child_weight=12,
                            subsample=0.80,
                            colsample_bytree=0.70,
                            reg_alpha=0.05,
                            reg_lambda=2.0,
                            objective="reg:squarederror",
                            random_state=random_state,
                            n_jobs=1,
                            tree_method="hist",
                        ),
                        scale=False,
                    ),
                )
            )
    return specs


def validate_no_feature_leakage(specs: list[ModelSpec], train: pd.DataFrame) -> dict:
    forbidden_tokens = [
        "target_",
        "future_",
        "price_response",
        "direction_alignment",
        "rule_impact",
        "final_impact",
        "ollama",
    ]
    violations = {}
    all_features = sorted({feature for spec in specs for feature in spec.features})
    for spec in specs:
        bad = [
            feature
            for feature in spec.features
            if any(token in feature.lower() for token in forbidden_tokens)
        ]
        if bad:
            violations[spec.name] = bad
    if violations:
        raise ValueError(f"Feature leakage detected: {violations}")
    correlations = (
        train[all_features]
        .corrwith(train[TARGET_RETURN])
        .abs()
        .sort_values(ascending=False)
        .dropna()
    )
    return {
        "status": "passed",
        "forbidden_feature_violations": violations,
        "feature_count_checked": len(all_features),
        "maximum_absolute_training_target_correlation": float(correlations.iloc[0]),
        "top_absolute_training_target_correlations": {
            str(key): float(value) for key, value in correlations.head(10).items()
        },
        "cot_timing": "Prepared input was built from COT release dates with a three-calendar-day release lag.",
        "split_purge": "Five rows are purged before validation and test so 5D targets do not cross boundaries.",
    }


def fit_predict_spec(
    spec: ModelSpec,
    train: pd.DataFrame,
    context: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> tuple[object | None, np.ndarray]:
    if spec.kind == "zero":
        return None, np.zeros(len(evaluation), dtype=float)
    if spec.kind == "sklearn":
        model = clone(spec.estimator)
        model.fit(train[spec.features], train[TARGET_RETURN].astype(float))
        return model, np.asarray(model.predict(evaluation[spec.features]), dtype=float)
    if spec.kind in {"arima_111", "simple_exp_smoothing", "holt_winters"}:
        return walk_forward_return_forecast(
            spec.kind,
            train["Close"].to_numpy(dtype=float),
            context["Close"].to_numpy(dtype=float),
            evaluation["Close"].to_numpy(dtype=float),
            horizon=5,
        )
    if spec.kind in {"linear_trend", "quadratic_trend"}:
        degree = 1 if spec.kind == "linear_trend" else 2
        history = pd.concat([train["Close"], context["Close"]], ignore_index=True)
        prediction = rolling_polynomial_return_forecast(
            history,
            evaluation["Close"].to_numpy(dtype=float),
            horizon=5,
            degree=degree,
            window=252,
        )
        return {"degree": degree, "window": 252}, prediction
    raise ValueError(f"Unsupported model kind: {spec.kind}")


def evaluate_specs(
    specs: list[ModelSpec],
    train: pd.DataFrame,
    context: pd.DataFrame,
    evaluation: pd.DataFrame,
    transaction_cost_bps: float,
    phase: str,
) -> dict:
    rows = []
    predictions = {}
    fitted_models = {}
    for index, spec in enumerate(specs, start=1):
        model_started = time.perf_counter()
        fitted, prediction = fit_predict_spec(spec, train, context, evaluation)
        if len(prediction) != len(evaluation) or not np.isfinite(prediction).all():
            raise ValueError(f"{spec.name} produced invalid predictions.")
        predictions[spec.name] = prediction
        fitted_models[spec.name] = fitted
        row = metric_row(
            model=spec.name,
            approach=spec.approach,
            paper_keys=spec.paper_keys,
            feature_group=spec.feature_group,
            phase=phase,
            frame=evaluation,
            predictions=prediction,
            transaction_cost_bps=transaction_cost_bps,
        )
        row["fit_predict_seconds"] = float(time.perf_counter() - model_started)
        rows.append(row)
        print(
            f"  [{index:02d}/{len(specs):02d}] {spec.name}: "
            f"RMSE={row['rmse']:.6f}, direction={row['directional_accuracy']:.3f}",
            flush=True,
        )
    return {
        "metrics": pd.DataFrame(rows),
        "predictions": predictions,
        "fitted_models": fitted_models,
    }


def fit_ensemble_weights(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    validation_metrics: pd.DataFrame,
    max_members: int,
) -> tuple[list[str], np.ndarray]:
    candidates = (
        validation_metrics.loc[validation_metrics["model"].ne("zero_return")]
        .sort_values(["rmse", "mae", "directional_accuracy"], ascending=[True, True, False])
        .head(max_members)["model"]
        .tolist()
    )
    if len(candidates) < 2:
        raise ValueError("At least two non-baseline models are required for an ensemble.")
    matrix = np.column_stack([predictions[name] for name in candidates])
    initial = np.full(len(candidates), 1.0 / len(candidates))

    def objective(weights: np.ndarray) -> float:
        error = matrix @ weights - y_true
        return float(np.mean(np.square(error)) + 1e-6 * np.sum(np.square(weights)))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(candidates),
        constraints=[{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    weights = result.x if result.success else initial
    weights = np.clip(weights, 0.0, None)
    weights = weights / weights.sum()
    active = weights > 1e-5
    return [name for name, keep in zip(candidates, active) if keep], weights[active] / weights[active].sum()


def blend_predictions(
    predictions: dict[str, np.ndarray],
    members: list[str],
    weights: np.ndarray,
) -> np.ndarray:
    matrix = np.column_stack([predictions[name] for name in members])
    return np.asarray(matrix @ weights, dtype=float)


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y_true
    rmse = float(np.sqrt(mean_squared_error(y_true, prediction)))
    target_std = float(np.std(y_true, ddof=1))
    denominator = np.abs(y_true) + np.abs(prediction) + 1e-8
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, prediction)),
        "median_absolute_error": float(median_absolute_error(y_true, prediction)),
        "mean_error_bias": float(np.mean(error)),
        "nrmse_target_std": float(rmse / target_std) if target_std > 0 else np.nan,
        "smape": float(np.mean(2.0 * np.abs(error) / denominator)),
        "r2": float(r2_score(y_true, prediction)),
        "pearson_correlation": safe_correlation(y_true, prediction),
    }


def direction_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    actual_direction = (y_true > 0).astype(int)
    predicted_direction = (prediction > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual_direction, predicted_direction, labels=[0, 1]).ravel()
    try:
        roc_auc = float(roc_auc_score(actual_direction, prediction))
        average_precision = float(average_precision_score(actual_direction, prediction))
    except ValueError:
        roc_auc = np.nan
        average_precision = np.nan
    accuracy = float(accuracy_score(actual_direction, predicted_direction))
    return {
        "directional_accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy_score(actual_direction, predicted_direction)),
        "precision_positive": float(
            precision_score(actual_direction, predicted_direction, zero_division=0)
        ),
        "recall_positive": float(recall_score(actual_direction, predicted_direction, zero_division=0)),
        "f1_positive": float(f1_score(actual_direction, predicted_direction, zero_division=0)),
        "specificity_negative": float(tn / (tn + fp)) if tn + fp else np.nan,
        "matthews_correlation": float(matthews_corrcoef(actual_direction, predicted_direction)),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "naive_binomial_accuracy_p_value_vs_50pct": float(
            binomtest(int((actual_direction == predicted_direction).sum()), len(y_true), 0.5, alternative="greater").pvalue
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def strategy_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    horizon: int,
    transaction_cost_bps: float,
) -> dict[str, float | int]:
    sample_indices = np.arange(0, len(y_true), horizon)
    actual = y_true[sample_indices]
    signal = np.sign(prediction[sample_indices])
    previous_signal = np.r_[0.0, signal[:-1]]
    turnover = np.abs(signal - previous_signal)
    costs = turnover * transaction_cost_bps / 10_000.0
    strategy_returns = signal * actual - costs
    periods_per_year = 252.0 / horizon
    standard_deviation = float(np.std(strategy_returns, ddof=1))
    sharpe = (
        float(np.mean(strategy_returns) / standard_deviation * np.sqrt(periods_per_year))
        if standard_deviation > 0
        else np.nan
    )
    wealth = np.cumprod(1.0 + strategy_returns)
    running_peak = np.maximum.accumulate(wealth)
    drawdown = wealth / running_peak - 1.0
    annualized_return = (
        float(wealth[-1] ** (periods_per_year / len(strategy_returns)) - 1.0)
        if len(strategy_returns) and wealth[-1] > 0
        else np.nan
    )
    return {
        "strategy_observations_non_overlapping": int(len(strategy_returns)),
        "strategy_cumulative_return": float(wealth[-1] - 1.0),
        "strategy_annualized_return": annualized_return,
        "strategy_annualized_volatility": float(standard_deviation * np.sqrt(periods_per_year)),
        "strategy_sharpe": sharpe,
        "strategy_max_drawdown": float(drawdown.min()),
        "strategy_win_rate": float((strategy_returns > 0).mean()),
        "strategy_average_turnover": float(np.mean(turnover)),
        "transaction_cost_bps": float(transaction_cost_bps),
    }


def residual_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = y_true - prediction
    return {
        "residual_autocorrelation_lag1": lag_correlation(residual, 1),
        "residual_autocorrelation_lag5": lag_correlation(residual, 5),
        "residual_autocorrelation_lag20": lag_correlation(residual, 20),
        "residual_std": float(np.std(residual, ddof=1)),
    }


def metric_row(
    model: str,
    approach: str,
    paper_keys: tuple[str, ...],
    feature_group: str,
    phase: str,
    frame: pd.DataFrame,
    predictions: np.ndarray,
    transaction_cost_bps: float,
) -> dict:
    y_true = frame[TARGET_RETURN].to_numpy(dtype=float)
    predicted_close = frame["Close"].to_numpy(dtype=float) * (1.0 + predictions)
    price_error = predicted_close - frame[TARGET_CLOSE].to_numpy(dtype=float)
    return {
        "model": model,
        "approach": approach,
        "paper_keys": ",".join(paper_keys),
        "feature_group": feature_group,
        "phase": phase,
        "rows": int(len(frame)),
        "date_start": frame["Date"].min().date().isoformat(),
        "date_end": frame["Date"].max().date().isoformat(),
        **regression_metrics(y_true, predictions),
        "price_rmse": float(np.sqrt(np.mean(np.square(price_error)))),
        "price_mae": float(np.mean(np.abs(price_error))),
        **direction_metrics(y_true, predictions),
        **strategy_metrics(y_true, predictions, horizon=5, transaction_cost_bps=transaction_cost_bps),
        **residual_metrics(y_true, predictions),
    }


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def lag_correlation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return np.nan
    return safe_correlation(values[lag:], values[:-lag])


def expanding_window_cv(
    model_frame: pd.DataFrame,
    specs: list[ModelSpec],
    horizon: int,
    n_splits: int,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    if n_splits < 2:
        return pd.DataFrame()
    test_size = max(126, len(model_frame) // (n_splits + 3))
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=horizon)
    rows = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(model_frame), start=1):
        fold_train = model_frame.iloc[train_indices].copy()
        fold_test = model_frame.iloc[test_indices].copy()
        context = model_frame.iloc[train_indices[-1] + 1 : test_indices[0]].copy()
        for spec in specs:
            _, prediction = fit_predict_spec(spec, fold_train, context, fold_test)
            metrics = regression_metrics(fold_test[TARGET_RETURN].to_numpy(dtype=float), prediction)
            directions = direction_metrics(fold_test[TARGET_RETURN].to_numpy(dtype=float), prediction)
            strategy = strategy_metrics(
                fold_test[TARGET_RETURN].to_numpy(dtype=float),
                prediction,
                horizon=horizon,
                transaction_cost_bps=transaction_cost_bps,
            )
            rows.append(
                {
                    "fold": fold,
                    "model": spec.name,
                    "train_start": fold_train["Date"].min().date().isoformat(),
                    "train_end": fold_train["Date"].max().date().isoformat(),
                    "test_start": fold_test["Date"].min().date().isoformat(),
                    "test_end": fold_test["Date"].max().date().isoformat(),
                    "train_rows": int(len(fold_train)),
                    "test_rows": int(len(fold_test)),
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "directional_accuracy": directions["directional_accuracy"],
                    "roc_auc": directions["roc_auc"],
                    "strategy_sharpe": strategy["strategy_sharpe"],
                }
            )
        print(f"  completed CV fold {fold}/{n_splits}", flush=True)
    return pd.DataFrame(rows)


def summarize_cv(cv_metrics: pd.DataFrame) -> pd.DataFrame:
    if cv_metrics.empty:
        return pd.DataFrame()
    metrics = ["rmse", "mae", "r2", "directional_accuracy", "roc_auc", "strategy_sharpe"]
    summary = cv_metrics.groupby("model")[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().sort_values("rmse_mean")


def rolling_volatility_forecast(
    history_prices: np.ndarray,
    evaluation_prices: np.ndarray,
    horizon: int,
    window: int = 20,
) -> np.ndarray:
    history = list(np.asarray(history_prices, dtype=float))
    predictions = []
    for price in evaluation_prices:
        history.append(float(price))
        log_returns = np.diff(np.log(np.asarray(history[-(window + 1) :], dtype=float)))
        predictions.append(float(np.std(log_returns, ddof=1) * np.sqrt(horizon)))
    return np.asarray(predictions, dtype=float)


def volatility_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual_variance = np.maximum(np.square(actual), 1e-12)
    predicted_variance = np.maximum(np.square(prediction), 1e-12)
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mae": float(mean_absolute_error(actual, prediction)),
        "mean_error_bias": float(np.mean(prediction - actual)),
        "correlation": safe_correlation(actual, prediction),
        "qlike": float(np.mean(np.log(predicted_variance) + actual_variance / predicted_variance)),
    }


def evaluate_volatility_models(
    train: pd.DataFrame,
    validation_context: pd.DataFrame,
    valid: pd.DataFrame,
    dev_train: pd.DataFrame,
    test_context: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
) -> dict:
    _, garch_valid = walk_forward_garch_forecast(
        train["Close"], validation_context["Close"], valid["Close"], horizon
    )
    valid_history = pd.concat([train["Close"], validation_context["Close"]], ignore_index=True)
    rolling_valid = rolling_volatility_forecast(
        valid_history.to_numpy(dtype=float), valid["Close"].to_numpy(dtype=float), horizon
    )
    _, garch_test = walk_forward_garch_forecast(
        dev_train["Close"], test_context["Close"], test["Close"], horizon
    )
    test_history = pd.concat([dev_train["Close"], test_context["Close"]], ignore_index=True)
    rolling_test = rolling_volatility_forecast(
        test_history.to_numpy(dtype=float), test["Close"].to_numpy(dtype=float), horizon
    )
    metrics = {
        "validation": {
            "garch_111": volatility_metrics(valid[TARGET_VOLATILITY].to_numpy(dtype=float), garch_valid),
            "rolling_20d": volatility_metrics(valid[TARGET_VOLATILITY].to_numpy(dtype=float), rolling_valid),
        },
        "test": {
            "garch_111": volatility_metrics(test[TARGET_VOLATILITY].to_numpy(dtype=float), garch_test),
            "rolling_20d": volatility_metrics(test[TARGET_VOLATILITY].to_numpy(dtype=float), rolling_test),
        },
        "note": "GARCH is evaluated against future 5D realized volatility, not return direction.",
    }
    predictions = test[["Date", "Close", TARGET_VOLATILITY]].copy()
    predictions["garch_111_forecast"] = garch_test
    predictions["rolling_20d_forecast"] = rolling_test
    return {"metrics": metrics, "predictions": predictions}


def make_prediction_frame(
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    ensemble_prediction: np.ndarray,
) -> pd.DataFrame:
    output = test[["Date", "Close", TARGET_CLOSE, TARGET_RETURN, TARGET_DIRECTION]].copy()
    all_predictions = {**predictions, "research_ensemble": ensemble_prediction}
    for name, prediction in all_predictions.items():
        output[f"{name}_predicted_return_5d"] = prediction
        output[f"{name}_predicted_close_5d"] = output["Close"] * (1.0 + prediction)
        output[f"{name}_predicted_direction_5d"] = (prediction > 0).astype(int)
    return output


def make_confusion_table(
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    ensemble_prediction: np.ndarray,
) -> pd.DataFrame:
    actual = (test[TARGET_RETURN].to_numpy(dtype=float) > 0).astype(int)
    rows = []
    for name, prediction in {**predictions, "research_ensemble": ensemble_prediction}.items():
        tn, fp, fn, tp = confusion_matrix(actual, prediction > 0, labels=[0, 1]).ravel()
        rows.append({"model": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return pd.DataFrame(rows)


def block_bootstrap_diagnostics(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    baseline_name: str,
    random_state: int,
    samples: int = 500,
    block_size: int = 20,
) -> dict:
    if baseline_name not in predictions:
        raise ValueError(f"Bootstrap baseline is missing: {baseline_name}")
    rng = np.random.default_rng(random_state)
    n_rows = len(y_true)
    starts = np.arange(max(1, n_rows - block_size + 1))
    bootstrap_indices = []
    for _ in range(samples):
        chosen = rng.choice(starts, size=int(np.ceil(n_rows / block_size)), replace=True)
        indices = np.concatenate(
            [np.arange(start, min(start + block_size, n_rows)) for start in chosen]
        )[:n_rows]
        bootstrap_indices.append(indices)

    model_diagnostics = {}
    baseline_prediction = predictions[baseline_name]
    for name, prediction in predictions.items():
        rmse_values = []
        accuracy_values = []
        roc_values = []
        loss_differences = []
        for indices in bootstrap_indices:
            actual = y_true[indices]
            predicted = prediction[indices]
            baseline = baseline_prediction[indices]
            rmse_values.append(float(np.sqrt(np.mean(np.square(actual - predicted)))))
            accuracy_values.append(float(((actual > 0) == (predicted > 0)).mean()))
            try:
                roc_values.append(float(roc_auc_score(actual > 0, predicted)))
            except ValueError:
                pass
            loss_differences.append(
                float(np.mean(np.square(actual - predicted) - np.square(actual - baseline)))
            )
        model_diagnostics[name] = {
            "rmse_95pct_ci": percentile_interval(rmse_values),
            "directional_accuracy_95pct_ci": percentile_interval(accuracy_values),
            "roc_auc_95pct_ci": percentile_interval(roc_values),
            "squared_loss_minus_zero_baseline_95pct_ci": percentile_interval(loss_differences),
        }
    return {
        "method": f"moving-block bootstrap, {samples} samples, block size {block_size}",
        "why_blocked": "Daily 5D targets overlap, so ordinary independent-observation intervals are too optimistic.",
        "baseline": baseline_name,
        "models": model_diagnostics,
        "loss_difference_interpretation": "Negative values favor the named model over the zero-return baseline.",
    }


def percentile_interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "lower": float(np.quantile(array, 0.025)),
        "median": float(np.quantile(array, 0.50)),
        "upper": float(np.quantile(array, 0.975)),
    }


def fit_regime_definitions(train: pd.DataFrame) -> dict[str, dict]:
    definitions: dict[str, dict] = {}

    def add_quantile_regime(
        name: str,
        source: str,
        quantiles: tuple[float, float],
        labels: tuple[str, str, str],
    ) -> None:
        if source not in train.columns:
            return
        values = pd.to_numeric(train[source], errors="coerce").dropna()
        if len(values) < 60 or values.nunique() < 3:
            return
        low, high = values.quantile(list(quantiles)).to_numpy(dtype=float)
        if not np.isfinite([low, high]).all() or low >= high:
            return
        definitions[name] = {
            "source": source,
            "low_threshold": float(low),
            "high_threshold": float(high),
            "quantiles": [float(quantiles[0]), float(quantiles[1])],
            "labels": list(labels),
            "calibration_non_missing_rows": int(len(values)),
        }

    add_quantile_regime(
        "volatility_regime",
        "volatility_20",
        (1.0 / 3.0, 2.0 / 3.0),
        ("low_volatility", "normal_volatility", "high_volatility"),
    )
    add_quantile_regime(
        "trend_regime",
        "return_20d",
        (1.0 / 3.0, 2.0 / 3.0),
        ("downtrend", "range", "uptrend"),
    )
    add_quantile_regime(
        "cot_positioning_regime",
        "npi_z_3y",
        (0.20, 0.80),
        ("speculative_short", "balanced_positioning", "speculative_long"),
    )
    add_quantile_regime(
        "liquidity_regime",
        "volume_vs_ma_20",
        (1.0 / 3.0, 2.0 / 3.0),
        ("low_volume", "normal_volume", "high_volume"),
    )
    add_quantile_regime(
        "news_supply_regime",
        "news_supply_shock_proxy",
        (0.50, 0.85),
        ("quiet_news", "normal_news", "supply_shock_news"),
    )

    weather_components = {
        "weather_brazil_minas_gerais_temperature_2m_max_roll20": 1.0,
        "weather_brazil_minas_gerais_vapour_pressure_deficit_max_roll20": 1.0,
        "weather_brazil_minas_gerais_soil_moisture_0_to_100cm_mean_roll20": -1.0,
        "weather_brazil_minas_gerais_precipitation_sum_roll20": -1.0,
    }
    component_stats = {}
    for column, sign in weather_components.items():
        if column not in train.columns:
            continue
        values = pd.to_numeric(train[column], errors="coerce")
        standard_deviation = float(values.std())
        if values.notna().sum() < 60 or not np.isfinite(standard_deviation) or standard_deviation <= 0:
            continue
        component_stats[column] = {
            "mean": float(values.mean()),
            "std": standard_deviation,
            "stress_sign": float(sign),
        }
    if component_stats:
        provisional = {
            "source": "training_standardized_brazil_weather_stress",
            "components": component_stats,
        }
        score = calculate_weather_stress_score(train, provisional).dropna()
        if len(score) >= 60 and score.nunique() >= 3:
            low, high = score.quantile([1.0 / 3.0, 2.0 / 3.0]).to_numpy(dtype=float)
            if np.isfinite([low, high]).all() and low < high:
                definitions["brazil_weather_regime"] = {
                    **provisional,
                    "low_threshold": float(low),
                    "high_threshold": float(high),
                    "quantiles": [1.0 / 3.0, 2.0 / 3.0],
                    "labels": ["favorable_weather", "normal_weather", "weather_stress"],
                    "calibration_non_missing_rows": int(len(score)),
                }
    return definitions


def calculate_weather_stress_score(frame: pd.DataFrame, definition: dict) -> pd.Series:
    components = []
    for column, details in definition.get("components", {}).items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        components.append(
            (values - float(details["mean"]))
            / float(details["std"])
            * float(details["stress_sign"])
        )
    if not components:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.concat(components, axis=1).mean(axis=1, skipna=False)


def assign_regimes(frame: pd.DataFrame, definitions: dict[str, dict]) -> pd.DataFrame:
    output = pd.DataFrame({"Date": frame["Date"].to_numpy()})
    for regime_name, definition in definitions.items():
        if definition["source"] == "training_standardized_brazil_weather_stress":
            values = calculate_weather_stress_score(frame, definition).reset_index(drop=True)
        else:
            values = pd.to_numeric(frame[definition["source"]], errors="coerce").reset_index(drop=True)
        low_label, middle_label, high_label = definition["labels"]
        labels = np.full(len(values), middle_label, dtype=object)
        labels[values.le(float(definition["low_threshold"])).fillna(False)] = low_label
        labels[values.ge(float(definition["high_threshold"])).fillna(False)] = high_label
        labels[values.isna()] = "unavailable"
        output[regime_name] = labels
        output[f"{regime_name}_source_value"] = values.to_numpy(dtype=float)

    if {"trend_regime", "volatility_regime"}.issubset(output.columns):
        unavailable = output[["trend_regime", "volatility_regime"]].eq("unavailable").any(axis=1)
        output["trend_volatility_regime"] = (
            output["trend_regime"].astype(str) + "__" + output["volatility_regime"].astype(str)
        )
        output.loc[unavailable, "trend_volatility_regime"] = "unavailable"
    else:
        output["trend_volatility_regime"] = "unavailable"
    output["calendar_year"] = pd.to_datetime(output["Date"]).dt.year.astype(str)
    return output


def evaluate_regime_metrics(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    phase: str,
    regime_columns: list[str],
) -> pd.DataFrame:
    actual_all = frame[TARGET_RETURN].to_numpy(dtype=float)
    rows = []
    for regime_column in regime_columns:
        if regime_column not in assignments.columns:
            continue
        for regime_value in sorted(assignments[regime_column].dropna().astype(str).unique()):
            positions = np.flatnonzero(assignments[regime_column].astype(str).eq(regime_value).to_numpy())
            if len(positions) < 20:
                continue
            actual = actual_all[positions]
            actual_direction = actual > 0
            zero_rmse = float(np.sqrt(np.mean(np.square(actual))))
            majority_accuracy = float(max(actual_direction.mean(), 1.0 - actual_direction.mean()))
            for model_name, prediction_all in predictions.items():
                prediction = np.asarray(prediction_all, dtype=float)[positions]
                regression = regression_metrics(actual, prediction)
                direction = direction_metrics(actual, prediction)
                rows.append(
                    {
                        "phase": phase,
                        "regime_family": regime_column,
                        "regime": regime_value,
                        "model": model_name,
                        "rows": int(len(positions)),
                        "date_start": frame.iloc[positions]["Date"].min().date().isoformat(),
                        "date_end": frame.iloc[positions]["Date"].max().date().isoformat(),
                        "actual_mean_return": float(np.mean(actual)),
                        "actual_return_std": float(np.std(actual, ddof=1)),
                        "actual_positive_rate": float(actual_direction.mean()),
                        "prediction_mean_return": float(np.mean(prediction)),
                        "prediction_positive_rate": float((prediction > 0).mean()),
                        "rmse": regression["rmse"],
                        "mae": regression["mae"],
                        "r2": regression["r2"],
                        "mean_error_bias": regression["mean_error_bias"],
                        "directional_accuracy": direction["directional_accuracy"],
                        "balanced_accuracy": direction["balanced_accuracy"],
                        "precision_positive": direction["precision_positive"],
                        "recall_positive": direction["recall_positive"],
                        "f1_positive": direction["f1_positive"],
                        "matthews_correlation": direction["matthews_correlation"],
                        "roc_auc": direction["roc_auc"],
                        "average_precision": direction["average_precision"],
                        "zero_return_rmse": zero_rmse,
                        "rmse_improvement_vs_zero_pct": (
                            float(1.0 - regression["rmse"] / zero_rmse)
                            if zero_rmse > 0
                            else np.nan
                        ),
                        "majority_direction_accuracy": majority_accuracy,
                        "directional_accuracy_minus_majority": float(
                            direction["directional_accuracy"] - majority_accuracy
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_regime_analysis(
    regime_metrics: pd.DataFrame,
    definitions: dict[str, dict],
    calibration_frame: pd.DataFrame | None = None,
) -> dict:
    eligible = regime_metrics.loc[
        regime_metrics["rows"].ge(30)
        & regime_metrics["regime"].ne("unavailable")
        & regime_metrics["model"].ne("zero_return")
    ].copy()
    test = eligible.loc[eligible["phase"].eq("test")].copy()
    group_keys = ["regime_family", "regime"]
    rmse_winners = (
        test.sort_values([*group_keys, "rmse", "mae"])
        .groupby(group_keys, as_index=False)
        .first()[[*group_keys, "model", "rows", "rmse", "mae", "directional_accuracy", "roc_auc"]]
    )
    direction_winners = (
        test.sort_values(
            [*group_keys, "directional_accuracy", "roc_auc", "rmse"],
            ascending=[True, True, False, False, True],
        )
        .groupby(group_keys, as_index=False)
        .first()[[*group_keys, "model", "rows", "directional_accuracy", "roc_auc", "rmse"]]
    )
    robustness = (
        test.groupby("model")
        .agg(
            evaluated_segments=("regime", "size"),
            median_rmse_improvement_vs_zero_pct=("rmse_improvement_vs_zero_pct", "median"),
            positive_rmse_improvement_segments=(
                "rmse_improvement_vs_zero_pct",
                lambda values: int((values > 0).sum()),
            ),
            median_directional_accuracy=("directional_accuracy", "median"),
            worst_directional_accuracy=("directional_accuracy", "min"),
            median_roc_auc=("roc_auc", "median"),
        )
        .reset_index()
        .sort_values(
            ["positive_rmse_improvement_segments", "median_directional_accuracy"],
            ascending=False,
        )
    )

    def paired_summary(left_model: str, right_model: str) -> dict | None:
        paired = test.loc[test["model"].isin([left_model, right_model])].pivot_table(
            index=group_keys,
            columns="model",
            values=["rmse", "directional_accuracy", "roc_auc"],
        )
        required = [("rmse", left_model), ("rmse", right_model)]
        if paired.empty or not all(column in paired.columns for column in required):
            return None
        paired = paired.dropna(subset=required)
        if paired.empty:
            return None
        return {
            "left_model": left_model,
            "right_model": right_model,
            "paired_segments": int(len(paired)),
            "left_lower_rmse_segments": int(
                (paired[("rmse", left_model)] < paired[("rmse", right_model)]).sum()
            ),
            "left_higher_direction_accuracy_segments": int(
                (
                    paired[("directional_accuracy", left_model)]
                    > paired[("directional_accuracy", right_model)]
                ).sum()
            ),
            "left_higher_roc_auc_segments": int(
                (paired[("roc_auc", left_model)] > paired[("roc_auc", right_model)]).sum()
            ),
        }

    comparisons = [
        comparison
        for comparison in [
            paired_summary("xgboost_regime_aware", "xgboost_integrated"),
            paired_summary("xgboost_integrated", "hist_gradient_boosting_integrated"),
        ]
        if comparison is not None
    ]
    summary = {
        "method": (
            "Regime thresholds are calibrated only on the initial training split. Metrics are then "
            "reported separately on validation and untouched test rows. Segments under 20 rows are omitted."
        ),
        "definitions": definitions,
        "eligible_test_segment_model_rows": int(len(test)),
        "test_rmse_winners": rmse_winners.to_dict(orient="records"),
        "test_direction_winners": direction_winners.to_dict(orient="records"),
        "test_model_regime_robustness": robustness.to_dict(orient="records"),
        "paired_model_comparisons": comparisons,
        "warning": (
            "Regime results are diagnostic subgroup estimates, not additional tuning targets. "
            "Small segments and repeated model comparisons can make apparent winners unstable."
        ),
    }
    if calibration_frame is not None:
        summary["calibration_period"] = date_span(calibration_frame)
    return summary


def fit_production_bundle(
    specs: list[ModelSpec],
    selected_members: list[str],
    labeled_frame: pd.DataFrame,
    full_frame: pd.DataFrame,
    horizon: int,
) -> dict:
    production_sklearn_models = {}
    production_time_series_models = {}
    all_prices = full_frame["Close"].dropna().to_numpy(dtype=float)
    for spec in specs:
        if spec.kind == "sklearn":
            model = clone(spec.estimator)
            model.fit(labeled_frame[spec.features], labeled_frame[TARGET_RETURN].astype(float))
            production_sklearn_models[spec.name] = model
        elif spec.kind == "arima_111":
            production_time_series_models[spec.name] = fit_arima111(all_prices)
        elif spec.kind == "simple_exp_smoothing":
            production_time_series_models[spec.name] = fit_simple_exp_smoothing(all_prices)
        elif spec.kind == "holt_winters":
            production_time_series_models[spec.name] = fit_holt_winters(all_prices, period=5)
        elif spec.kind in {"linear_trend", "quadratic_trend"}:
            production_time_series_models[spec.name] = {
                "degree": 1 if spec.kind == "linear_trend" else 2,
                "window": 252,
                "recent_close_history": all_prices[-252:],
                "horizon": horizon,
            }
    production_time_series_models["garch_111_volatility"] = fit_garch11(all_prices)
    return {
        "sklearn_models": production_sklearn_models,
        "ensemble_members": selected_members,
        "time_series_models": production_time_series_models,
        "labeled_training_rows": int(len(labeled_frame)),
        "labeled_training_end": labeled_frame["Date"].max().date().isoformat(),
        "latest_price_date": full_frame["Date"].max().date().isoformat(),
    }


def spec_to_dict(spec: ModelSpec) -> dict:
    return {
        "name": spec.name,
        "approach": spec.approach,
        "paper_keys": list(spec.paper_keys),
        "kind": spec.kind,
        "feature_group": spec.feature_group,
        "feature_count": len(spec.features),
        "features": spec.features,
    }


def write_all_plots(
    comparison: pd.DataFrame,
    cv_summary: pd.DataFrame,
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    ensemble_prediction: np.ndarray,
    selected_base: str,
    volatility_predictions: pd.DataFrame,
    plots_dir: Path,
    horizon: int,
    transaction_cost_bps: float,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    test_comparison = comparison.loc[comparison["phase"].eq("test")].sort_values("rmse")

    figure, axes = plt.subplots(1, 2, figsize=(15, 8))
    sns.barplot(data=test_comparison, y="model", x="rmse", ax=axes[0], color="#287271")
    axes[0].set_title("Holdout 5D Return RMSE")
    axes[0].set_xlabel("RMSE")
    axes[0].set_ylabel("")
    sns.barplot(
        data=test_comparison.sort_values("directional_accuracy", ascending=False),
        y="model",
        x="directional_accuracy",
        ax=axes[1],
        color="#D88C45",
    )
    axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Holdout Directional Accuracy")
    axes[1].set_xlabel("Accuracy")
    axes[1].set_ylabel("")
    figure.tight_layout()
    figure.savefig(plots_dir / "research_paper_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    non_baseline = test_comparison.loc[test_comparison["model"].ne("zero_return")]
    top_models = list(
        dict.fromkeys(
            ["research_ensemble"]
            + non_baseline.sort_values("roc_auc", ascending=False)["model"].head(3).tolist()
            + non_baseline.sort_values("rmse")["model"].head(3).tolist()
        )
    )
    all_predictions = {**predictions, "research_ensemble": ensemble_prediction}
    actual_direction = (test[TARGET_RETURN].to_numpy(dtype=float) > 0).astype(int)
    figure, axis = plt.subplots(figsize=(9, 7))
    for name in top_models:
        false_positive_rate, true_positive_rate, _ = roc_curve(actual_direction, all_predictions[name])
        score = roc_auc_score(actual_direction, all_predictions[name])
        axis.plot(false_positive_rate, true_positive_rate, linewidth=1.7, label=f"{name} ({score:.3f})")
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axis.set_title("Holdout ROC Curves for Positive 5D Return")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(plots_dir / "research_paper_roc_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    confusion_models = list(
        dict.fromkeys(
            ["research_ensemble"]
            + non_baseline.sort_values("directional_accuracy", ascending=False)["model"].head(3).tolist()
        )
    )[:4]
    figure, axes = plt.subplots(2, 2, figsize=(10, 9))
    for axis, name in zip(axes.flat, confusion_models):
        matrix = confusion_matrix(actual_direction, all_predictions[name] > 0, labels=[0, 1])
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(name)
        axis.set_xlabel("Predicted direction")
        axis.set_ylabel("Actual direction")
    for axis in axes.flat[len(confusion_models) :]:
        axis.axis("off")
    figure.suptitle("Holdout Confusion Matrices", y=1.01)
    figure.tight_layout()
    figure.savefig(plots_dir / "research_paper_confusion_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    recent = min(300, len(test))
    dates = test["Date"].iloc[-recent:]
    actual = test[TARGET_RETURN].to_numpy(dtype=float)[-recent:]
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(dates, actual, color="black", linewidth=1.2, label="Actual 5D return")
    display_models = list(
        dict.fromkeys(
            ["research_ensemble"]
            + non_baseline.sort_values("directional_accuracy", ascending=False)["model"].head(2).tolist()
            + non_baseline.sort_values("rmse")["model"].head(1).tolist()
            + [selected_base]
        )
    )[:4]
    colors = ["#287271", "#D88C45", "#B23A48", "#6B5B95"]
    for name, color in zip(display_models, colors):
        axis.plot(
            dates,
            all_predictions[name][-recent:],
            color=color,
            linewidth=1.1,
            alpha=0.9,
            label=name,
        )
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.set_title("Recent Holdout Returns: Actual vs Predicted")
    axis.set_ylabel("5D return")
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "research_paper_actual_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    residual = test[TARGET_RETURN].to_numpy(dtype=float) - ensemble_prediction
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(test["Date"], residual, color="#9C6644", linewidth=0.8)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Ensemble Residuals Over Time")
    axes[0].set_ylabel("Actual minus predicted return")
    sns.histplot(residual, bins=45, kde=True, ax=axes[1], color="#287271")
    axes[1].set_title("Ensemble Residual Distribution")
    figure.tight_layout()
    figure.savefig(plots_dir / "research_paper_residual_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(
        volatility_predictions["Date"],
        volatility_predictions[TARGET_VOLATILITY],
        color="black",
        linewidth=1.1,
        label="Realized next-5D volatility",
    )
    axis.plot(
        volatility_predictions["Date"],
        volatility_predictions["garch_111_forecast"],
        color="#B23A48",
        linewidth=1.0,
        label="GARCH(1,1)",
    )
    axis.plot(
        volatility_predictions["Date"],
        volatility_predictions["rolling_20d_forecast"],
        color="#287271",
        linewidth=1.0,
        label="20D rolling baseline",
    )
    axis.set_title("Holdout Volatility Forecast Check")
    axis.set_ylabel("5D realized volatility")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "research_paper_garch_volatility.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    if not cv_summary.empty:
        figure, axis = plt.subplots(figsize=(11, 7))
        cv_plot = cv_summary.sort_values("rmse_mean")
        axis.errorbar(
            cv_plot["rmse_mean"],
            np.arange(len(cv_plot)),
            xerr=cv_plot["rmse_std"].fillna(0),
            fmt="o",
            color="#287271",
            ecolor="#9CADA7",
            capsize=3,
        )
        axis.set_yticks(np.arange(len(cv_plot)), cv_plot["model"])
        axis.set_title("Expanding-Window CV RMSE (Mean and Standard Deviation)")
        axis.set_xlabel("RMSE")
        axis.set_ylabel("")
        figure.tight_layout()
        figure.savefig(plots_dir / "research_paper_cross_validation.png", dpi=180, bbox_inches="tight")
        plt.close(figure)

    plot_strategy_equity(
        test,
        all_predictions,
        list(
            dict.fromkeys(
                ["research_ensemble"]
                + non_baseline.sort_values("directional_accuracy", ascending=False)["model"].head(2).tolist()
                + ["zero_return"]
            )
        ),
        plots_dir,
        horizon,
        transaction_cost_bps,
    )


def plot_strategy_equity(
    test: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    models: list[str],
    plots_dir: Path,
    horizon: int,
    transaction_cost_bps: float,
) -> None:
    indices = np.arange(0, len(test), horizon)
    dates = test["Date"].iloc[indices]
    actual = test[TARGET_RETURN].to_numpy(dtype=float)[indices]
    figure, axis = plt.subplots(figsize=(13, 6))
    for name in models:
        signal = np.sign(predictions[name][indices])
        turnover = np.abs(signal - np.r_[0.0, signal[:-1]])
        returns = signal * actual - turnover * transaction_cost_bps / 10_000.0
        equity = np.cumprod(1.0 + returns)
        axis.plot(dates, equity, linewidth=1.4, label=name)
    buy_hold = np.cumprod(1.0 + actual)
    axis.plot(dates, buy_hold, color="black", linestyle="--", linewidth=1.0, label="long-only benchmark")
    axis.set_title("Non-Overlapping 5D Strategy Equity (After Transaction Costs)")
    axis.set_ylabel("Growth of 1.0")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "research_paper_strategy_equity.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_regime_plots(
    regime_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    plots_dir: Path,
) -> None:
    test_metrics = regime_metrics.loc[
        regime_metrics["phase"].eq("test")
        & regime_metrics["rows"].ge(30)
        & regime_metrics["regime"].ne("unavailable")
    ].copy()
    if test_metrics.empty:
        return
    overall = comparison.loc[
        comparison["phase"].eq("test") & comparison["model"].ne("zero_return")
    ]
    selected_models = list(
        dict.fromkeys(
            [
                "research_ensemble",
                "xgboost_integrated",
                "xgboost_regime_aware",
                "cot_contract_stress_forest",
                "hist_gradient_boosting_integrated",
            ]
            + overall.sort_values("rmse")["model"].head(3).tolist()
            + overall.sort_values("directional_accuracy", ascending=False)["model"].head(3).tolist()
        )
    )
    selected_models = [
        model for model in selected_models if model in set(test_metrics["model"])
    ][:10]
    plot_data = test_metrics.loc[test_metrics["model"].isin(selected_models)].copy()
    plot_data["segment"] = (
        plot_data["regime_family"].str.replace("_regime", "", regex=False)
        + ": "
        + plot_data["regime"].str.replace("_", " ", regex=False)
    )
    segment_order = list(dict.fromkeys(plot_data["segment"].tolist()))

    for metric, title, filename, center, value_format in [
        (
            "directional_accuracy",
            "Holdout Directional Accuracy by Training-Calibrated Regime",
            "research_paper_regime_direction_accuracy.png",
            0.50,
            ".2f",
        ),
        (
            "rmse_improvement_vs_zero_pct",
            "Holdout RMSE Improvement vs Zero-Return Baseline by Regime",
            "research_paper_regime_rmse_improvement.png",
            0.0,
            ".1%",
        ),
    ]:
        pivot = plot_data.pivot(index="segment", columns="model", values=metric).reindex(
            index=segment_order,
            columns=selected_models,
        )
        figure_height = max(7.0, 0.34 * len(pivot) + 2.0)
        figure, axis = plt.subplots(figsize=(16, figure_height))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=value_format,
            cmap="RdYlGn",
            center=center,
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"shrink": 0.65},
            annot_kws={"fontsize": 7},
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", labelrotation=45, labelsize=8)
        axis.tick_params(axis="y", labelrotation=0, labelsize=8)
        figure.tight_layout()
        figure.savefig(plots_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
