"""Optimize and evaluate a publication-aware 30-day Arabica return model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"]:
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/arabica-futures-matplotlib")

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "Scripts/project/src"))
from return_forecasting import (Candidate, block_comparison, build_frame, candidates, expanding_folds,
                                fit_candidate, mature_training_rows, metrics, predict_bundle, predict_member)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--horizon-unit", choices=["sessions", "calendar"], default="sessions")
    parser.add_argument("--holdout-start", default="2022-01-01")
    parser.add_argument("--cv-splits", type=int, default=4)
    parser.add_argument("--cv-test-size", type=int, default=504)
    parser.add_argument("--include-experimental-news", action="store_true",
                        help="Allow retrospectively selected news candidates; changes bundle status to experimental.")
    parser.add_argument("--quick", action="store_true", help="Smaller development search; use a separate output directory.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Scripts/project/artifacts/returns_30d")
    return parser.parse_args()


def fit_recipe(recipe, spec_lookup, frame, groups, cutoff, quick):
    members = []
    for name, weight in recipe["weights"].items():
        spec = spec_lookup[name]
        train = mature_training_rows(frame, cutoff, spec.train_years)
        member = fit_candidate(spec, train, groups[spec.group], quick)
        member.update(weight=weight, train_rows=len(train), train_last_date=str(train.Date.max().date()),
                      train_last_target_date=str(train.target_end_date.max().date()))
        members.append(member)
    return {"members": members, "recipe": recipe}


def select_recipe(oof: pd.DataFrame, specs: list[Candidate]) -> tuple[dict, pd.DataFrame]:
    """Bounded CV-only selection, including conservative shrinkage toward zero."""
    rows = []
    for spec in specs:
        rows.append({"recipe": spec.name, "weights": {spec.name: 1.0},
                     **metrics(oof.target_return, oof[spec.name])})
    ranked = sorted(rows, key=lambda row: row["rmse"])
    nontrivial = [row for row in ranked if row["recipe"] not in {"zero", "historical_mean"}][:3]
    if nontrivial:
        for count in [1, min(3, len(nontrivial))]:
            names = [row["recipe"] for row in nontrivial[:count]]
            for shrink in [.25, .50, .75, 1.0]:
                weights = {name: shrink/count for name in names}
                p = sum(oof[name].to_numpy() * weight for name, weight in weights.items())
                rows.append({"recipe": f"top{count}_shrink_{shrink:g}", "weights": weights,
                             **metrics(oof.target_return, p)})
    best = min(rows, key=lambda row: row["rmse"])
    selected = {"name": best["recipe"], "weights": best["weights"], "cv_rmse": best["rmse"]}
    table = pd.DataFrame([{**row, "weights": json.dumps(row["weights"], sort_keys=True)} for row in rows])
    return selected, table.sort_values("rmse").reset_index(drop=True)


def latest_forecast(bundle, frame):
    latest = frame.tail(1)
    forecast = float(predict_bundle(bundle, latest)[0])
    date = latest.Date.iloc[0]
    horizon, unit = bundle["horizon"], bundle["horizon_unit"]
    estimate = date + (pd.offsets.BDay(horizon) if unit == "sessions" else pd.Timedelta(days=horizon))
    band = bundle["cv_absolute_error_q80"]
    return pd.DataFrame([{
        "as_of_date": date, "horizon": horizon, "horizon_unit": unit,
        "target_date_estimate": estimate,
        "target_date_policy": "Weekday estimate; exchange holidays not projected" if unit == "sessions" else "First observed market session on/after this calendar date",
        "close": float(latest.Close.iloc[0]), "predicted_return": forecast,
        "predicted_return_pct": 100*forecast,
        "implied_close": float(latest.Close.iloc[0]) * (1+forecast),
        "cv_empirical_lower_80": forecast-band, "cv_empirical_upper_80": forecast+band,
        "interval_policy": "Absolute out-of-fold residual band; exploratory, selected on same CV, no coverage guarantee",
        "recipe": bundle["recipe"]["name"],
        "data_status": "Historical local-cache forecast; not a live current-market quote",
    }])


def main():
    args = parse_args()
    started = time.perf_counter()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("Building raw-price targets and publication-aware features...", flush=True)
    frame, groups, cot_audit, metadata = build_frame(ROOT, args.horizon, args.horizon_unit)
    cutoff = pd.Timestamp(args.holdout_start)
    folds = expanding_folds(frame, cutoff, args.cv_splits, args.cv_test_size)
    specs = candidates(args.include_experimental_news, args.quick)
    lookup = {s.name: s for s in specs}
    cv_rows, oof_parts, fold_audit = [], [], []
    for number, (train, valid) in enumerate(folds, 1):
        print(f"CV {number}/{len(folds)}: {len(train)} mature train rows, {valid.Date.min().date()} to {valid.Date.max().date()}", flush=True)
        part = valid[["Date", "Close", "target_end_date", "target_return"]].copy()
        part["fold"] = number
        for spec in specs:
            fit_train = mature_training_rows(frame, valid.Date.iloc[0], spec.train_years)
            fitted = fit_candidate(spec, fit_train, groups[spec.group], args.quick)
            prediction = predict_member(fitted, valid)
            part[spec.name] = prediction
            cv_rows.append({"fold": number, "candidate": spec.name, "feature_group": spec.group,
                            "train_rows": len(fit_train), "features": len(fitted["features"]),
                            **metrics(valid.target_return, prediction)})
        oof_parts.append(part)
        fold_audit.append({"fold": number, "train_start": train.Date.min(), "train_end": train.Date.max(),
                           "latest_training_label_end": train.target_end_date.max(),
                           "validation_start": valid.Date.min(), "validation_end": valid.Date.max(),
                           "latest_validation_label_end": valid.target_end_date.max(),
                           "holdout_boundary": cutoff})
    oof = pd.concat(oof_parts, ignore_index=True)
    recipe, ranking = select_recipe(oof, specs)
    print("CV-selected recipe (frozen before holdout):", recipe, flush=True)
    oof["selected"] = sum(oof[name] * weight for name, weight in recipe["weights"].items())
    error_q80 = float(np.quantile(np.abs(oof.target_return - oof.selected), .80))
    # Save the decision before reading holdout performance.
    (out / "selection.json").write_text(json.dumps({"recipe": recipe, "candidates": [asdict(s) for s in specs],
        "holdout_start": str(cutoff.date()), "refit_policy": "Quarterly, fixed recipe; only labels ending before block start"}, indent=2))
    holdout = frame.loc[frame.Date.ge(cutoff) & frame.target_return.notna()].copy()
    if len(holdout) < max(60, 2*args.horizon):
        raise ValueError("Holdout is too small for the requested horizon.")
    comparison_recipes = {"selected": recipe,
                          "zero": {"name": "zero", "weights": {"zero": 1}},
                          "historical_mean": {"name": "historical_mean", "weights": {"historical_mean": 1}},
                          "compact_legacy_30d": {"name": "compact_legacy_30d", "weights": {"compact_legacy_30d": 1}}}
    holdout_parts, refit_rows = [], []
    for period, block in holdout.groupby(holdout.Date.dt.to_period("Q"), sort=True):
        block_start = block.Date.iloc[0]
        print(f"Holdout {period}: refit using labels ending before {block_start.date()}", flush=True)
        part = block[["Date", "Close", "target_end_date", "target_return", "target_close"]].copy()
        for name, this_recipe in comparison_recipes.items():
            bundle = fit_recipe(this_recipe, lookup, frame, groups, block_start, args.quick)
            part[name] = predict_bundle(bundle, block)
            for member in bundle["members"]:
                refit_rows.append({"quarter": str(period), "comparison": name, "member": member["spec"]["name"],
                                   "weight": member["weight"], "first_prediction_date": block_start,
                                   "train_rows": member["train_rows"], "train_last_date": member["train_last_date"],
                                   "train_last_target_date": member["train_last_target_date"]})
        part["empirical_lower_80"] = part.selected-error_q80
        part["empirical_upper_80"] = part.selected+error_q80
        holdout_parts.append(part)
    predictions = pd.concat(holdout_parts, ignore_index=True)
    comparison = pd.DataFrame([{"model": name, **metrics(predictions.target_return, predictions[name])}
                               for name in comparison_recipes])
    nonoverlap_rows = []
    # Greedy true-date disjoint label windows work for sessions and calendar targets.
    for offset in range(args.horizon):
        positions, next_start = [], pd.Timestamp.min
        for idx in range(offset, len(predictions)):
            row = predictions.iloc[idx]
            if row.Date >= next_start:
                positions.append(idx)
                next_start = row.target_end_date
        sample = predictions.iloc[positions]
        for name in comparison_recipes:
            nonoverlap_rows.append({"offset": offset, "model": name, **metrics(sample.target_return, sample[name])})
    latest_cutoff = frame.Date.max() + pd.Timedelta(days=1)
    latest_bundle = fit_recipe(recipe, lookup, frame, groups, latest_cutoff, args.quick)
    latest_bundle.update(horizon=args.horizon, horizon_unit=args.horizon_unit,
                         cv_absolute_error_q80=error_q80, feature_metadata=metadata,
                         fitted_as_of=str(frame.Date.max().date()),
                         status="experimental_news" if args.include_experimental_news else "research")
    latest = latest_forecast(latest_bundle, frame)
    comparisons = {name: block_comparison(predictions.target_return, predictions.selected, predictions[name], block=block)
                   for name in ["zero", "compact_legacy_30d"] for block in [max(60, args.horizon*2)]}
    zero_sensitivity = {str(block): block_comparison(predictions.target_return, predictions.selected, predictions.zero, block=block)
                        for block in [30, 90]}
    source_paths = [ROOT / "data/yahoo/arabica_coffee_futures_history.csv",
                    ROOT / "data/COT/coffee_c_all_cot_data.csv",
                    ROOT / "data/weather/open_meteo_coffee_regions_daily.csv",
                    ROOT / "data/COT/cot_release_overrides.csv",
                    ROOT / "Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_daily_summary.csv",
                    ROOT / "Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_weekly_summary.csv"]
    source_paths = [path for path in source_paths if path.exists()]
    summary = {"horizon": args.horizon, "horizon_unit": args.horizon_unit, "selection": recipe,
               "cv_rows": len(oof), "cv_folds": len(folds), "holdout_start": str(predictions.Date.min().date()),
               "holdout_end": str(predictions.Date.max().date()), "holdout_refit_policy": "Quarterly with recipe frozen on pre-holdout CV",
               "holdout_metrics": comparison.to_dict(orient="records"), "block_comparisons": comparisons,
               "zero_bootstrap_block_sensitivity": zero_sensitivity,
               "empirical_80pct_band_holdout_coverage": float(((predictions.target_return >= predictions.empirical_lower_80) &
                                                             (predictions.target_return <= predictions.empirical_upper_80)).mean()),
               "cv_absolute_error_q80": error_q80, "data": metadata,
               "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
               "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
               "limitations": ["Reused historical holdout is exploratory, not untouched prospective evidence.",
                   "COT includes estimated publication dates and known-delay guards, not complete historical vintages.",
                   "Weather is retrospectively reanalyzed; a five-day lag does not certify source-vintage availability.",
                   "News is excluded by default because retrospective selection cannot be fully undone by a six-session lag.",
                   "Daily 30-day labels overlap; use block uncertainty and non-overlapping diagnostics.",
                   "OHLC inconsistencies, tiny volume and continuous-contract roll effects limit data/execution interpretation.",
                   "Forecast begins at last local price date, not today's date; projected target date is approximate.",
                   "Empirical error band uses selection CV residuals, so coverage is not guaranteed."],
               "elapsed_seconds": round(time.perf_counter()-started, 2)}
    for name, data in [("cv_folds", pd.DataFrame(fold_audit)), ("cv_metrics", pd.DataFrame(cv_rows)),
                       ("cv_predictions", oof), ("cv_ranking", ranking), ("holdout_predictions", predictions),
                       ("holdout_metrics", comparison), ("holdout_refits", pd.DataFrame(refit_rows)),
                       ("nonoverlapping_metrics", pd.DataFrame(nonoverlap_rows)),
                       ("latest_forecast", latest), ("cot_release_audit", cot_audit)]:
        data.to_csv(out / f"{name}.csv", index=False)
    manifest = pd.DataFrame([{"group": group, "feature": col} for group, cols in groups.items() for col in cols])
    manifest.to_csv(out / "feature_manifest.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str, allow_nan=False))
    joblib.dump(latest_bundle, out / "model.joblib", compress=3)
    write_report(out, summary, latest)
    write_plot(out, predictions, comparison)
    print(comparison[["model", "rmse", "mae", "direction_accuracy", "r2_vs_zero"]].to_string(index=False), flush=True)
    print(latest[["as_of_date", "predicted_return_pct", "target_date_estimate"]].to_string(index=False), flush=True)
    print(f"Saved {out}; elapsed {summary['elapsed_seconds']}s", flush=True)


def write_report(out, summary, latest):
    rows = summary["holdout_metrics"]
    selected = next(row for row in rows if row["model"] == "selected")
    zero = next(row for row in rows if row["model"] == "zero")
    improved = selected["rmse"] < zero["rmse"]
    text = [f"# Arabica {summary['horizon']}-{summary['horizon_unit']} return experiment", "",
            f"CV-selected recipe: `{summary['selection']['name']}`. Weights: `{summary['selection']['weights']}`.", "",
            "The selected recipe " + ("improves" if improved else "does not improve") + " holdout RMSE versus zero-return. This is a historical research result, not a demonstrated profitable strategy.", "",
            f"Holdout forecast origins: {summary['holdout_start']} to {summary['holdout_end']}; quarterly refits with fully matured labels. Selection used only earlier purged expanding folds.", "",
            "| Model | RMSE | MAE | Direction accuracy |", "|---|---:|---:|---:|"]
    text += [f"| {r['model']} | {r['rmse']:.2%} | {r['mae']:.2%} | {r['direction_accuracy']:.2%} |" for r in rows]
    text += ["", "Zero is a neutral forecast; its sign accuracy is not an up/down classifier baseline.", "",
             f"Selected minus zero RMSE, 60-session block interval: {summary['block_comparisons']['zero']['rmse_difference_95pct_block_interval']}.", "",
             f"Latest cached origin: {latest.as_of_date.iloc[0].date()}; predicted return {latest.predicted_return_pct.iloc[0]:+.2f}%. This is not a live forecast for today.", "",
             "See `feature_manifest.csv`, `cot_release_audit.csv`, `cv_folds.csv` and `holdout_refits.csv` for reproducible alignment and split audits.", "", "## Limits", ""]
    text.extend(f"- {item}" for item in summary["limitations"])
    (out / "report.md").write_text("\n".join(text)+"\n")


def write_plot(out, predictions, comparison):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(predictions.Date, 100*predictions.target_return, color="#738796", alpha=.7, label="Observed forward return")
    axes[0].plot(predictions.Date, 100*predictions.selected, color="#99582a", label="CV-selected forecast")
    axes[0].axhline(0, color="black", linewidth=.5)
    axes[0].set_ylabel("Return (%)")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Arabica 30-day returns — quarterly walk-forward holdout")
    axes[1].barh(comparison.model, 100*comparison.rmse, color=["#99582a", "#738796", "#a9b9c4", "#ccd5da"])
    axes[1].set_xlabel("Holdout RMSE (percentage points; lower is better)")
    fig.tight_layout()
    fig.savefig(out / "holdout.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
