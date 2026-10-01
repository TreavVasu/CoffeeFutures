"""Run paper-family parameter changes on the monthly model's fixed origins."""
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

for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from MonthFu.src.features import load_prices
from MonthFu.src.modeling import add_targets, block_comparison, nonoverlap_positions
from MonthFu.experiments.src.paper_sweep import (
    PaperSpec, direction_block_comparison, direction_regression_metrics, fit_paper, paper_features, paper_specs, predict_paper)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False)+"\n")


def fit_predict(spec, frame, groups, cutoff, origins, stage, diagnostics):
    try:
        member = fit_paper(spec, frame, groups, cutoff)
        diagnostic = {**member["diagnostic"], "stage": stage}
        diagnostics.append(diagnostic)
        predictions = predict_paper(member, frame, origins)
        if not np.isfinite(predictions).all():
            raise ValueError("Nonfinite forecast")
        return member, predictions
    except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
        if not diagnostics or diagnostics[-1].get("model") != spec.name or diagnostics[-1].get("stage") != stage:
            diagnostics.append({"model": spec.name, "family": spec.family, "cutoff": str(cutoff.date()),
                                "stage": stage, "optimizer_success": False})
        diagnostics[-1].update(error=str(exc), prediction_status="invalid_fit_no_fallback")
        print(f"INVALID {stage} {spec.name}: {exc}", flush=True)
        return None, np.full(len(origins), np.nan)


def origin_frame(frame, reference):
    origins = frame.set_index("Date").loc[pd.DatetimeIndex(reference.Date)].reset_index()
    if not np.allclose(origins.target_return, reference.target_return, equal_nan=True, rtol=1e-10, atol=1e-12):
        raise ValueError("Targets differ from original monthly benchmark")
    if not (origins.target_end_date.to_numpy() == reference.target_end_date.to_numpy()).all():
        raise ValueError("Endpoints differ from original monthly benchmark")
    return origins


def ranking_table(predictions, specs, diagnostics):
    rows = []
    for spec in specs:
        valid = predictions[spec.name].notna()
        failures = [row for row in diagnostics if row.get("stage", "").startswith("cv") and row["model"] == spec.name
                    and not row.get("optimizer_success", False)]
        row = {"model": spec.name, "family": spec.family, "parameters": json.dumps(asdict(spec)),
               "valid_rows": int(valid.sum()), "expected_rows": len(predictions), "invalid_fit_count": len(failures),
               "eligible": bool(valid.all() and not failures)}
        if valid.any():
            row.update(direction_regression_metrics(predictions.loc[valid, "target_return"], predictions.loc[valid, spec.name]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["eligible", "rmse"], ascending=[False, True])


def write_report(destination, summary, ranking, metrics):
    primary = metrics.loc[metrics.model == "paper_selected_rmse"].iloc[0]
    direction = metrics.loc[metrics.model == "paper_selected_macro_f1"].iloc[0]
    old = metrics.loc[metrics.model == "monthly_selected"].iloc[0]
    lines = [f"# Paper parameter sweep: {summary['horizon']}-calendar-day return", "",
             "Selection used the original four pre-2022 chronological folds. Configurations and family winners were frozen before evaluating the same 2022+ origins.", "",
             f"The RMSE-selected configuration is `{summary['selection']['rmse_model']}`; the macro-F1-selected configuration is `{summary['selection']['macro_f1_model']}`.", "",
             f"Macro-F1 winner holdout direction: **{direction.direction_accuracy:.2%}**, macro-F1: **{direction.macro_f1:.3f}**, RMSE: **{direction.rmse:.4%}**. The matched monthly baseline has **{old.direction_accuracy:.2%}** direction and **{old.macro_f1:.3f}** macro-F1.", "",
             f"RMSE winner holdout RMSE: **{primary.rmse:.4%}**, direction accuracy: **{primary.direction_accuracy:.2%}**, macro-F1: **{primary.macro_f1:.3f}**. Original monthly selected RMSE: **{old.rmse:.4%}**; direction: **{old.direction_accuracy:.2%}**.", "",
             "| Frozen configuration/reference | RMSE | Direction | Down precision | Down recall | Up precision | Up recall | Macro-F1 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in metrics.itertuples(index=False):
        lines.append(f"| {row.model} | {row.rmse:.4%} | {row.direction_accuracy:.2%} | {row.down_precision:.2%} | {row.down_recall:.2%} | {row.up_precision:.2%} | {row.up_recall:.2%} | {row.macro_f1:.3f} |")
    lines += ["", "These are regressors converted to direction by return sign. Exact-zero realized returns are excluded from binary metrics; predicted zero is assigned Up. Strict-sign accuracy retains the old neutral-zero convention and is also saved.", "",
              f"The development search contained {summary['candidate_count']} configurations; {summary['invalid_cv_configurations']} had an optimizer/fitting failure and were ineligible. Every failure is retained in `fit_diagnostics.csv`, with no silent replacement or favorable-row selection.", "",
              "ARIMA uses custom scipy CSS estimation, stable AR/MA coefficients, d=0/1, and 252/1,260-session histories. It is not an exact statsmodels likelihood replication. Price states use every observed close through each forecast origin; coefficients are frozen within each development fold and refitted quarterly in the holdout. Calendar forecast steps use only weekday counts, not the actual future market calendar.", "",
              "Holdout results are exploratory: the historical holdout was examined previously, and overlapping monthly returns require block/non-overlapping diagnostics. No deep recurrent network was introduced: the small effective monthly sample and existing cache limitations do not justify its complexity. ELM width, ridge penalty, activation and seeds provide a controlled nonlinear sensitivity check.", "",
              "See `cv_ranking.csv`, `holdout_by_year.csv`, `nonoverlapping_metrics.csv`, `metrics.json`, and the source/research note for exact parameters and limitations."]
    if summary["direction_block_comparisons_f1_winner"]:
        interval = summary["direction_block_comparisons_f1_winner"]["60"]["differences"]["direction_accuracy"]
        lines += ["", f"Paired 60-session moving-block direction change versus monthly selected: {interval['measured']:+.2%}; 95% interval [{interval['lower_95']:+.2%}, {interval['upper_95']:+.2%}]. An interval spanning zero leaves improvement uncertain."]
    (destination/"report.md").write_text("\n".join(lines)+"\n")


def run_horizon(horizon, quick, output_root):
    started = time.time()
    destination = output_root/f"{horizon}calendar"
    destination.mkdir(parents=True, exist_ok=True)
    source = ROOT/"MonthFu/artifacts"/f"{horizon}calendar"
    specs = paper_specs(quick)
    lookup = {spec.name: spec for spec in specs}
    prices, audit = load_prices(ROOT/"data/yahoo/arabica_coffee_futures_history.csv")
    price_frame, groups = paper_features(prices)
    frame = add_targets(price_frame, horizon, "calendar")
    cv_reference = pd.read_csv(source/"cv_predictions.csv", parse_dates=["Date", "target_end_date"])
    holdout_reference = pd.read_csv(source/"holdout_predictions.csv", parse_dates=["Date", "target_end_date"])
    if cv_reference.target_end_date.max() >= pd.Timestamp("2022-01-01"):
        raise ValueError("Development labels extend into holdout")
    diagnostics, parts = [], []
    for fold, reference in cv_reference.groupby("fold", sort=True):
        origins = origin_frame(frame, reference)
        cutoff = origins.Date.min()
        part = origins[["Date", "Close", "target_end_date", "target_return", "forecast_steps"]].copy()
        part["fold"] = fold
        for count, spec in enumerate(specs, 1):
            _, part[spec.name] = fit_predict(spec, frame, groups, cutoff, origins, f"cv_{fold}", diagnostics)
            if count % 10 == 0:
                print(f"{horizon}calendar fold {fold}: {count}/{len(specs)}", flush=True)
        parts.append(part)
        pd.concat(parts, ignore_index=True).to_csv(destination/"cv_predictions.partial.csv", index=False)
    oof = pd.concat(parts, ignore_index=True)
    ranking = ranking_table(oof, specs, diagnostics)
    eligible = ranking.loc[ranking.eligible]
    if eligible.empty:
        raise ValueError("No complete converged configuration")
    rmse_model = eligible.iloc[0].model
    f1_model = eligible.sort_values(["macro_f1", "balanced_accuracy", "rmse"], ascending=[False, False, True]).iloc[0].model
    family_rmse = eligible.sort_values("rmse").groupby("family", sort=True).first().model.to_dict()
    family_f1 = eligible.sort_values(["macro_f1", "rmse"], ascending=[False, True]).groupby("family", sort=True).first().model.to_dict()
    selection = {"rmse_model": rmse_model, "macro_f1_model": f1_model,
                 "family_rmse_models": family_rmse, "family_macro_f1_models": family_f1,
                 "frozen_on": "Four matched pre-2022 OOF blocks; no holdout-based selection",
                 "specs": [asdict(spec) for spec in specs],
                 "tie_rule": "Exact-zero actual returns excluded from binary metrics; predicted-zero tie assigned Up"}
    # Persist the exact selected recipe before any holdout fit/prediction.
    write_json(destination/"selection.json", selection)
    ranking.to_csv(destination/"cv_ranking.csv", index=False)
    oof.to_csv(destination/"cv_predictions.csv", index=False)
    diagnostics_cv = pd.DataFrame(diagnostics)
    diagnostics_cv.to_csv(destination/"cv_fit_diagnostics.csv", index=False)
    scored_models = sorted(set([rmse_model, f1_model, "legacy_fixed_arima_111", "legacy_fixed_holt_5",
                               *family_rmse.values(), *family_f1.values()]))
    parts = []
    for quarter, reference in holdout_reference.groupby(holdout_reference.Date.dt.to_period("Q"), sort=True):
        origins = origin_frame(frame, reference)
        cutoff = origins.Date.min()
        part = reference[["Date", "Close", "target_end_date", "target_return"]].copy()
        part["forecast_steps"] = origins.forecast_steps.to_numpy()
        for name in scored_models:
            _, predictions = fit_predict(lookup[name], frame, groups, cutoff, origins, f"holdout_{quarter}", diagnostics)
            part[name] = predictions
        for old_name, new_name in [("selected", "monthly_selected"), ("previous_cv_ensemble", "previous_cv_ensemble"),
                                   ("previous_transferred_ensemble", "previous_transferred_ensemble"),
                                   ("historical_mean", "historical_mean"), ("zero", "zero")]:
            part[new_name] = reference[old_name].to_numpy()
        part["training_majority_return_sign"] = reference.training_majority_direction.to_numpy()*1e-9
        parts.append(part)
        print(f"{horizon}calendar holdout {quarter} complete", flush=True)
    holdout = pd.concat(parts, ignore_index=True)
    holdout["paper_selected_rmse"] = holdout[rmse_model]
    holdout["paper_selected_macro_f1"] = holdout[f1_model]
    metrics_rows = []
    evaluated = scored_models + ["paper_selected_rmse", "paper_selected_macro_f1", "monthly_selected", "previous_cv_ensemble",
                                "previous_transferred_ensemble", "historical_mean", "zero", "training_majority_return_sign"]
    for name in evaluated:
        valid = holdout[name].notna()
        row = {"model": name, "valid_rows": int(valid.sum()), "expected_rows": len(holdout), "complete": bool(valid.all())}
        if valid.any():
            row.update(direction_regression_metrics(holdout.loc[valid, "target_return"], holdout.loc[valid, name]))
        metrics_rows.append(row)
    metrics = pd.DataFrame(metrics_rows)
    holdout.to_csv(destination/"holdout_predictions.csv", index=False)
    metrics.to_csv(destination/"holdout_metrics.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(destination/"fit_diagnostics.csv", index=False)
    annual, disjoint = [], []
    for year, rows in holdout.groupby(holdout.Date.dt.year):
        for name in evaluated:
            if rows[name].notna().all():
                annual.append({"year": year, "model": name, **direction_regression_metrics(rows.target_return, rows[name])})
    for offset in range(horizon):
        rows = holdout.iloc[nonoverlap_positions(holdout, offset)]
        for name in ["paper_selected_rmse", "paper_selected_macro_f1", "monthly_selected", "previous_cv_ensemble"]:
            if len(rows) and rows[name].notna().all():
                disjoint.append({"offset": offset, "model": name, **direction_regression_metrics(rows.target_return, rows[name])})
    pd.DataFrame(annual).to_csv(destination/"holdout_by_year.csv", index=False)
    pd.DataFrame(disjoint).to_csv(destination/"nonoverlapping_metrics.csv", index=False)
    latest_origin = frame.tail(1)
    cutoff = frame.Date.max()+pd.Timedelta(days=1)
    latest_rows, latest_members = [], {}
    for name in scored_models:
        member, prediction = fit_predict(lookup[name], frame, groups, cutoff, latest_origin, "latest", diagnostics)
        if member is not None:
            latest_members[name] = member
        latest_rows.append({"model": name, "origin_date": str(latest_origin.Date.iloc[0].date()),
                            "scheduled_target_date": str((latest_origin.Date.iloc[0]+pd.Timedelta(days=horizon)).date()),
                            "predicted_return": prediction[0], "predicted_direction": ("Up" if prediction[0] >= 0 else "Down") if member is not None else None,
                            "valid_fit": member is not None,
                            "forecast_steps": int(latest_origin.forecast_steps.iloc[0])})
    latest = pd.DataFrame(latest_rows)
    latest.to_csv(destination/"latest_forecast.csv", index=False)
    bundle = {"horizon": horizon, "horizon_unit": "calendar", "members": latest_members,
              "selection": selection, "fitted_as_of": str(frame.Date.max().date()), "feature_groups": groups,
              "bundle_version": 1, "note": "Historical replay must read holdout_predictions.csv; final weights are not historical OOS forecasts"}
    joblib.dump(bundle, destination/"models.joblib")
    comparisons = {}
    if holdout.paper_selected_rmse.notna().all():
        for name in ["monthly_selected", "previous_cv_ensemble", "zero"]:
            comparisons[name] = block_comparison(holdout.target_return, holdout.paper_selected_rmse, holdout[name], block=60)
    summary = {"horizon": horizon, "candidate_count": len(specs), "quick": quick, "cv_rows": len(oof),
               "holdout_rows": len(holdout), "invalid_cv_configurations": int((~ranking.eligible).sum()),
               "selection": selection, "holdout_metrics": metrics.to_dict("records"),
               "block_comparisons_rmse_winner": comparisons,
               "direction_block_comparisons_f1_winner": {
                   str(block): direction_block_comparison(holdout.target_return, holdout.paper_selected_macro_f1,
                                                          holdout.monthly_selected, block=block)
                   for block in [30, 60, 90]} if holdout.paper_selected_macro_f1.notna().all() else {},
               "prices": audit,
               "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "scipy": scipy.__version__,
                               "statsmodels": "Not installed; custom scipy CSS used; no dependencies installed"},
               "elapsed_seconds": time.time()-started,
               "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [Path(__file__), ROOT/"MonthFu/experiments/src/paper_sweep.py",
                                               ROOT/"data/yahoo/arabica_coffee_futures_history.csv", source/"cv_folds.csv"]}}
    write_json(destination/"metrics.json", summary)
    pd.DataFrame(diagnostics).to_csv(destination/"fit_diagnostics.csv", index=False)
    write_json(destination/"feature_groups.json", groups)
    write_report(destination, summary, ranking, metrics)
    print(f"Completed {horizon}calendar: RMSE selection {rmse_model}; macro-F1 selection {f1_model}; seconds {time.time()-started:.1f}", flush=True)
    return summary


def replay(destination, replay_date, model=None):
    selection = json.loads((destination/"selection.json").read_text())
    name = model or selection["macro_f1_model"]
    predictions = pd.read_csv(destination/"holdout_predictions.csv", parse_dates=["Date"])
    matched = predictions.loc[predictions.Date == pd.Timestamp(replay_date)]
    if len(matched) != 1 or name not in matched:
        raise ValueError("Requested date/model has no saved OOS forecast")
    print(matched[["Date", "target_end_date", "target_return", name]].to_csv(index=False), end="")


def predict_latest(destination, model=None):
    bundle = joblib.load(destination/"models.joblib")
    prices, _ = load_prices(ROOT/"data/yahoo/arabica_coffee_futures_history.csv")
    frame, _ = paper_features(prices)
    frame = add_targets(frame, bundle["horizon"], "calendar")
    name = model or bundle["selection"]["macro_f1_model"]
    if name not in bundle["members"]:
        raise ValueError("Requested model has no valid saved fit")
    value = predict_paper(bundle["members"][name], frame, frame.tail(1))[0]
    print(pd.DataFrame([{"model": name, "origin_date": str(frame.Date.iloc[-1].date()),
                         "scheduled_target_date": str((frame.Date.iloc[-1]+pd.Timedelta(days=bundle["horizon"])).date()),
                         "predicted_return": value, "direction": "Up" if value >= 0 else "Down"}]).to_csv(index=False), end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", nargs="+", type=int, default=[30, 28])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT/"MonthFu/experiments/artifacts/paper_parameters")
    parser.add_argument("--replay-date")
    parser.add_argument("--predict-latest", action="store_true", help="Predict from saved models without fitting or selecting")
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.replay_date:
        replay(args.output_root/f"{args.horizons[0]}calendar", args.replay_date, args.model)
        return
    if args.predict_latest:
        predict_latest(args.output_root/f"{args.horizons[0]}calendar", args.model)
        return
    summaries = [run_horizon(horizon, args.quick, args.output_root) for horizon in args.horizons]
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root/"summary.json", summaries)


if __name__ == "__main__":
    main()
