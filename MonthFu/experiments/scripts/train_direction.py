"""Direction/return feature ablations; selection is frozen before 2022 holdout."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"]:
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / "MonthFu" / "experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS / "src"))

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from direction_features import build_experiment_frame, cot_feature_columns, cot_schema_hash, cot_source_policy
from direction_metrics import (binary_labels_from_returns, classification_metrics,
    nonoverlap_positions, paired_block_bootstrap, select_abstention_band,
    select_threshold, selective_metrics)
from experiment_models import (Spec, candidates, fit_member, predict_recipe,
                               predict_member, regression_direction_score)
from MonthFu.src.modeling import mature_training_rows, metrics


def ready(value):
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def dump(path, value):
    path.write_text(json.dumps(ready(value), indent=2, allow_nan=False) + "\n")


def binary_slice(frame):
    y = binary_labels_from_returns(frame.target_return)
    mask = np.isfinite(y)
    return y[mask], mask


def direction_row(frame, prediction, threshold=.5, classifier=True):
    y, mask = binary_slice(frame)
    score = np.asarray(prediction) if classifier else regression_direction_score(prediction)
    result = classification_metrics(y, score[mask], threshold)
    result["excluded_neutral_rows"] = int((~mask).sum())
    result["score_type"] = "probability_up" if classifier else "uncalibrated_return_rank_score"
    if not classifier:
        # These monotonic return scores are not probabilistic forecasts.
        result["brier"] = result["log_loss"] = None
    return result


def classify_selection(oof, values, cutoff):
    y, mask = binary_slice(oof)
    rows = oof.loc[mask]
    return select_threshold(y, np.asarray(values)[mask], dates=rows.Date,
        target_end_dates=rows.target_end_date, holdout_start=cutoff,
        thresholds=np.arange(.35, .651, .025))


def select_recipes(oof, specs, groups, cutoff):
    ranking, decision, recipes = [], {}, {}
    for spec in specs:
        values = oof[spec.name].to_numpy()
        if spec.task == "class":
            selection = classify_selection(oof, values, cutoff)
            decision[spec.name] = selection
            score = selection["selection_metrics"]
        else:
            score = {**metrics(oof.target_return, values), **direction_row(oof, values, classifier=False)}
        ranking.append({"candidate": spec.name, "task": spec.task, "group": spec.group,
                        "family": spec.family, "variant": spec.variant, **score})
    table = pd.DataFrame(ranking)
    for task in ["class", "reg"]:
        for group in groups:
            chosen = []
            for family in ["linear", "xgb", "forest"]:
                rows = table.loc[table.task.eq(task) & table.group.eq(group) & table.family.eq(family)]
                rows = (rows.sort_values(["macro_f1", "balanced_accuracy", "candidate"], ascending=[False, False, True])
                        if task == "class" else rows.sort_values(["rmse", "candidate"]))
                candidate = rows.candidate.iloc[0]
                name = f"{task}__{group}__{family}"
                recipes[name] = {"name": name, "task": task, "group": group,
                    "weights": {candidate: 1.}, "threshold": decision[candidate]["threshold"] if task == "class" else .5}
                chosen.append(candidate)
            name = f"{task}__{group}__equal_ensemble"
            values = np.mean(oof[chosen].to_numpy(), axis=1)
            oof[name] = values
            if task == "class":
                decision[name] = classify_selection(oof, values, cutoff)
                score = decision[name]["selection_metrics"]
            else:
                score = {**metrics(oof.target_return, values), **direction_row(oof, values, classifier=False)}
            recipes[name] = {"name": name, "task": task, "group": group,
                "weights": {c: 1/3 for c in chosen},
                "threshold": decision[name]["threshold"] if task == "class" else .5}
            ranking.append({"candidate": name, "task": task, "group": group,
                            "family": "equal_ensemble", "variant": None, **score})
    table = pd.DataFrame(ranking)
    # Select only family winners and explicit equal ensembles for deployment.
    deployable = table.loc[table.candidate.isin(
        [next(iter(r["weights"])) if len(r["weights"]) == 1 else n for n, r in recipes.items()])]
    best_class = deployable.loc[deployable.task.eq("class")].sort_values(
        ["macro_f1", "balanced_accuracy", "candidate"], ascending=[False, False, True]).candidate.iloc[0]
    best_reg = deployable.loc[deployable.task.eq("reg")].sort_values(["rmse", "candidate"]).candidate.iloc[0]
    def recipe_for(candidate):
        if candidate in recipes:
            return dict(recipes[candidate])
        return dict(next(r for r in recipes.values() if r["weights"] == {candidate: 1.}))
    recipes["selected_direction"] = recipe_for(best_class)
    recipes["selected_return"] = recipe_for(best_reg)
    selected = predict_oof(recipes["selected_direction"], oof)
    y, mask = binary_slice(oof)
    rows = oof.loc[mask]
    abstention = select_abstention_band(y, selected[mask], recipes["selected_direction"]["threshold"],
        dates=rows.Date, target_end_dates=rows.target_end_date, holdout_start=cutoff)
    return recipes, table, decision, abstention


def predict_oof(recipe, oof):
    return sum(w*oof[c].to_numpy() for c, w in recipe["weights"].items())


def add_baselines(part, old, train):
    merged = part[["Date"]].merge(old[["Date", "selected", "previous_cv_ensemble"]], on="Date", validate="one_to_one")
    if len(merged) != len(part) or not merged.Date.to_numpy().tolist() == part.Date.to_numpy().tolist():
        raise AssertionError("Baseline dates must match exactly")
    part["baseline_monthly_return"] = merged.selected.to_numpy()
    part["baseline_paper_return"] = merged.previous_cv_ensemble.to_numpy()
    part["baseline_majority_probability"] = train.loc[train.target_return.ne(0)].target_return.gt(0).mean()
    part["baseline_mean_return"] = train.target_return.mean()
    part["baseline_zero_return"] = 0.


def evaluate(frame, recipes):
    class_rows, reg_rows = [], []
    for name, recipe in recipes.items():
        pred = frame[name].to_numpy()
        class_rows.append({"model": name, "task": recipe["task"], "group": recipe["group"],
            **direction_row(frame, pred, recipe["threshold"], recipe["task"] == "class")})
        if recipe["task"] == "reg":
            reg_rows.append({"model": name, "group": recipe["group"], **metrics(frame.target_return, pred)})
    for name in ["baseline_monthly_return", "baseline_paper_return", "baseline_mean_return", "baseline_zero_return"]:
        class_rows.append({"model": name, "task": "reg", "group": "previous",
                           **direction_row(frame, frame[name], classifier=False)})
        reg_rows.append({"model": name, "group": "previous", **metrics(frame.target_return, frame[name])})
    for name, values in [("baseline_majority", frame.baseline_majority_probability),
                         ("always_up", np.ones(len(frame))), ("always_down", np.zeros(len(frame)))]:
        class_rows.append({"model": name, "task": "baseline", "group": "previous",
                           **direction_row(frame, values)})
    return pd.DataFrame(class_rows), pd.DataFrame(reg_rows)


def source_hashes():
    paths = sorted((EXPERIMENTS / "src").glob("*.py")) + [Path(__file__)]
    paths += [ROOT/"MonthFu/src/features.py", ROOT/"MonthFu/src/modeling.py",
              ROOT/"MonthFu/src/cot_release_features.py", ROOT/"MonthFu/src/external_return_features.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def artifact_directory(horizon=30, quick=False, cot_policy="all"):
    if cot_policy not in {"all", "benchmark"}:
        raise ValueError("COT policy must be all or benchmark")
    policy_dir = "all_cot" if cot_policy == "all" else "benchmark_retrained"
    return EXPERIMENTS / "artifacts" / policy_dir / ("quick" if quick else "direction") / f"{horizon}calendar"


def export_training_data(out, frame, groups, metadata):
    """Persist actual source predictors without filling unknown historical cells."""
    columns = list(dict.fromkeys(c for members in groups.values() for c in members))
    features = ["Date", "Close", *[c for c in columns if c != "Close"]]
    labels = ["target_end_date", "target_return", "target_close", "target_sessions", "forecast_steps"]
    frame[[*features, *labels]].to_csv(out/"features_and_targets.csv.gz", index=False, compression="gzip")
    cutoff = frame.Date.max() + pd.Timedelta(days=1)
    training = mature_training_rows(frame, cutoff)
    training[[*features, *labels]].to_csv(out/"training_dataset.csv.gz", index=False, compression="gzip")
    pd.DataFrame(metadata["feature_manifest"]).to_csv(out/"feature_manifest.csv", index=False)
    cot = cot_feature_columns(frame)
    before_2009 = training.loc[training.Date.lt("2009-01-01")]
    audit = {"cot_policy": metadata["cot_policy"], "cot_feature_schema_sha256": cot_schema_hash(cot),
        "cot_features": cot, "origin_rows": len(frame), "latest_mature_training_rows": len(training),
        "final_training_cutoff": str(cutoff.date()), "pre_2009_training_rows": len(before_2009),
        "pre_2009_observed_legacy_rows": int(before_2009[[c for c in cot if c.startswith("cot_legacy_")]].notna().any(axis=1).sum()),
        "groups_include_entire_cot_bank": {name: set(cot).issubset(columns) for name, columns in groups.items()},
        "dataset_policy": "Raw missing predictor cells remain missing in both exports; only mature final-refit labels enter training_dataset. Fold training is reconstructed by target_end_date < fold cutoff.",
        "empty_feature_policy": "Required COT columns survive coverage and variance checks. Train-only median imputation keeps empty columns using sklearn's numeric placeholder plus a missingness indicator; empty/constant columns provide no fitted position signal."}
    dump(out/"training_dataset_audit.json", audit)
    return audit


def train(horizon=30, quick=False, cot_policy="all"):
    started = time.perf_counter()
    key = f"{horizon}calendar"
    out = artifact_directory(horizon, quick, cot_policy)
    out.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp("2022-01-01")
    frame, groups, metadata = build_experiment_frame(ROOT, horizon, "calendar", cot_policy=cot_policy)
    olddir = ROOT / "MonthFu/artifacts" / key
    oldcv = pd.read_csv(olddir/"cv_predictions.csv", parse_dates=["Date", "target_end_date"])
    oldhold = pd.read_csv(olddir/"holdout_predictions.csv", parse_dates=["Date", "target_end_date"])
    specs = candidates(groups, require_all_cot=cot_policy == "all")
    lookup = {s.name: s for s in specs}
    dump(out/"candidate_parameters.json", [asdict(s) for s in specs])
    dump(out/"feature_metadata.json", metadata)
    dump(out/"feature_groups.json", groups)
    dataset_audit = export_training_data(out, frame, groups, metadata)
    fits, fold_rows, parts = [], [], []
    for fold, anchor in oldcv.groupby("fold", sort=True):
        valid = frame.loc[frame.Date.isin(anchor.Date)].copy()
        if len(valid) != len(anchor) or not valid.target_end_date.lt(cutoff).all():
            raise AssertionError("Validation origins or label maturity disagree with existing benchmark")
        rows = mature_training_rows(frame, valid.Date.iloc[0])
        print(f"[{key}] CV {fold}: {len(rows)} train, {len(valid)} origins, {len(specs)} models", flush=True)
        part = valid[["Date", "Close", "target_end_date", "target_return"]].copy()
        part["fold"] = fold
        values = {}
        for spec in specs:
            model = fit_member(spec, rows, groups[spec.group], quick)
            values[spec.name] = predict_member(model, valid)
            fits.append({"phase": "cv", "fold": int(fold), "candidate": spec.name,
                         "cutoff": str(valid.Date.iloc[0].date()),
                         **{k: model[k] for k in ["features", "required_cot_features", "cot_inclusion_audit", "train_rows", "train_last_date", "train_last_target_date", "warnings"]}})
        part = pd.concat([part, pd.DataFrame(values, index=part.index)], axis=1)
        add_baselines(part, oldcv, rows)
        parts.append(part)
        fold_rows.append({"fold": int(fold), "train_rows": len(rows), "train_end": rows.Date.max(),
            "latest_training_label_end": rows.target_end_date.max(), "validation_start": valid.Date.min(),
            "validation_end": valid.Date.max(), "latest_validation_label_end": valid.target_end_date.max()})
        pd.concat(parts, ignore_index=True).to_csv(out/"cv_predictions_partial.csv", index=False)
    oof = pd.concat(parts, ignore_index=True)
    recipes, ranking, decisions, abstention = select_recipes(oof, specs, groups, cutoff)
    for name, recipe in recipes.items():
        oof[name] = predict_oof(recipe, oof)
    selection = {"holdout_start": str(cutoff.date()), "recipes": recipes, "cot_policy": cot_policy,
        "required_cot_features": metadata["cot_feature_names"] if cot_policy == "all" else [],
        "cot_feature_schema_sha256": metadata["cot_feature_schema_sha256"],
        "threshold_decisions": decisions, "selected_abstention": abstention,
        "primary_direction_metric": "Pooled pre-2022 OOF macro F1, then balanced accuracy",
        "primary_return_metric": "Pooled pre-2022 OOF RMSE",
        "selection_policy": "Two fixed parameter recipes per family and feature group; select one per family; equal-weight three-family ensemble; thresholds fixed from OOF only.",
        "holdout_status": "Previously inspected historical holdout: exploratory follow-up, not a new untouched test.",
        "refit_policy": "Quarterly, target_end_date strictly before first prediction date",
        "neural_policy": "No deep sequence network: monthly overlapping labels yield only a few hundred effective independent outcomes. Fixed nonlinear ELM is tested in the separate paper sweep.",
        "classification_zero_policy": "Exactly zero realized returns are excluded consistently from binary scores and classifier fitting; returns retained for regression. Predicted zero is Up for binary threshold reporting, neutral for original strict regression-sign accuracy."}
    # Persist all choices before any holdout model fits or metrics are computed.
    dump(out/"selection.json", selection)
    oof.to_csv(out/"cv_predictions.csv", index=False)
    ranking.to_csv(out/"cv_ranking.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out/"cv_folds.csv", index=False)
    cvclass, cvreg = evaluate(oof, recipes)
    cvclass.to_csv(out/"cv_classification_metrics.csv", index=False)
    cvreg.to_csv(out/"cv_regression_metrics.csv", index=False)
    print(f"[{key}] Frozen direction: {recipes['selected_direction']}; return: {recipes['selected_return']}", flush=True)
    needed = sorted({c for recipe in recipes.values() for c in recipe["weights"]})
    hold = frame.loc[frame.Date.isin(oldhold.Date)].copy()
    holdparts = []
    for quarter, block in hold.groupby(hold.Date.dt.to_period("Q"), sort=True):
        rows = mature_training_rows(frame, block.Date.iloc[0])
        print(f"[{key}] Holdout {quarter}: {len(needed)} distinct models", flush=True)
        fitted = {c: fit_member(lookup[c], rows, groups[lookup[c].group], quick) for c in needed}
        for c, model in fitted.items():
            fits.append({"phase": "holdout", "quarter": str(quarter), "candidate": c,
                "cutoff": str(block.Date.iloc[0].date()),
                **{k: model[k] for k in ["features", "required_cot_features", "cot_inclusion_audit", "train_rows", "train_last_date", "train_last_target_date", "warnings"]}})
        part = block[["Date", "Close", "target_end_date", "target_return", "target_sessions"]].copy()
        part["quarter"] = str(quarter)
        part = pd.concat([part, pd.DataFrame(
            {name: predict_recipe(recipe, fitted, block) for name, recipe in recipes.items()}, index=part.index)], axis=1)
        add_baselines(part, oldhold, rows)
        holdparts.append(part)
        pd.concat(holdparts, ignore_index=True).to_csv(out/"holdout_predictions_partial.csv", index=False)
    hold = pd.concat(holdparts, ignore_index=True)
    hold.to_csv(out/"holdout_predictions.csv", index=False)
    dump(out/"fit_audit.json", fits)
    holdclass, holdreg = evaluate(hold, recipes)
    holdclass.to_csv(out/"holdout_classification_metrics.csv", index=False)
    holdreg.to_csv(out/"holdout_regression_metrics.csv", index=False)
    yearly_class, yearly_reg = [], []
    for year, rows in hold.groupby(hold.Date.dt.year):
        c, r = evaluate(rows, recipes)
        yearly_class.append(c.assign(year=year)); yearly_reg.append(r.assign(year=year))
    pd.concat(yearly_class).to_csv(out/"holdout_classification_by_year.csv", index=False)
    pd.concat(yearly_reg).to_csv(out/"holdout_regression_by_year.csv", index=False)
    independent = []
    for offset in range(30):
        rows = hold.iloc[nonoverlap_positions(hold, offset)]
        c, _ = evaluate(rows, recipes)
        independent.append(c.assign(offset=offset))
    pd.concat(independent).to_csv(out/"nonoverlapping_classification_metrics.csv", index=False)
    y, mask = binary_slice(hold)
    bootstrap = {str(length): paired_block_bootstrap(y, hold.selected_direction.to_numpy()[mask],
        regression_direction_score(hold.baseline_monthly_return)[mask],
        threshold_selected=recipes["selected_direction"]["threshold"], block=length, draws=1000)
        for length in [30, 60, 90]}
    dump(out/"direction_comparison_bootstrap.json", bootstrap)
    selective = selective_metrics(y, hold.selected_direction.to_numpy()[mask], abstention["lower"], abstention["upper"])
    dump(out/"holdout_selective_metrics.json", selective)
    changes = []
    baseline = holdclass.loc[holdclass.model.eq("baseline_monthly_return")].iloc[0]
    base_rmse = float(holdreg.loc[holdreg.model.eq("baseline_monthly_return"), "rmse"].iloc[0])
    for name in ["selected_direction", "class__basic__equal_ensemble", "selected_return", "reg__basic__equal_ensemble"]:
        row = holdclass.loc[holdclass.model.eq(name)].iloc[0]
        item = {"model": name, **{f"delta_{m}": row[m]-baseline[m] for m in
            ["accuracy", "macro_f1", "balanced_accuracy", "up_precision", "up_recall", "down_precision", "down_recall"]}}
        if recipes[name]["task"] == "reg":
            rmse = float(holdreg.loc[holdreg.model.eq(name), "rmse"].iloc[0])
            item.update(rmse=rmse, baseline_rmse=base_rmse, rmse_relative_improvement=1-rmse/base_rmse)
        changes.append(item)
    pd.DataFrame(changes).to_csv(out/"changes_vs_monthly.csv", index=False)
    # Only the selected and simple-basic recipes need a deployment bundle.
    final_recipes = {n: recipes[n] for n in ["selected_direction", "selected_return",
        "class__basic__equal_ensemble", "reg__basic__equal_ensemble"]}
    final_cutoff = frame.Date.max()+pd.Timedelta(days=1)
    final_train = mature_training_rows(frame, final_cutoff)
    final_names = sorted({n for r in final_recipes.values() for n in r["weights"]})
    final_members = {n: fit_member(lookup[n], final_train, groups[lookup[n].group], quick) for n in final_names}
    for name, model in final_members.items():
        fits.append({"phase": "final", "candidate": name, "cutoff": str(final_cutoff.date()),
            **{k: model[k] for k in ["features", "required_cot_features", "cot_inclusion_audit", "train_rows", "train_last_date", "train_last_target_date", "warnings"]}})
    dump(out/"fit_audit.json", fits)
    dump(out/"cot_inclusion_audit.json", {"dataset": dataset_audit,
        "required_policy": cot_policy == "all", "fit_count": len(fits),
        "all_required_fields_retained_every_fit": all(
            set(metadata["cot_feature_names"]).issubset(fit["features"]) for fit in fits)
            if cot_policy == "all" else None,
        "fits": [{"phase": fit["phase"], "candidate": fit["candidate"],
            "cutoff": fit["cutoff"], "required_cot_features": len(fit["required_cot_features"]),
            "empty_cot_training_columns": sum(item["empty_training_column"] for item in fit["cot_inclusion_audit"]),
            "constant_observed_cot_columns": sum(item["unique_observed_values"] == 1 for item in fit["cot_inclusion_audit"])}
            for fit in fits]})
    bundle = {"horizon": horizon, "unit": "calendar", "fitted_as_of": str(frame.Date.max().date()),
        "bundle_version": 2, "cot_policy": cot_policy,
        "required_cot_features": metadata["cot_feature_names"] if cot_policy == "all" else [],
        "cot_feature_schema_sha256": metadata["cot_feature_schema_sha256"],
        "cot_source_metadata": metadata.get("cot", {}),
        "cot_source_policy": cot_source_policy(metadata),
        "recipes": final_recipes, "members": final_members, "abstention": abstention,
        "features": groups, "source_hashes": source_hashes(), "holdout_start": str(cutoff.date())}
    joblib.dump(bundle, out/"model.joblib")
    latest = frame.tail(1)
    forecasts = []
    for name, recipe in final_recipes.items():
        value = float(predict_recipe(recipe, final_members, latest)[0])
        forecasts.append({"model": name, "task": recipe["task"], "as_of_date": frame.Date.max(),
            "target_date_estimate": frame.Date.max()+pd.Timedelta(days=horizon), "horizon_days": horizon,
            "value": value, "probability_up": value if recipe["task"] == "class" else None,
            "predicted_return": value if recipe["task"] == "reg" else None,
            "direction": "Up" if value >= (recipe["threshold"] if recipe["task"] == "class" else 0) else "Down",
            "threshold": recipe["threshold"] if recipe["task"] == "class" else 0,
            "cached_close": latest.Close.iloc[0], "data_status": "Latest cached origin; not a current market quote"})
    pd.DataFrame(forecasts).to_csv(out/"latest_forecast.csv", index=False)
    summary = {"horizon": horizon, "unit": "calendar", "cot_policy": cot_policy,
        "training_dataset_audit": dataset_audit, "feature_counts": metadata["feature_counts"],
        "candidate_count": len(specs), "cv_fits": len(specs)*len(parts), "holdout_fits": len(needed)*len(holdparts),
        "cv_rows": len(oof), "holdout_rows": len(hold), "recipes": final_recipes,
        "selected_direction": holdclass.loc[holdclass.model.eq("selected_direction")].iloc[0].to_dict(),
        "baseline_monthly_direction": baseline.to_dict(), "selective_policy": selective,
        "selected_return": holdreg.loc[holdreg.model.eq("selected_return")].iloc[0].to_dict(),
        "changes": changes, "bootstrap_60": bootstrap["60"], "selection_status": selection["holdout_status"],
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
        "source_hashes": source_hashes(), "elapsed_seconds": time.perf_counter()-started}
    dump(out/"metrics.json", summary)
    print(f"[{key}] Complete: direction macroF1 {summary['selected_direction']['macro_f1']:.4f}; {summary['elapsed_seconds']:.0f}s", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", nargs="+", type=int, default=[30, 28])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cot-policy", choices=["all", "benchmark"], default="all",
                        help="all requires every numeric COT field in every fit; benchmark retains original curated group selection in a new output directory")
    args = parser.parse_args()
    if not set(args.horizons).issubset({28, 30}):
        parser.error("This comparison requires existing 28/30calendar benchmark artifacts")
    if args.jobs > 1 and len(args.horizons) > 1:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(args.horizons))) as pool:
            results = list(pool.map(train, args.horizons, [args.quick]*len(args.horizons),
                                    [args.cot_policy]*len(args.horizons)))
    else:
        results = [train(h, args.quick, args.cot_policy) for h in args.horizons]
    if not args.quick:
        dump(artifact_directory(args.horizons[0], False, args.cot_policy).parent/"summary.json", results)


if __name__ == "__main__":
    main()
