"""Recover completed evaluation checkpoints after export interruption.

Labels, predictions and all selection choices are loaded from completed CSVs;
no holdout retuning occurs. Refit only the saved selected recipe for inference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn


def resume_export(args,out,frame,groups,metadata,audits,lookup,started):
    from train import ROOT, MONTHFU, fit_recipe, latest_forecast, json_ready, write_report, write_plot
    from modeling import block_comparison, metrics
    decision = json.loads((out / "selection.json").read_text())
    if decision["candidates"] != [member.__dict__ for member in lookup.values()]:
        raise ValueError("Saved candidate configuration differs from requested export configuration.")
    recipe = decision["recipe"]
    cv = pd.read_csv(out / "cv_predictions.csv",parse_dates=["Date","target_end_date"])
    scores = pd.read_csv(out / "holdout_predictions.csv",parse_dates=["Date","target_end_date"])
    # Check true labels and endpoints against current cache before preserving results.
    for data in [cv,scores]:
        check = data.merge(frame[["Date","target_return","target_end_date"]],on="Date",suffixes=("","_rebuilt"),validate="one_to_one")
        if len(check)!=len(data) or not check.target_end_date.eq(check.target_end_date_rebuilt).all():
            raise ValueError("Checkpoint target calendar differs from source cache.")
        np.testing.assert_allclose(check.target_return,check.target_return_rebuilt,rtol=1e-12,atol=1e-12)
    rebuilt_cv = sum(cv[n]*w for n,w in recipe["weights"].items())
    np.testing.assert_allclose(cv.selected,rebuilt_cv,rtol=1e-12,atol=1e-12)
    error_q80 = float(np.quantile(np.abs(cv.target_return-cv.selected),.80))
    bundle = fit_recipe(recipe,lookup,frame,groups,frame.Date.max()+pd.Timedelta(days=1),args.quick)
    bundle.update(horizon=args.horizon,horizon_unit=args.horizon_unit,cv_absolute_error_q80=error_q80,
        feature_metadata=metadata,fitted_as_of=str(frame.Date.max().date()),
        status="experimental_news" if args.include_experimental_news else "research",bundle_version=1)
    latest = latest_forecast(bundle,frame)
    saved_latest = pd.read_csv(out / "latest_forecast.csv")
    np.testing.assert_allclose(latest.predicted_return,saved_latest.predicted_return,rtol=1e-12,atol=1e-12)
    models = ["selected","zero","historical_mean","previous_transferred_ensemble","previous_cv_ensemble","compact_legacy"]
    comparison = pd.DataFrame([{"model":name,**metrics(scores.target_return,scores[name])} for name in models])
    refits = pd.read_csv(out / "holdout_refits.csv",parse_dates=["first_prediction_date","train_last_target_date"])
    if not refits.train_last_target_date.lt(refits.first_prediction_date).all():
        raise ValueError("Checkpoint supervised labels cross a forecast boundary.")
    folds = pd.read_csv(out / "cv_folds.csv",parse_dates=["latest_training_label_end","validation_start","latest_validation_label_end","holdout_boundary"])
    if not (folds.latest_training_label_end.lt(folds.validation_start).all() and folds.latest_validation_label_end.lt(folds.holdout_boundary).all()):
        raise ValueError("Checkpoint validation boundary is not purged.")
    # Make unsupervised state-fitting price history explicit alongside mature labels.
    state_mask = refits.member.eq("previous_hw")
    prior_session = refits.loc[state_mask,"first_prediction_date"].map(lambda cutoff:frame.loc[frame.Date.lt(cutoff),"Date"].max())
    refits.loc[state_mask,"state_price_fit_end"] = prior_session
    refits.loc[state_mask,"state_price_fit_start"] = frame.Date.min()
    refits.to_csv(out / "holdout_refits.csv",index=False)
    source_paths = [ROOT / p for p in ["data/yahoo/arabica_coffee_futures_history.csv","data/COT/coffee_c_all_cot_data.csv",
        "data/weather/open_meteo_coffee_regions_daily.csv","data/COT/cot_release_overrides.csv","MonthFu/data/cot_release_overrides.csv",
        "Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_daily_summary.csv",
        "Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_weekly_summary.csv"]]
    summary = {"horizon":args.horizon,"horizon_unit":args.horizon_unit,"selection":recipe,"cv_rows":len(cv),
        "cv_folds":len(folds),"candidate_count":len(lookup),"cv_previous_ensemble":metrics(cv.target_return,cv.previous_transferred_ensemble),
        "cv_previous_retuned":metrics(cv.target_return,cv.previous_cv_ensemble),"cv_selected":metrics(cv.target_return,cv.selected),
        "cv_zero":metrics(cv.target_return,cv.zero),"holdout_start":str(scores.Date.min().date()),"holdout_end":str(scores.Date.max().date()),
        "holdout_refit_policy":"Quarterly with recipe frozen on pre-2022 expanding CV","holdout_metrics":comparison.to_dict(orient="records"),
        "direction_baselines":{name:float((np.sign(scores.target_return)==scores[name]).mean()) for name in ["training_majority_direction","always_up_direction"]},
        "block_comparisons":{name:block_comparison(scores.target_return,scores.selected,scores[name],block=60)
            for name in ["zero","historical_mean","previous_transferred_ensemble","previous_cv_ensemble","compact_legacy"]},
        "zero_bootstrap_block_sensitivity":{str(b):block_comparison(scores.target_return,scores.selected,scores.zero,block=b) for b in [30,90]},
        "empirical_80pct_band_holdout_coverage":float(((scores.target_return>=scores.empirical_lower_80)&(scores.target_return<=scores.empirical_upper_80)).mean()),
        "cv_absolute_error_q80":error_q80,"data":metadata,
        "input_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths if p.exists()},
        "code_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(MONTHFU.rglob('*.py'))},
        "versions":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"sklearn":sklearn.__version__},
        "limitations":["Historical holdout reused in prior project research; all gains are exploratory.",
            "Previous ensemble is an adaptation of old methods to monthly targets and basic features, not the exact saved five-day model.",
            "Historical COT publication dates include estimates/correction guards, with no complete original-vintage archive.",
            "Weather is retrospective reanalysis; the five-day lag does not remove revisions.",
            "News excluded by default because delaying summaries cannot undo retrospective selection bias.",
            "Daily monthly labels overlap; use block intervals and true-date non-overlap samples.",
            "Continuous contract rolls, fees, slippage and an execution strategy are not modeled.",
            "Latest forecast uses the last cached price date with an estimated future session date.",
            "Empirical bands use selected-CV residuals; no prospective coverage guarantee.",
            "Some non-selected Elastic Net fits hit the iteration limit; these candidates are exploratory and not in any selected recipe."],
        "checkpoint_export":{"recovered_after":"Strict JSON export rejected undefined neutral-forecast correlations",
            "policy":"Verified saved labels, selected CV recipe and maturity boundaries; retained all evaluation predictions, refitted only inference weights",
            "original_training_log":"MonthFu/artifacts/training.log"},
        "elapsed_seconds_export_only":round(time.perf_counter()-started,2)}
    (out / "metrics.json").write_text(json.dumps(json_ready(summary),indent=2,default=str,allow_nan=False))
    joblib.dump(bundle,out / "model.joblib",compress=3)
    write_report(out,summary,latest)
    write_plot(out,scores,comparison,out.name)
    print(f"Recovered {out.name}: {recipe['name']}; RMSE {comparison.iloc[0].rmse:.6f}",flush=True)
    return summary
