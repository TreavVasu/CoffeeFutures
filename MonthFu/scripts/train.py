"""Train monthly Arabica models with pre-holdout selection and quarterly refits."""
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

for key in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","LOKY_MAX_CPU_COUNT"]:
    os.environ.setdefault(key,"1")
os.environ.setdefault("MPLCONFIGDIR","/tmp/monthfu-matplotlib")
ROOT = Path(__file__).resolve().parents[2]
MONTHFU = ROOT / "MonthFu"
sys.path.insert(0,str(MONTHFU / "src"))

import joblib
import numpy as np
import pandas as pd
import sklearn
from features import build_feature_frame
from modeling import (Candidate, PREVIOUS_WEIGHTS, add_targets, block_comparison, candidates,
    expanding_folds, fit_candidate, mature_training_rows, metrics, nonoverlap_positions,
    predict_bundle, predict_member, select_recipe)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon",type=int,default=30)
    parser.add_argument("--horizon-unit",choices=["calendar","sessions"],default="calendar")
    parser.add_argument("--holdout-start",default="2022-01-01")
    parser.add_argument("--cv-splits",type=int,default=4)
    parser.add_argument("--cv-test-size",type=int,default=504)
    parser.add_argument("--include-experimental-news",action="store_true")
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--resume-export",action="store_true",help="Verify saved CV/holdout tables and rebuild bundle/reports without repeating evaluation fits")
    return parser.parse_args()


def fit_recipe(recipe, lookup, frame, groups, cutoff, quick, cache=None):
    cache = {} if cache is None else cache
    members = []
    for name,weight in recipe["weights"].items():
        if name not in cache:
            spec = lookup[name]
            train = mature_training_rows(frame,cutoff,spec.train_years)
            fitted = fit_candidate(spec,train,groups[spec.group],quick,
                                  price_history=frame.loc[frame.Date.lt(cutoff)])
            fitted.update(train_rows=len(train),train_last_date=str(train.Date.max().date()),
                          train_last_target_date=str(train.target_end_date.max().date()))
            if spec.kind == "holt_winters":
                fitted.update(state_price_fit_start=str(frame.Date.min().date()),
                              state_price_fit_end=str(fitted["state_as_of"].date()))
            cache[name] = fitted
        members.append({**cache[name],"weight":weight})
    return {"members":members,"recipe":recipe}


def json_ready(value):
    """Pandas tables turn optional correlations into NaN; JSON records use null."""
    if isinstance(value,dict):
        return {key:json_ready(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):
        return None
    return value


def latest_forecast(bundle,frame):
    latest = frame.tail(1)
    forecast = float(predict_bundle(bundle,latest)[0])
    date, horizon, unit = latest.Date.iloc[0], bundle["horizon"], bundle["horizon_unit"]
    estimate = date + (pd.offsets.BDay(horizon) if unit=="sessions" else pd.Timedelta(days=horizon))
    if unit=="calendar" and estimate.weekday()>=5:
        estimate += pd.offsets.BDay(0)
    band = bundle["cv_absolute_error_q80"]
    return pd.DataFrame([{"as_of_date":date,"fitted_as_of":bundle["fitted_as_of"],
        "horizon":horizon,"horizon_unit":unit,"target_date_estimate":estimate,
        "target_date_policy":"Weekday estimate; exchange holidays and future missing observations not projected",
        "close":float(latest.Close.iloc[0]),"predicted_return":forecast,"predicted_return_pct":100*forecast,
        "implied_close":float(latest.Close.iloc[0])*(1+forecast),
        "cv_empirical_lower_80":forecast-band,"cv_empirical_upper_80":forecast+band,
        "interval_policy":"Selected-CV absolute error band; no prospective coverage guarantee",
        "recipe":bundle["recipe"]["name"],
        "data_status":"Forecast from last cached price close; not a live current-market quote"}])


def train(args,shared_features=None):
    started = time.perf_counter()
    key = f"{args.horizon}{'calendar' if args.horizon_unit=='calendar' else 'sessions'}"
    out = (args.output_dir or MONTHFU / "artifacts" / key).resolve()
    if not out.is_relative_to(MONTHFU):
        raise ValueError("Keep experiment outputs inside MonthFu.")
    out.mkdir(parents=True,exist_ok=True)
    print(f"[{key}] Building targets and availability-aware features",flush=True)
    base,manifest,audits = build_feature_frame(ROOT) if shared_features is None else shared_features
    groups = manifest["groups"]
    frame = add_targets(base,args.horizon,args.horizon_unit)
    metadata = dict(audits["metadata"])
    metadata.update(feature_counts={k:len(v) for k,v in groups.items()},
                    signal_time="After daily Coffee C close; close-to-close research returns")
    cutoff = pd.Timestamp(args.holdout_start)
    # Common validation origins across default horizons; each target purges its own labels.
    anchor = add_targets(base,max(30,args.horizon) if args.horizon_unit=="sessions" else 30,"sessions")
    fold_anchors = expanding_folds(anchor,cutoff,args.cv_splits,args.cv_test_size)
    specs = candidates(args.include_experimental_news,args.quick)
    lookup = {s.name:s for s in specs}
    if getattr(args,"resume_export",False):
        from resume_export import resume_export
        return resume_export(args,out,frame,groups,metadata,audits,lookup,started)
    cv_rows,oof_parts,fold_audit = [],[],[]
    for number,(_,valid_anchor) in enumerate(fold_anchors,1):
        valid = frame.loc[frame.Date.isin(valid_anchor.Date)].copy()
        if valid.target_return.isna().any() or not valid.target_end_date.lt(cutoff).all():
            raise ValueError("All validation labels must mature before the holdout.")
        train_rows = mature_training_rows(frame,valid.Date.iloc[0])
        print(f"[{key}] CV {number}/{len(fold_anchors)}: {len(train_rows)} mature train rows; {valid.Date.min().date()}–{valid.Date.max().date()}",flush=True)
        part = valid[["Date","Close","target_end_date","target_return"]].copy()
        part["fold"] = number
        for spec in specs:
            fit_train = mature_training_rows(frame,valid.Date.iloc[0],spec.train_years)
            fitted = fit_candidate(spec,fit_train,groups[spec.group],args.quick,
                                  price_history=frame.loc[frame.Date.lt(valid.Date.iloc[0])])
            prediction = predict_member(fitted,valid)
            part[spec.name] = prediction
            cv_rows.append({"fold":number,"candidate":spec.name,"feature_group":spec.group,
                            "train_rows":len(fit_train),"features":len(fitted["features"]),
                            **metrics(valid.target_return,prediction)})
        oof_parts.append(part)
        fold_audit.append({"fold":number,"train_start":train_rows.Date.min(),"train_end":train_rows.Date.max(),
            "latest_training_label_end":train_rows.target_end_date.max(),"validation_start":valid.Date.min(),
            "validation_end":valid.Date.max(),"latest_validation_label_end":valid.target_end_date.max(),
            "holdout_boundary":cutoff})
    oof = pd.concat(oof_parts,ignore_index=True)
    previous_recipe,_ = select_recipe(oof,[lookup[n] for n in PREVIOUS_WEIGHTS])
    previous_recipe["name"] = "previous_cv_ensemble"
    recipe,ranking = select_recipe(oof,specs)
    if previous_recipe["cv_rmse"] < recipe["cv_rmse"]:
        recipe = previous_recipe.copy()
    oof["previous_cv_ensemble"] = sum(oof[n]*w for n,w in previous_recipe["weights"].items())
    ranking = pd.concat([ranking,pd.DataFrame([{"recipe":"previous_cv_ensemble",
        "weights":previous_recipe["weights"],**metrics(oof.target_return,oof.previous_cv_ensemble)}])],ignore_index=True).sort_values("rmse")
    oof["previous_transferred_ensemble"] = sum(oof[n]*w for n,w in PREVIOUS_WEIGHTS.items())
    oof["selected"] = sum(oof[n]*w for n,w in recipe["weights"].items())
    error_q80 = float(np.quantile(np.abs(oof.target_return-oof.selected),.80))
    cv_previous = metrics(oof.target_return,oof.previous_transferred_ensemble)
    print(f"[{key}] Frozen CV recipe: {recipe}",flush=True)
    selection = {"recipe":recipe,"previous_cv_recipe":previous_recipe,"candidates":[asdict(s) for s in specs],
                 "holdout_start":str(cutoff.date()),"refit_policy":"Quarterly; only labels ending strictly before refit",
                 "primary_metric":"Pooled expanding-CV RMSE; horizons compared by skill vs zero",
                 "horizon_selection":"30 calendar days fixed as primary before holdout; alternatives are sensitivity experiments"}
    (out / "selection.json").write_text(json.dumps(selection,indent=2))
    holdout = frame.loc[frame.Date.ge(cutoff) & frame.target_return.notna()].copy()
    if len(holdout)<60:
        raise ValueError("Holdout too small.")
    recipes = {"selected":recipe,"zero":{"name":"zero","weights":{"zero":1}},
        "historical_mean":{"name":"historical_mean","weights":{"historical_mean":1}},
        "previous_transferred_ensemble":{"name":"previous_transferred_ensemble","weights":PREVIOUS_WEIGHTS},
        "previous_cv_ensemble":previous_recipe,
        "compact_legacy":{"name":"compact_legacy","weights":{"compact_legacy":1}}}
    holdout_parts,refit_rows = [],[]
    for period,block in holdout.groupby(holdout.Date.dt.to_period("Q"),sort=True):
        block_start = block.Date.iloc[0]
        print(f"[{key}] Walk-forward {period}",flush=True)
        part = block[["Date","Close","target_end_date","target_return","target_close","target_sessions"]].copy()
        cache = {}
        for name,this_recipe in recipes.items():
            bundle = fit_recipe(this_recipe,lookup,frame,groups,block_start,args.quick,cache)
            part[name] = predict_bundle(bundle,block)
            for member in bundle["members"]:
                refit_rows.append({"quarter":str(period),"comparison":name,"member":member["spec"]["name"],
                    "weight":member["weight"],"first_prediction_date":block_start,"train_rows":member["train_rows"],
                    "train_last_date":member["train_last_date"],"train_last_target_date":member["train_last_target_date"],
                    "state_price_fit_start":member.get("state_price_fit_start"),
                    "state_price_fit_end":member.get("state_price_fit_end")})
        past = mature_training_rows(frame,block_start)
        majority = float(np.sign(np.sign(past.target_return).mean()))
        part["training_majority_direction"] = majority
        part["always_up_direction"] = 1.
        part["empirical_lower_80"],part["empirical_upper_80"] = part.selected-error_q80,part.selected+error_q80
        holdout_parts.append(part)
    predictions = pd.concat(holdout_parts,ignore_index=True)
    comparison = pd.DataFrame([{"model":name,**metrics(predictions.target_return,predictions[name])} for name in recipes])
    direction_baselines = {name:float((np.sign(predictions.target_return)==predictions[name]).mean())
                          for name in ["training_majority_direction","always_up_direction"]}
    nonoverlap_rows = []
    for offset in range(30):
        sample = predictions.iloc[nonoverlap_positions(predictions,offset)]
        for name in recipes:
            nonoverlap_rows.append({"offset":offset,"model":name,**metrics(sample.target_return,sample[name])})
    latest_bundle = fit_recipe(recipe,lookup,frame,groups,frame.Date.max()+pd.Timedelta(days=1),args.quick)
    latest_bundle.update(horizon=args.horizon,horizon_unit=args.horizon_unit,cv_absolute_error_q80=error_q80,
        feature_metadata=metadata,fitted_as_of=str(frame.Date.max().date()),
        status="experimental_news" if args.include_experimental_news else "research",bundle_version=1)
    latest = latest_forecast(latest_bundle,frame)
    comparisons = {name:block_comparison(predictions.target_return,predictions.selected,predictions[name],block=60)
                   for name in ["zero","historical_mean","previous_transferred_ensemble","previous_cv_ensemble","compact_legacy"]}
    sensitivity = {str(b):block_comparison(predictions.target_return,predictions.selected,predictions.zero,block=b) for b in [30,90]}
    inputs = [ROOT / p for p in ["data/yahoo/arabica_coffee_futures_history.csv","data/COT/coffee_c_all_cot_data.csv",
        "data/weather/open_meteo_coffee_regions_daily.csv","data/COT/cot_release_overrides.csv","MonthFu/data/cot_release_overrides.csv",
        "Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_daily_summary.csv",
        "Scripts/project/artifacts/outputs/gdelt_coffee_events_2000_2026_weekly_summary.csv"]]
    summary = {"horizon":args.horizon,"horizon_unit":args.horizon_unit,"selection":recipe,"cv_rows":len(oof),
        "cv_folds":len(fold_anchors),"candidate_count":len(specs),"cv_previous_ensemble":cv_previous,
        "cv_previous_retuned":metrics(oof.target_return,oof.previous_cv_ensemble),
        "cv_selected":metrics(oof.target_return,oof.selected),"cv_zero":metrics(oof.target_return,oof.zero),
        "holdout_start":str(predictions.Date.min().date()),"holdout_end":str(predictions.Date.max().date()),
        "holdout_refit_policy":"Quarterly with recipe frozen on pre-2022 expanding CV",
        "holdout_metrics":comparison.to_dict(orient="records"),"direction_baselines":direction_baselines,
        "block_comparisons":comparisons,"zero_bootstrap_block_sensitivity":sensitivity,
        "empirical_80pct_band_holdout_coverage":float(((predictions.target_return>=predictions.empirical_lower_80)&
                                                      (predictions.target_return<=predictions.empirical_upper_80)).mean()),
        "cv_absolute_error_q80":error_q80,"data":metadata,
        "input_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs if p.exists()},
        "code_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(MONTHFU.rglob('*.py'))},
        "versions":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"sklearn":sklearn.__version__},
        "limitations":["Historical holdout has been reused in project research; results are exploratory.",
            "Previous ensemble is a method adaptation: five-day weights transferred, monthly RF/AR re-estimated on basic features, HW horizon extended; not the exact old saved model.",
            "COT historical releases include estimates and correction guards; no complete original-vintage archive.",
            "Weather cache is retrospective reanalysis; five-calendar-day lag does not remove future revisions.",
            "News excluded from primary search; even delayed summaries retain retrospective selection bias.",
            "Daily monthly labels overlap; block intervals and non-overlap samples account partly for dependence.",
            "Continuous contract rolls, price quality, fees and execution are not modeled.",
            "Latest forecast begins at last cached price date, with an approximate projected target date.",
            "Selected-CV error bands are exploratory and do not guarantee future coverage."],
        "elapsed_seconds":round(time.perf_counter()-started,2)}
    audit_table = audits.get("cot_release_audit",audits.get("cot_audit",pd.DataFrame()))
    yearly = pd.DataFrame([{"year":int(year),"model":name,**metrics(p.target_return,p[name])}
        for year,p in predictions.groupby(predictions.Date.dt.year) for name in recipes])
    for name,data in [("cv_folds",pd.DataFrame(fold_audit)),("cv_metrics",pd.DataFrame(cv_rows)),("cv_predictions",oof),
        ("cv_ranking",ranking.assign(weights=ranking.weights.map(lambda w:json.dumps(w,sort_keys=True)))),
        ("holdout_predictions",predictions),("holdout_metrics",comparison),("holdout_by_year",yearly),("holdout_refits",pd.DataFrame(refit_rows)),
        ("nonoverlapping_metrics",pd.DataFrame(nonoverlap_rows)),("latest_forecast",latest),("cot_release_audit",audit_table)]:
        data.to_csv(out / f"{name}.csv",index=False)
    pd.DataFrame([{"group":g,"feature":c} for g,cols in groups.items() for c in cols]).to_csv(out / "feature_manifest.csv",index=False)
    age_cols = [c for c in frame if c.startswith(("cot_","weather_","news_")) and any(s in c for s in ["age","stale","available_date","report_date","publication_date","source_date"])]
    frame[["Date",*age_cols]].to_csv(out / "availability_rows.csv",index=False)
    (out / "metrics.json").write_text(json.dumps(json_ready(summary),indent=2,default=str,allow_nan=False))
    joblib.dump(latest_bundle,out / "model.joblib",compress=3)
    write_report(out,summary,latest)
    write_plot(out,predictions,comparison,key)
    print(f"[{key}] " + comparison[["model","rmse","direction_accuracy","r2_vs_zero"]].to_string(index=False),flush=True)
    print(f"[{key}] Saved {out}; {summary['elapsed_seconds']} seconds",flush=True)
    return summary


def write_report(out,summary,latest):
    selected = next(r for r in summary["holdout_metrics"] if r["model"]=="selected")
    previous = next(r for r in summary["holdout_metrics"] if r["model"]=="previous_transferred_ensemble")
    gain = 1-selected["rmse"]/previous["rmse"]
    text = [f"# Arabica {summary['horizon']} {summary['horizon_unit']} day return experiment","",
        f"Frozen CV recipe: `{summary['selection']['name']}`, weights `{summary['selection']['weights']}`.","",
        f"Holdout RMSE change versus adapted previous ensemble: {gain:+.2%} improvement. Skill versus zero squared error: {selected['r2_vs_zero']:+.2%}.","",
        f"Origins {summary['holdout_start']}–{summary['holdout_end']}. Quarterly fits use only labels ending before the quarter's first origin.","",
        "| Model | RMSE | MAE | Direction | Skill vs zero |","|---|---:|---:|---:|---:|"]
    text += [f"| {r['model']} | {r['rmse']:.2%} | {r['mae']:.2%} | {r['direction_accuracy']:.2%} | {r['r2_vs_zero']:+.2%} |" for r in summary["holdout_metrics"]]
    text += ["",f"Training-only majority-direction accuracy: {summary['direction_baselines']['training_majority_direction']:.2%}. Zero is neutral, so its direction accuracy is not a binary classifier benchmark.","",
        f"Selected-minus-previous RMSE 95% interval (60-session blocks): {summary['block_comparisons']['previous_transferred_ensemble']['rmse_difference_95pct_block_interval']}.","",
        f"Selected-minus-zero RMSE 95% interval: {summary['block_comparisons']['zero']['rmse_difference_95pct_block_interval']}. An interval spanning zero does not establish a reliable improvement over the no-change forecast.","",
        f"Latest cached origin: {latest.as_of_date.iloc[0].date()}, predicted return {latest.predicted_return_pct.iloc[0]:+.2f}%, target approximately {latest.target_date_estimate.iloc[0].date()}. This is a cached-data forecast.","",
        "See CSVs for label maturity, release alignment, every candidate's CV predictions, quarterly refits, non-overlapping samples and the complete feature manifest.","","## Limitations",""]
    text += [f"- {item}" for item in summary["limitations"]]
    (out / "report.md").write_text("\n".join(text)+"\n")


def write_plot(out,predictions,comparison,key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,1,figsize=(13,8),gridspec_kw={"height_ratios":[2,1]})
    axes[0].plot(predictions.Date,100*predictions.target_return,color="#8596a3",alpha=.7,label="Observed return")
    axes[0].plot(predictions.Date,100*predictions.selected,color="#9b602b",label="CV-selected forecast")
    axes[0].axhline(0,color="black",linewidth=.5)
    axes[0].set_ylabel("Return (%)")
    axes[0].legend(loc="upper left")
    axes[0].set_title(f"Arabica {key}: quarterly walk-forward evaluation")
    axes[1].barh(comparison.model,100*comparison.rmse,color="#8596a3")
    axes[1].set_xlabel("RMSE in percentage points; lower is better")
    fig.tight_layout()
    fig.savefig(out / "holdout.png",dpi=150)
    plt.close(fig)


if __name__=="__main__":
    train(parse_args())
