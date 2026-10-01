"""Compare fixed monthly targets and save a common-date horizon summary."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from train import MONTHFU, parse_args, train
from modeling import metrics
import pandas as pd

HORIZONS = [(30,"calendar"),(21,"sessions"),(28,"calendar"),(21,"calendar"),(30,"sessions")]


def run_one(horizon,unit,quick,news,holdout_start,cot_policy="all"):
    family = "all_cot" if cot_policy == "all" else "benchmark_updated"
    args = argparse.Namespace(horizon=horizon,horizon_unit=unit,holdout_start=holdout_start,
        cv_splits=4,cv_test_size=504,include_experimental_news=news,quick=quick,
        cot_policy=cot_policy,
        output_dir=MONTHFU / "artifacts" / family / ("quick" if quick else "") / f"{horizon}{unit}")
    return train(args)


def summarize(artifact_root=None,report_name="RESULTS.md"):
    artifact_root = artifact_root or MONTHFU / "artifacts"
    summaries, predictions = {},{}
    for horizon,unit in HORIZONS:
        key = f"{horizon}{unit}"
        folder = artifact_root / key
        if (folder / "metrics.json").exists():
            summaries[key] = json.loads((folder / "metrics.json").read_text())
            predictions[key] = pd.read_csv(folder / "holdout_predictions.csv",parse_dates=["Date"])
    common_dates = set.intersection(*(set(p.Date) for p in predictions.values()))
    rows = []
    for key,summary in summaries.items():
        p = predictions[key].loc[predictions[key].Date.isin(common_dates)]
        selected,previous,zero,retuned = (metrics(p.target_return,p[name]) for name in ["selected","previous_transferred_ensemble","zero","previous_cv_ensemble"])
        rows.append({"horizon":summary["horizon"],"unit":summary["horizon_unit"],"experiment":key,
            "primary":key=="30calendar","cv_skill_vs_zero":summary["cv_selected"]["r2_vs_zero"],
            "cv_rmse":summary["cv_selected"]["rmse"],"cv_previous_rmse":summary["cv_previous_ensemble"]["rmse"],
            "cv_rmse_improvement_vs_previous":1-summary["cv_selected"]["rmse"]/summary["cv_previous_ensemble"]["rmse"],
            "common_holdout_rows":len(p),"holdout_rmse":selected["rmse"],"previous_rmse":previous["rmse"],
            "zero_rmse":zero["rmse"],"holdout_skill_vs_zero":selected["r2_vs_zero"],
            "holdout_rmse_improvement_vs_previous":1-selected["rmse"]/previous["rmse"],
            "retuned_previous_rmse":retuned["rmse"],
            "holdout_rmse_improvement_vs_retuned_previous":1-selected["rmse"]/retuned["rmse"],
            "direction_accuracy":selected["direction_accuracy"],
            "always_up_accuracy":float((p.target_return>0).mean()),
            "training_majority_accuracy":float((p.target_return.apply(lambda r:1 if r>0 else -1 if r<0 else 0)==p.training_majority_direction).mean()),
            "recipe":summary["selection"]["name"]})
    table = pd.DataFrame(rows)
    table.to_csv(artifact_root / "horizon_comparison.csv",index=False)
    forecasts = pd.concat([pd.read_csv(artifact_root / key / "latest_forecast.csv").assign(experiment=key)
                           for key in summaries],ignore_index=True)
    forecasts.to_csv(artifact_root / "latest_forecasts.csv",index=False)
    best = table.sort_values("cv_skill_vs_zero",ascending=False).iloc[0]
    primary = table.loc[table.primary].iloc[0]
    lines = ["# MonthFu results","",
        "The primary target was fixed at 30 calendar days before inspecting the monthly holdout. Alternatives check sensitivity to one month and the weekly COT cycle. Horizon comparisons use common holdout origins and normalized squared-error skill; lower raw RMSE at a shorter horizon does not establish a better model.","",
        f"Primary 30-calendar-day RMSE: {primary.holdout_rmse:.2%}; adapted previous ensemble: {primary.previous_rmse:.2%}; relative RMSE improvement: {primary.holdout_rmse_improvement_vs_previous:+.2%}.","",
        f"Best pre-holdout normalized CV skill among these horizon experiments: `{best.experiment}` ({best.cv_skill_vs_zero:+.2%}). This does not replace the fixed primary horizon.","",
        "| Horizon | CV skill vs zero | Holdout RMSE | Previous RMSE | Retuned previous RMSE | Improvement vs previous | Direction | Always up |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r.experiment} | {r.cv_skill_vs_zero:+.2%} | {r.holdout_rmse:.2%} | {r.previous_rmse:.2%} | {r.retuned_previous_rmse:.2%} | {r.holdout_rmse_improvement_vs_previous:+.2%} | {r.direction_accuracy:.2%} | {r.always_up_accuracy:.2%} |" for r in table.itertuples()]
    lines += ["",f"All horizon holdout comparisons above use {len(common_dates)} common forecast origins. Full per-horizon tables can contain additional dates.","",
        "Previous is an adaptation of the existing RF/Holt-Winters/AR research ensemble to each new target. Its five-day validation weights are preserved, RF/AR are retrained on the new return target, and Holt-Winters forecasts the extended horizon. Basic features approximate the older feature family. It is not the exact saved five-day model.","",
        "COT snapshots are shifted to publication, then the first following observed Coffee C session; changes and extremes are calculated on weekly reports before daily filling. Delayed/corrected/backcast reports receive conservative guards. Weather uses a five-calendar-day availability proxy, and news is excluded by default because retrospective selection cannot be undone by delaying rows.","",
        "The historical holdout has been inspected in earlier project research. Positive improvement is exploratory. Read each horizon's report and moving-block confidence interval before interpreting small differences as reliable forecasting value.","",
        "Latest predictions originate from the last cached price date, not today's date. Empirical bands are not probability guarantees; futures rolls, fees, slippage and an executable strategy are outside this experiment."]
    if "30calendar" in summaries:
        primary_summary = summaries["30calendar"]
        selected = primary_summary["selection"]
        policy = primary_summary.get("cot_policy","benchmark")
        lines += ["",f"Primary selected recipe: `{selected['weights']}`. COT selection policy: `{policy}`; all-COT runs constrain every selected member to receive the complete numeric COT feature schema.","",
            f"Primary 60-session block interval for RMSE difference versus adapted previous: `{primary_summary['block_comparisons']['previous_transferred_ensemble']['rmse_difference_95pct_block_interval']}`; versus zero: `{primary_summary['block_comparisons']['zero']['rmse_difference_95pct_block_interval']}`. The zero-reference interval spans zero, so a reliable edge over no change is not established."]
    (MONTHFU / report_name).write_text("\n".join(lines)+"\n")
    print(table.to_string(index=False),flush=True)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs",type=int,default=2)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--include-experimental-news",action="store_true")
    parser.add_argument("--holdout-start",default="2022-01-01")
    parser.add_argument("--cot-policy",choices=["all","benchmark"],default="all")
    parser.add_argument("--summarize-only",action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 4:
        raise ValueError("Use between one and four experiment workers.")
    if not args.summarize_only:
        if args.include_experimental_news:
            raise ValueError("Run optional news ablations with train.py into a separate MonthFu output directory.")
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_one,h,u,args.quick,False,args.holdout_start,args.cot_policy) for h,u in HORIZONS]
            for future in as_completed(futures):
                future.result()
    family = "all_cot" if args.cot_policy == "all" else "benchmark_updated"
    summarize(MONTHFU / "artifacts" / family / ("quick" if args.quick else ""),
              f"RESULTS_{family.upper()}{'_QUICK' if args.quick else ''}.md")


if __name__=="__main__":
    main()
